# SPDX-License-Identifier: Apache-2.0
"""Build the Windows and Linux releases of msk-pipe (.github/workflows/release.yml).

Steps (each a subcommand, run from the repository root; ``--platform`` win-64 or linux-64,
default: the platform the script runs on)::

    python packaging/build_release.py tools --out build/tools        # pixi-pack, pixi-unpack
    python packaging/build_release.py wheels --out build/wheels      # non-wheel packages
    build/tools/pixi-pack ... --inject build/wheels/*.whl            # environment.tar
    python packaging/build_release.py atlas --out build/assets       # pinned atlas asset
    python packaging/build_release.py maxpath --root build/test      # longest path (Windows)
    python packaging/build_release.py assemble --version 0.1.0 ...   # dist/ archive + sums
    python packaging/build_release.py sums --file SHA256SUMS.txt F   # add F to the list

The release folder (``msk-pipe-<version>-win64.zip``, ``...-linux-x86_64.tar.gz``)::

    msk-pipe-<version>/
        msk-pipe.cmd, mskpipe.cmd     GUI (double-click) and command line; Linux: msk-pipe,
                                      mskpipe (shell scripts)
        environment.tar               pixi-pack of the "cpu" environment
        bin/                          pixi-unpack and the setup script (_setup.cmd with the
                                      VC++ runtime DLLs and maxpath.txt; Linux: _setup.sh
                                      with cacert.pem)
        atlas/lhdl/                   attachment atlas (own licence, CC BY-NC-SA 2.0 BE)
        licenses/                     licences of every bundled package
        LICENSE, NOTICE, THIRD_PARTY_LICENSES.md, CITATION.cff, README.txt, version.txt

Only the standard library is used, except ``wheels`` (reads pixi.lock with PyYAML and
builds with pip).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
PINS = PACKAGING / "release.toml"
LAUNCHER = PACKAGING / "launcher"
DOCS = ("LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md", "CITATION.cff")
# pixi-unpack.exe (Rust, MSVC) needs the Visual C++ runtime, which a clean Windows does not
# have; it runs before the environment exists, so its DLLs are put next to it (app-local,
# taken from the environment's own conda package vc14_runtime, licence in licenses/).
RUNTIME_DLLS = ("vcruntime140.dll", "vcruntime140_1.dll")
# Linux: pixi-unpack (static) sets up a TLS client although it downloads nothing and stops
# on a system without CA certificates (minimal containers); it gets the bundle of the
# environment's own conda package ca-certificates (licence in licenses/).
CA_BUNDLE = PurePosixPath("ssl/cacert.pem")
MAX_ASSET_BYTES = 2 * 2**30 - 2**20  # GitHub release assets must be < 2 GiB
STORED_SUFFIXES = (".tar", ".exe", ".whl", ".zip")  # already compressed


@dataclass(frozen=True)
class Target:
    """What differs between the release packages."""

    platform: str  # pixi platform
    asset: str  # msk-pipe-<version>-<asset>
    launcher: Path  # folder with the start scripts, README.txt and bin/
    scripts: tuple[str, ...]  # start scripts in the release root
    exe: str  # suffix of executables
    crlf: bool  # line ends of the scripts and text files


TARGETS = {
    "win-64": Target(
        "win-64", "win64.zip", LAUNCHER, ("msk-pipe.cmd", "mskpipe.cmd"), ".exe", True
    ),
    "linux-64": Target(
        "linux-64", "linux-x86_64.tar.gz", LAUNCHER / "linux", ("msk-pipe", "mskpipe"), "", False
    ),
}


def host_platform() -> str:
    return "win-64" if sys.platform == "win32" else "linux-64"


class BuildError(RuntimeError):
    """The release cannot be built."""


def load_pins(path: Path = PINS) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected: str) -> Path:
    """Download ``url`` (if not already there with the right checksum) and check it."""
    if not expected:
        raise BuildError(f"No SHA-256 pinned for {url}")
    if dest.is_file() and sha256(dest) == expected:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as f:
        shutil.copyfileobj(response, f, 1 << 20)
    digest = sha256(part)
    if digest != expected:
        part.unlink()
        raise BuildError(f"Checksum mismatch for {url}: expected {expected}, got {digest}")
    part.replace(dest)
    return dest


# ---------------------------------------------------------------- wheels


def wheel_sources(lock: Mapping[str, Any], env: str, platform: str) -> list[str]:
    """PyPI packages of a locked environment that are not wheels (pixi-pack cannot pack
    them), as pip requirement strings: the project itself, git packages, sdists."""
    out = []
    for entry in lock["environments"][env]["packages"][platform]:
        url = entry.get("pypi")
        if url is None or url.endswith(".whl"):
            continue
        if url in (".", "./"):
            out.append(str(ROOT))
        elif url.startswith("git+"):
            parsed = urlparse(url[4:])
            rev = dict(p.split("=", 1) for p in parsed.query.split("&") if "=" in p).get("rev")
            rev = rev or parsed.fragment
            if not rev:
                raise BuildError(f"Git package without a pinned revision: {url}")
            base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            out.append(f"git+{base}@{rev}")
        else:
            out.append(url)
    return out


def git_packages(lock: Mapping[str, Any], env: str, platform: str) -> dict[str, str]:
    """Distribution name (lower case) -> pinned commit of the git packages."""
    pypi = {p["pypi"]: p for p in lock["packages"] if "pypi" in p}
    out = {}
    for entry in lock["environments"][env]["packages"][platform]:
        url = entry.get("pypi", "")
        if url.startswith("git+") and url in pypi:
            parsed = urlparse(url[4:])
            rev = dict(p.split("=", 1) for p in parsed.query.split("&") if "=" in p).get("rev")
            out[pypi[url]["name"].lower()] = rev or parsed.fragment
    return out


def build_info(lock: Mapping[str, Any], env: str, platform: str) -> dict[str, Any]:
    """Content of ``src/mskpipe/build_info.json`` (see :mod:`mskpipe.core.build_info`)."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {"commit": head, "packages": git_packages(lock, env, platform)}


def build_wheels(out: Path, lock_path: Path, env: str, platform: str) -> list[Path]:
    import yaml

    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    sources = wheel_sources(lock, env, platform)
    out.mkdir(parents=True, exist_ok=True)
    info_file = ROOT / "src" / "mskpipe" / "build_info.json"
    info_file.write_text(json.dumps(build_info(lock, env, platform), indent=2), encoding="utf-8")
    argv = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir", "-w", str(out)]
    try:
        subprocess.run([*argv, *sources], check=True)
    finally:
        info_file.unlink(missing_ok=True)
    wheels = sorted(out.glob("*.whl"))
    if len(wheels) != len(sources):
        raise BuildError(f"Built {len(wheels)} wheels for {len(sources)} sources: {sources}")
    for wheel in wheels:
        print(wheel.name)
    return wheels


def wheel_version(wheels: Path, name: str = "mskpipe") -> str:
    """Version of ``name`` from its wheel file name (hatch-vcs: from the git tag)."""
    for wheel in wheels.glob(f"{name}-*.whl"):
        return wheel.name.split("-")[1]
    raise BuildError(f"No {name} wheel in {wheels}")


# ---------------------------------------------------------------- atlas


def fetch_atlas(out: Path, pins: Mapping[str, Any]) -> Path:
    """Download the pinned atlas asset of its GitHub release (``gh``) and check it."""
    atlas = pins["atlas"]
    if not atlas.get("sha256"):
        raise BuildError("[atlas] sha256 is empty in packaging/release.toml")
    out.mkdir(parents=True, exist_ok=True)
    dest = out / atlas["file"]
    if not dest.is_file():
        argv = ["gh", "release", "download", atlas["tag"], "-p", atlas["file"], "-D", str(out)]
        subprocess.run(argv, check=True)
    digest = sha256(dest)
    if digest != atlas["sha256"]:
        raise BuildError(f"Atlas checksum mismatch: expected {atlas['sha256']}, got {digest}")
    return dest


def extract_atlas(archive: Path, dest: Path) -> None:
    """Unpack an atlas zip into ``dest`` (the folder holding atlas.json)."""
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts:
                raise BuildError(f"Unsafe path in {archive.name}: {name}")
        index = [n for n in names if PurePosixPath(n).name == "atlas.json"]
        if len(index) != 1:
            raise BuildError(f"{archive.name}: expected one atlas.json, found {len(index)}")
        prefix = str(PurePosixPath(index[0]).parent)
        prefix = "" if prefix == "." else prefix + "/"
        for name in names:
            if not name.startswith(prefix):
                continue
            target = dest.joinpath(*PurePosixPath(name[len(prefix) :]).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    for required in ("LICENSE", "README.md"):
        if not (dest / required).is_file():
            raise BuildError(f"The atlas must contain {required} (its own licence)")


# ---------------------------------------------------------------- assemble


def ca_bundle(prefix: Path) -> Path:
    """CA certificates of an unpacked environment (conda package ca-certificates)."""
    path = prefix.joinpath(*CA_BUNDLE.parts)
    if not path.is_file() or b"BEGIN CERTIFICATE" not in path.read_bytes():
        raise BuildError(f"No {CA_BUNDLE} in {prefix} (conda package ca-certificates)")
    return path


def runtime_dlls(prefix: Path) -> list[Path]:
    """Visual C++ runtime DLLs of an unpacked environment (root, else Library/bin)."""
    found = []
    for name in RUNTIME_DLLS:
        for folder in (prefix, prefix / "Library" / "bin"):
            match = next((p for p in folder.glob("*") if p.name.lower() == name), None)
            if match is not None:
                found.append(match)
                break
    if not any(p.name.lower() == RUNTIME_DLLS[0] for p in found):
        raise BuildError(f"No {RUNTIME_DLLS[0]} in {prefix} (conda package vc14_runtime)")
    return found


def longest_path(root: Path) -> int:
    """Length of the longest file path below ``root``, relative to it (Windows style)."""
    longest = 0
    for path in root.rglob("*"):
        longest = max(longest, len(str(path.relative_to(root))))
    return longest


def stage(
    folder: Path,
    *,
    version: str,
    pack: Path,
    unpack: Path,
    licenses: Path,
    maxpath: int | None = None,
    atlas: Path | None,
    atlas_folder: str = "lhdl",
    runtime: list[Path] | None = None,
    platform: str = "win-64",
) -> Path:
    """Lay out the release folder ``folder`` (created fresh)."""
    target = TARGETS[platform]
    write = _write_crlf if target.crlf else _write_lf
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "bin").mkdir(parents=True)
    # batch files need CRLF line ends (labels, goto), text files for Notepad too; shell
    # scripts need LF. Neither depends on how git checked the files out.
    for name in target.scripts:
        write(folder / name, (target.launcher / name).read_text(encoding="utf-8"))
    for path in (target.launcher / "bin").iterdir():
        write(folder / "bin" / path.name, path.read_text(encoding="utf-8"))
    readme = target.launcher / "README.txt"
    write(folder / "README.txt", readme.read_text(encoding="utf-8").replace("{version}", version))
    write(folder / "version.txt", version + "\n")
    if maxpath is not None:
        write(folder / "bin" / "maxpath.txt", f"{maxpath}\n")
    shutil.copy2(unpack, folder / "bin" / f"pixi-unpack{target.exe}")
    for path in runtime or ():  # files pixi-unpack needs before the environment exists
        shutil.copy2(path, folder / "bin" / path.name.lower())
    shutil.copy2(pack, folder / "environment.tar")
    for name in DOCS:
        shutil.copy2(ROOT / name, folder / name)
    shutil.copytree(licenses, folder / "licenses")
    if atlas is not None:
        extract_atlas(atlas, folder / "atlas" / atlas_folder)
    return folder


def executables(platform: str) -> tuple[str, ...]:
    """Files of a POSIX release folder that must be executable (relative, POSIX style)."""
    target = TARGETS[platform]
    return () if target.crlf else (*target.scripts, "bin/pixi-unpack")


def _write_crlf(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))


def _write_lf(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def zip_folder(folder: Path, archive: Path, limit: int = MAX_ASSET_BYTES) -> Path:
    """Zip ``folder`` (as its top-level directory); compressed files are stored."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", allowZip64=True) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_dir():
                continue
            mode = zipfile.ZIP_STORED if path.suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            zf.write(path, path.relative_to(folder.parent).as_posix(), compress_type=mode)
    size = archive.stat().st_size
    if size > limit:
        raise BuildError(f"{archive.name} has {size / 2**30:.2f} GiB, GitHub allows < 2 GiB")
    return archive


def tar_folder(
    folder: Path, archive: Path, executable: Iterable[str] = (), limit: int = MAX_ASSET_BYTES
) -> Path:
    """``folder`` (as its top-level directory) as .tar.gz. File modes do not depend on the
    file system the release was staged on: 0755 for ``executable`` (paths relative to
    ``folder``) and directories, 0644 otherwise; no user names. The environment is
    compressed already, so gzip only wraps it (level 1)."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    runnable = {PurePosixPath(p) for p in executable}
    missing = sorted(str(p) for p in runnable if not folder.joinpath(*p.parts).is_file())
    if missing:
        raise BuildError(f"Not in {folder.name}: {missing}")

    def entry(info: tarfile.TarInfo) -> tarfile.TarInfo:
        relative = PurePosixPath(*PurePosixPath(info.name).parts[1:])
        info.mode = 0o755 if info.isdir() or relative in runnable else 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with (
        archive.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf,
    ):
        tf.add(folder, arcname=folder.name, recursive=False, filter=entry)
        for path in sorted(folder.rglob("*")):
            name = (PurePosixPath(folder.name) / path.relative_to(folder).as_posix()).as_posix()
            tf.add(path, arcname=name, recursive=False, filter=entry)
    size = archive.stat().st_size
    if size > limit:
        raise BuildError(f"{archive.name} has {size / 2**30:.2f} GiB, GitHub allows < 2 GiB")
    return archive


def write_sums(files: Iterable[Path], out: Path) -> Path:
    lines = [f"{sha256(p)}  {p.name}" for p in sorted(files)]
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out


def add_sums(files: Iterable[Path], out: Path) -> Path:
    """Add ``files`` to the checksum list ``out`` (created if missing; a file listed
    already gets its new checksum), sorted by file name."""
    sums: dict[str, str] = {}
    if out.is_file():
        for line in out.read_text(encoding="ascii").splitlines():
            digest, _, name = line.partition("  ")
            if name:
                sums[name] = digest
    for path in files:
        sums[path.name] = sha256(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{sums[name]}  {name}\n" for name in sorted(sums))
    out.write_bytes(text.encode("ascii"))
    return out


def assemble(args: argparse.Namespace) -> list[Path]:
    pins = load_pins()
    target = TARGETS[args.platform]
    windows = args.platform == "win-64"
    if windows and not args.maxpath:
        raise BuildError("The Windows release needs --maxpath")
    prefix = Path(args.runtime_from)
    dist = Path(args.out)
    dist.mkdir(parents=True, exist_ok=True)
    name = f"msk-pipe-{args.version}"
    folder = stage(
        Path(args.work) / name,
        version=args.version,
        pack=Path(args.pack),
        unpack=Path(args.unpack),
        licenses=Path(args.licenses),
        maxpath=int(Path(args.maxpath).read_text(encoding="ascii").strip()) if windows else None,
        atlas=Path(args.atlas) if args.atlas else None,
        atlas_folder=pins["atlas"]["folder"],
        runtime=runtime_dlls(prefix) if windows else [ca_bundle(prefix)],
        platform=args.platform,
    )
    archive = dist / f"{name}-{target.asset}"
    if windows:
        outputs = [zip_folder(folder, archive)]
    else:
        outputs = [tar_folder(folder, archive, executables(args.platform))]
    if args.atlas:
        outputs.append(Path(shutil.copy2(args.atlas, dist / Path(args.atlas).name)))
    outputs.append(add_sums(outputs, dist / "SHA256SUMS.txt"))
    for path in outputs:
        print(f"{path} ({path.stat().st_size / 2**20:.1f} MiB)")
    return outputs


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def platform_option(p: argparse.ArgumentParser) -> None:
        p.add_argument("--platform", choices=sorted(TARGETS), default=host_platform())

    p = sub.add_parser("tools", help="download the pinned pixi-pack and pixi-unpack")
    p.add_argument("--out", required=True)
    platform_option(p)

    p = sub.add_parser("wheels", help="build wheels of the non-wheel PyPI packages")
    p.add_argument("--out", required=True)
    p.add_argument("--lock", default=str(ROOT / "pixi.lock"))
    platform_option(p)

    p = sub.add_parser("version", help="print the mskpipe version of the built wheel")
    p.add_argument("--wheels", required=True)

    p = sub.add_parser("atlas", help="download and check the pinned atlas asset")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="no atlas pinned yet: warn and build without it (test builds only)",
    )

    p = sub.add_parser("maxpath", help="longest path below an unpacked release root")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("assemble", help="release folder -> zip or tar.gz + SHA256SUMS")
    p.add_argument("--version", required=True)
    p.add_argument("--pack", required=True, help="environment.tar from pixi-pack")
    p.add_argument("--unpack", required=True, help="pixi-unpack of the platform")
    p.add_argument("--licenses", required=True, help="output of collect_licenses.py")
    p.add_argument("--maxpath", help="output of the maxpath command (Windows)")
    p.add_argument("--atlas", help="atlas zip (omit for a release without atlas)")
    p.add_argument(
        "--runtime-from",
        required=True,
        help="unpacked environment with the files pixi-unpack needs in bin/: the Visual "
        "C++ runtime DLLs (Windows), the CA certificates (Linux)",
    )
    p.add_argument("--work", default="build/stage")
    p.add_argument("--out", default="dist")
    platform_option(p)

    p = sub.add_parser("sums", help="add files to a SHA256SUMS list")
    p.add_argument("--file", required=True, help="the list (created if missing)")
    p.add_argument("files", nargs="+")

    args = parser.parse_args(argv)
    pins = load_pins()
    try:
        if args.command == "tools":
            pp = pins["pixi-pack"][args.platform]
            out, exe = Path(args.out), TARGETS[args.platform].exe
            for tool in ("pack", "unpack"):
                path = fetch(pp[f"{tool}_url"], out / f"pixi-{tool}{exe}", pp[f"{tool}_sha256"])
                path.chmod(0o755)
        elif args.command == "wheels":
            env = pins["environment"]
            if args.platform not in env["platforms"]:
                raise BuildError(f"No release for {args.platform} in packaging/release.toml")
            build_wheels(Path(args.out), Path(args.lock), env["name"], args.platform)
        elif args.command == "version":
            print(wheel_version(Path(args.wheels)))
        elif args.command == "atlas":
            if not pins["atlas"].get("sha256") and args.allow_missing:
                print("WARNING: no atlas pinned in packaging/release.toml; building without it")
                return 0
            print(fetch_atlas(Path(args.out), pins))
        elif args.command == "maxpath":
            value = longest_path(Path(args.root))
            Path(args.out).write_text(f"{value}\n", encoding="ascii")
            print(value)
        elif args.command == "assemble":
            assemble(args)
        elif args.command == "sums":
            print(add_sums([Path(f) for f in args.files], Path(args.file)).read_text("ascii"))
    except BuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
