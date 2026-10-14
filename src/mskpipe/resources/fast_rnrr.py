# SPDX-License-Identifier: Apache-2.0
"""Fast-RNRR built on the user's computer (``mskpipe setup fast-rnrr``).

Fast-RNRR (Yao et al., CVPR 2020, https://github.com/Juyong/Fast_RNRR) is patented and its
authors allow it for research purposes only; the repository has no licence that allows
redistribution. msk-pipe therefore ships neither its source nor a binary: with the user's
explicit consent to the authors' terms, this module downloads the pinned source files
from the authors' repository and builds the executable locally::

    <MSKPIPE_HOME>/fast_rnrr/<commit7>/Fast_RNRR[.exe]   the executable
    <MSKPIPE_HOME>/fast_rnrr/<commit7>/build.json        sources, compiler, check, consent
    <MSKPIPE_HOME>/fast_rnrr/<commit7>/NOTICE.txt        the authors' terms and citation

Everything is pinned with its SHA-256 in ``fast_rnrr_build.json`` (written by
``tools/fast_rnrr_pins.py``): the 20 source files and 3 test files of Fast-RNRR at one
commit (``raw.githubusercontent.com``; the repository's 185 MB of data are not needed),
OpenMesh 9.1 (BSD-3-Clause; the C++ library inside the ``openmesh`` sdist on PyPI), Eigen
3.4.0 (MPL-2.0) and the Zig C/C++ compiler (``ziglang`` wheel on PyPI, MIT), so no Visual
Studio, CMake or other installation is needed. The authors' source is compiled unchanged;
the only addition is a serial ``omp.h`` (Zig has no OpenMP runtime: the program runs on
one thread, ~1.5 s for the 20 000 triangles msk-pipe passes it).

All OpenMesh ``.cc`` files are compiled into the executable instead of a static library:
OpenMesh registers its file readers through static objects that a linker would drop from
a library, and the program would then read empty meshes and exit with code 0. The build
ends with the authors' own test (``data/test``: source, target and their result): the new
executable must reproduce their result, otherwise it is not installed.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from mskpipe.resources.download import download, file_sha256
from mskpipe.resources.paths import home
from mskpipe.resources.tool import Cancelled, Log

FAST_RNRR_ENV = "MSKPIPE_FAST_RNRR"
EXECUTABLE_NAMES = ("Fast_RNRR.exe", "Fast_RNRR")
SPEC_FILE = "fast_rnrr_build.json"
MARKER = "build.json"
NOTICE_FILE = "NOTICE.txt"
ACCEPT_FLAG = "--accept-research-licence"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class FastRnrrSetupError(RuntimeError):
    """Fast-RNRR cannot be built here, the build failed or the consent is missing."""


class PinnedFile(BaseModel):
    path: str
    sha256: str


class FastRnrrSource(BaseModel):
    repository: str
    commit: str
    raw_url: str  # format string with {commit} and {path}
    files: list[PinnedFile]
    sources: list[str]  # globs of the files compiled (as the authors' CMakeLists.txt)
    include_dirs: list[str]
    check: dict[str, str]  # source, target, expected (paths in ``files``)


class Archive(BaseModel):
    name: str
    version: str
    licence: str
    url: str
    sha256: str
    root: str  # top folder inside the archive


class Compiler(BaseModel):
    name: str
    version: str
    url: str
    sha256: str
    target: str  # zig target triple, "native" = this computer


class BuildSpec(BaseModel):
    format: str = "mskpipe.fast_rnrr_build"
    version: int = 1
    notice: str
    citation: str
    fast_rnrr: FastRnrrSource
    openmesh: Archive
    openmesh_sources: list[str]  # globs below the OpenMesh source root
    eigen: Archive
    flags: list[str]
    defines: dict[str, list[str]]  # platform -> extra -D for that platform
    compilers: dict[str, Compiler]  # platform tag -> Zig wheel
    tolerance: float  # max |vertex - expected| / bounding-box diagonal

    @property
    def short(self) -> str:
        return self.fast_rnrr.commit[:7]


class FastRnrrStatus(BaseModel):
    commit: str
    built: bool
    path: str | None = None
    compiler: str | None = None
    built_at: str | None = None
    supported: bool  # a compiler is pinned for this platform


# ---------------------------------------------------------------- spec / location


def load_spec() -> BuildSpec:
    text = files("mskpipe.resources").joinpath(SPEC_FILE).read_text(encoding="utf-8")
    return BuildSpec.model_validate_json(text)


def platform_tag() -> str | None:
    """Key of :attr:`BuildSpec.compilers` for this computer, None if not supported."""
    machine = platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        return None
    if sys.platform == "win32":
        return "win_amd64"
    if sys.platform.startswith("linux"):
        return "manylinux_x86_64"
    return None


def exe_name() -> str:
    return "Fast_RNRR.exe" if sys.platform == "win32" else "Fast_RNRR"


def build_root() -> Path:
    return home() / "fast_rnrr"


def build_dir(spec: BuildSpec | None = None) -> Path:
    return build_root() / (spec or load_spec()).short


def read_marker(folder: Path) -> dict | None:
    try:
        data = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def built_executable(spec: BuildSpec | None = None) -> Path | None:
    """The executable built by :func:`build_fast_rnrr` for the pinned commit, or None."""
    try:
        spec = spec or load_spec()
    except (OSError, ValueError):
        return None
    folder = build_dir(spec)
    marker = read_marker(folder)
    exe = folder / exe_name()
    if marker is None or marker.get("commit") != spec.fast_rnrr.commit or not exe.is_file():
        return None
    return exe


def locate(value: str | os.PathLike[str] | None = None) -> tuple[Path | None, str]:
    """Executable a run uses and where it comes from: ``value`` ("config"), else
    ``$MSKPIPE_FAST_RNRR`` ("env"), else the build of ``mskpipe setup fast-rnrr``
    ("build"), else ``Fast_RNRR`` on PATH ("path"); ``(None, ...)`` if missing. A given
    ``value`` or variable that is not a file is reported as missing, not skipped."""
    if value:
        return (Path(value) if Path(value).is_file() else None), "config"
    env = os.environ.get(FAST_RNRR_ENV)
    if env:
        return (Path(env) if Path(env).is_file() else None), "env"
    built = built_executable()
    if built is not None:
        return built, "build"
    for name in EXECUTABLE_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found), "path"
    return None, "none"


def describe_executable(exe: Path) -> dict:
    """Build information of an executable built here (commit, compiler), else {}."""
    marker = read_marker(exe.parent)
    if marker is None or marker.get("executable") != exe.name:
        return {}
    return {
        "commit": marker.get("commit"),
        "compiler": marker.get("compiler"),
        "sha256": marker.get("sha256"),
    }


def fast_rnrr_status() -> FastRnrrStatus:
    spec = load_spec()
    exe = built_executable(spec)
    marker = read_marker(exe.parent) if exe else None
    return FastRnrrStatus(
        commit=spec.fast_rnrr.commit,
        built=exe is not None,
        path=str(exe) if exe else None,
        compiler=(marker or {}).get("compiler"),
        built_at=(marker or {}).get("built_at"),
        supported=platform_tag() in spec.compilers,
    )


# ---------------------------------------------------------------- build


def build_fast_rnrr(
    log: Log,
    *,
    accepted: bool,
    cancelled: Cancelled | None = None,
    force: bool = False,
    keep_build_files: bool = False,
    jobs: int | None = None,
    spec: BuildSpec | None = None,
) -> Path:
    """Download the pinned sources and the compiler, build Fast-RNRR, check it against
    the authors' test result and install it. Returns the executable."""
    spec = spec or load_spec()
    if not accepted:
        raise FastRnrrSetupError(
            f"{spec.notice}\nConfirm that you use it for research purposes only: {ACCEPT_FLAG}"
        )
    tag = platform_tag()
    if tag not in spec.compilers:
        raise FastRnrrSetupError(
            f"Building Fast-RNRR is supported on Windows x64 and Linux x64, not here ({tag})"
        )
    final = build_dir(spec)
    if built_executable(spec) is not None and not force:
        log(f"Fast-RNRR ({spec.short}) is already built: {final / exe_name()}")
        return final / exe_name()
    compiler = spec.compilers[tag]
    root = build_root()
    downloads = root / "downloads"
    work = root / f"{spec.short}.build"
    staging = root / f"{spec.short}.partial"
    for folder in (work, staging):
        shutil.rmtree(folder, ignore_errors=True)
    stop = cancelled or (lambda: False)
    log(spec.notice)
    try:
        # ---- downloads (checked SHA-256; reused when already there)
        src = work / "fast_rnrr"
        for item in spec.fast_rnrr.files:
            url = spec.fast_rnrr.raw_url.format(commit=spec.fast_rnrr.commit, path=item.path)
            _fetch(url, src.joinpath(*PurePosixPath(item.path).parts), item.sha256, log)
            _check_cancel(stop)
        archives = {}
        for archive in (spec.openmesh, spec.eigen):
            archives[archive.name] = _fetch(
                archive.url, downloads / _url_name(archive.url), archive.sha256, log
            )
            _check_cancel(stop)
        wheel = _fetch(compiler.url, downloads / _url_name(compiler.url), compiler.sha256, log)
        _check_cancel(stop)

        # ---- unpack
        log("Unpacking the compiler and libraries")
        zig = _unpack_zig(wheel, work / "zig")
        openmesh = _unpack_tar(archives[spec.openmesh.name], work, spec.openmesh.root)
        eigen = _unpack_tar(archives[spec.eigen.name], work, spec.eigen.root)
        compat = work / "compat"
        compat.mkdir(parents=True, exist_ok=True)
        (compat / "omp.h").write_text(OMP_STUB, encoding="utf-8")
        _check_cancel(stop)

        # ---- compile and link
        includes = [compat, openmesh, eigen] + [src / d for d in spec.fast_rnrr.include_dirs]
        units = _glob(openmesh, spec.openmesh_sources) + _glob(src, spec.fast_rnrr.sources)
        exe = _compile_and_link(
            zig,
            compiler.target,
            [*spec.flags, *(f"-D{d}" for d in spec.defines.get(tag, []))],
            includes,
            units,
            work / "obj",
            work / exe_name(),
            log,
            stop,
            jobs or max(1, min(os.cpu_count() or 1, 8)),
            {
                "ZIG_GLOBAL_CACHE_DIR": str(work / "zig-cache"),
                "ZIG_LOCAL_CACHE_DIR": str(work / "zig-cache"),
            },
        )

        # ---- the authors' test
        check = run_check(exe, src, spec, work / "check", log)
        if not check["ok"]:
            raise FastRnrrSetupError(
                "The built Fast-RNRR does not reproduce the authors' test result "
                f"(max deviation {check['max_rel']:.2e} of the size, allowed {spec.tolerance:g}); "
                "it was not installed"
            )

        # ---- install
        staging.mkdir(parents=True)
        shutil.copy2(exe, staging / exe_name())
        (staging / NOTICE_FILE).write_text(
            f"{spec.notice}\n\nSource: {spec.fast_rnrr.repository} "
            f"(commit {spec.fast_rnrr.commit})\n\nCitation:\n{spec.citation}\n",
            encoding="utf-8",
        )
        marker = {
            "commit": spec.fast_rnrr.commit,
            "repository": spec.fast_rnrr.repository,
            "executable": exe_name(),
            "sha256": file_sha256(staging / exe_name()),
            "compiler": f"{compiler.name} {compiler.version} ({compiler.target})",
            "platform": tag,
            "libraries": {
                a.name: {"version": a.version, "licence": a.licence, "sha256": a.sha256}
                for a in (spec.openmesh, spec.eigen)
            },
            "flags": [*spec.flags, *(f"-D{d}" for d in spec.defines.get(tag, []))],
            "check": check,
            "accepted": {"notice": spec.notice, "flag": ACCEPT_FLAG},
            "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        }
        (staging / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
        if final.exists():
            _move_aside(final, root / f"{spec.short}.old")
        staging.replace(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if not keep_build_files:
            shutil.rmtree(work, ignore_errors=True)
            shutil.rmtree(downloads, ignore_errors=True)
    log(f"Fast-RNRR built: {final / exe_name()}")
    return final / exe_name()


def remove_fast_rnrr(log: Log, spec: BuildSpec | None = None) -> bool:
    """Delete the built executable and leftovers; False if nothing was built."""
    spec = spec or load_spec()
    root = build_root()
    folder = build_dir(spec)
    existed = folder.exists()
    if existed:
        _move_aside(folder, root / f"{spec.short}.old")
    for name in (f"{spec.short}.build", f"{spec.short}.partial", "downloads"):
        shutil.rmtree(root / name, ignore_errors=True)
    log(f"Fast-RNRR removed: {folder}" if existed else "Fast-RNRR was not built")
    return existed


OMP_STUB = """\
/* msk-pipe: serial stand-in for <omp.h> (the Zig toolchain has no OpenMP runtime).
   Fast-RNRR includes <omp.h> unconditionally and uses only omp_get_wtime(); its
   '#pragma omp' loops then run on one thread. */
#pragma once
#include <chrono>
inline double omp_get_wtime() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_max_threads() { return 1; }
inline void omp_set_num_threads(int) {}
"""


def run_check(exe: Path, src: Path, spec: BuildSpec, work: Path, log: Log) -> dict:
    """Run the authors' test (``source`` onto ``target``) and compare with ``expected``."""
    from mskpipe.io.mesh_io import read_mesh

    work.mkdir(parents=True, exist_ok=True)
    paths = {k: src.joinpath(*PurePosixPath(v).parts) for k, v in spec.fast_rnrr.check.items()}
    prefix = work / "check_"
    log("Checking the build with the authors' test data")
    proc = subprocess.run(
        [str(exe), str(paths["source"]), str(paths["target"]), str(prefix)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        creationflags=_NO_WINDOW,
        check=False,
    )
    result = Path(f"{prefix}res.obj")
    if proc.returncode != 0 or not result.is_file():
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-10:])
        raise FastRnrrSetupError(f"The built Fast-RNRR failed on the test data:\n{tail}")
    check = compare_result(
        read_mesh(result).vertices, read_mesh(paths["expected"]).vertices, spec.tolerance
    )
    if check["vertices"]:
        log(
            f"Authors' test: max deviation {check['max_rel']:.1e} of the size "
            f"({check['vertices']} vertices)"
        )
    return check


def compare_result(got, want, tolerance: float) -> dict:
    """Largest vertex deviation relative to the bounding-box diagonal of ``want``."""
    import numpy as np

    got, want = np.asarray(got, float), np.asarray(want, float)
    if got.shape != want.shape or not len(want):
        return {"ok": False, "max_rel": float("inf"), "max_abs": None, "vertices": 0}
    diag = float(np.linalg.norm(want.max(0) - want.min(0))) or 1.0
    max_abs = float(np.abs(got - want).max())
    rel = max_abs / diag
    return {"ok": rel <= tolerance, "max_rel": rel, "max_abs": max_abs, "vertices": len(got)}


# ---------------------------------------------------------------- helpers


def _check_cancel(stop: Callable[[], bool]) -> None:
    if stop():
        from mskpipe.resources.tool import ToolCancelled

        raise ToolCancelled("Fast-RNRR build cancelled")


def _url_name(url: str) -> str:
    from urllib.parse import unquote, urlparse

    return unquote(PurePosixPath(urlparse(url).path).name)


def _fetch(url: str, dest: Path, sha256: str, log: Log) -> Path:
    if dest.is_file() and file_sha256(dest) == sha256:
        return dest
    log(f"Downloading {_url_name(url)}")
    download(url, dest, sha256=sha256)
    return dest


def _safe_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise FastRnrrSetupError(f"Unsafe path in an archive: {name}")
    return path.parts


def _unpack_zig(wheel: Path, dest: Path) -> Path:
    """The ``ziglang`` package of the wheel -> ``dest``; returns the zig executable."""
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            parts = _safe_parts(info.filename)
            if not parts or parts[0] != "ziglang" or info.is_dir():
                continue
            target = dest.joinpath(*parts[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as s, target.open("wb") as d:
                shutil.copyfileobj(s, d, 1 << 20)
    exe = dest / ("zig.exe" if sys.platform == "win32" else "zig")
    if not exe.is_file():
        raise FastRnrrSetupError(f"No zig executable in {wheel.name}")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def _unpack_tar(archive: Path, dest: Path, root: str) -> Path:
    """Unpack ``root`` of a .tar.gz into ``dest`` (regular files and folders only)."""
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            parts = _safe_parts(member.name)
            if not parts or not PurePosixPath(*parts).is_relative_to(root):
                continue
            target = dest.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as s, target.open("wb") as d:  # type: ignore[union-attr]
                    shutil.copyfileobj(s, d, 1 << 20)
    folder = dest.joinpath(*PurePosixPath(root).parts)
    if not folder.is_dir():
        raise FastRnrrSetupError(f"{archive.name} has no {root}")
    return folder


def _glob(root: Path, patterns: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found += sorted(root.glob(pattern))
    if not found:
        raise FastRnrrSetupError(f"No sources {list(patterns)} in {root}")
    return found


def _compile_and_link(
    zig: Path,
    target: str,
    flags: Sequence[str],
    includes: Sequence[Path],
    units: Sequence[Path],
    obj_dir: Path,
    out: Path,
    log: Log,
    stop: Callable[[], bool],
    jobs: int,
    env_extra: dict[str, str],
) -> Path:
    env = {**os.environ, **env_extra}
    base = [str(zig), "c++", "-target", target, *flags, *(f"-I{p}" for p in includes)]
    obj_dir.mkdir(parents=True, exist_ok=True)
    log(f"Compiling {len(units)} files with Zig ({target}, {jobs} at a time)")

    def compile_one(i: int, unit: Path) -> tuple[Path, str | None]:
        if stop():
            return unit, "cancelled"
        obj = obj_dir / f"{i:03d}_{unit.stem}.o"
        proc = subprocess.run(
            [*base, "-c", str(unit), "-o", str(obj)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
            creationflags=_NO_WINDOW,
            check=False,
        )
        return obj, None if proc.returncode == 0 else (proc.stdout + proc.stderr)

    objects, done = [], 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for obj, error in pool.map(lambda a: compile_one(*a), enumerate(units)):
            if error == "cancelled":
                _check_cancel(lambda: True)
            if error is not None:
                lines = [ln for ln in error.splitlines() if "error" in ln.lower()][:10]
                raise FastRnrrSetupError(f"Compiling {obj.name} failed:\n" + "\n".join(lines))
            objects.append(obj)
            done += 1
            log(f"  compiled {done}/{len(units)}: {units[done - 1].name}")
    _check_cancel(stop)
    log("Linking")
    proc = subprocess.run(
        [str(zig), "c++", "-target", target, *flags, *map(str, objects), "-o", str(out)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
        creationflags=_NO_WINDOW,
        check=False,
    )
    if proc.returncode != 0 or not out.is_file():
        raise FastRnrrSetupError("Linking failed:\n" + (proc.stdout + proc.stderr)[-2000:])
    return out


def _move_aside(folder: Path, old: Path) -> None:
    """Rename out of the way first (fails as a whole while the exe runs on Windows)."""
    shutil.rmtree(old, ignore_errors=True)
    try:
        folder.replace(old)
    except OSError as exc:
        raise FastRnrrSetupError(
            f"Fast-RNRR is in use (a running registration?); try again later ({exc})"
        ) from exc
    shutil.rmtree(old, ignore_errors=True)
