# SPDX-License-Identifier: Apache-2.0
"""Muscle Wrapping 2.x demo asset: build it once locally, fetch and check it in CI.

The demo is not part of the pipeline (msk-pipe ends with the Muscle Wrapping input). It is
a separately licensed asset of the GitHub release ``mw2-demo-v1`` (``packaging/release.toml``
``[mw2_demo]``)::

    pixi exec --spec python=3.11 --spec zstandard python packaging/make_mw2_demo_zip.py build \
        --mw2 <MW2 folder with Build/Bin> --example <run>/06_mw2_input --out dist
    python packaging/make_mw2_demo_zip.py fetch --out build/demo     # pinned asset (gh)
    python packaging/make_mw2_demo_zip.py check <demo folder>        # bin/ and licences
    python packaging/make_mw2_demo_zip.py verify <work folder>       # result of run_demo.cmd

``build`` writes ``msk-pipe-mw2-demo-v1-win64.zip``::

    msk-pipe-mw2-demo-v1/
        run_demo.cmd, README.txt       Muscle Wrapping on example/ or on a msk-pipe run
        bin/                           opensim-cmd.exe, OsimMuscleGeneratorTool.dll and the
                                       DLLs they import (Release build of MW2)
        example/                       06_mw2_input of msk-pipe on LHDL CT (CC BY-NC-SA 2.0 BE)
        licenses/                      licence texts of everything in bin/
        LICENSE (MW2, Apache-2.0), NOTICE, THIRD_PARTY_LICENSES.md, demo.json

``bin/`` holds exactly the transitive imports of the two roots, read from the PE import
tables; Windows system DLLs are left out, and a missing DLL, a Debug runtime or ALGLIB
(GPL, in the MW2 source tree but not linked into the plugin) stop the build. The Visual C++
runtime and the OpenMP runtime come from pinned conda-forge packages with their licence
texts: MW2 is built with ``/openmp:llvm``, which links against ``libomp140.x86_64.dll``;
Microsoft ships that file only as non-redistributable, so the LLVM OpenMP runtime of
conda-forge (``libomp.dll``, same ``__kmpc_*`` entry points) is installed under that name.

Standard library only, except ``build`` (``zstandard`` for the ``.conda`` packages).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_release import BuildError, _write_crlf, fetch, load_pins, sha256

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "packaging" / "mw2_demo"
DATE = (2026, 1, 1, 0, 0, 0)
SETUP = "setup_MuscleGeneratorTool.xml"
ROOTS = ("opensim-cmd.exe", "OsimMuscleGeneratorTool.dll")
DWNLD = Path("Sources/ExternalLibs/Dwnld")
NOT_SHIPPED = {"Testing", "GitSetup", "Maintenance"}  # VTK folders without shipped code
TEXT_SUFFIXES = (".cmd", ".txt", ".md")

# Present on every Windows 10/11 x64 (System32); never shipped.
_SYSTEM = """advapi32 bcrypt cfgmgr32 comctl32 comdlg32 crypt32 d3d11 d3d12 d3dcompiler_47
dbghelp dnsapi dwmapi dxgi gdi32 glu32 imm32 iphlpapi kernel32 kernelbase mpr msvcrt netapi32
ntdll ole32 oleaut32 opengl32 powrprof psapi rpcrt4 secur32 setupapi shell32 shlwapi ucrtbase
user32 userenv uxtheme version winmm winspool ws2_32 wsock32"""
SYSTEM_DLLS = frozenset(_SYSTEM.split())
# Debug runtimes: a Release package must not import them.
DEBUG_DLLS = frozenset(
    {"vcruntime140d.dll", "vcruntime140_1d.dll", "msvcp140d.dll", "ucrtbased.dll"}
    | {"concrt140d.dll", "libomp140d.x86_64.dll", "vcomp140d.dll"}
)
# Byte strings of code that must not be in a shipped binary.
FORBIDDEN = {b"ALGLIB: allocation error": "ALGLIB (GPL-2.0-or-later)"}
# Folders of licenses/ every demo must have.
REQUIRED_LICENCES = (
    "MuscleWrapping2",
    "OpenSim",
    "Simbody",
    "VTK",
    "Discregrid",
    "Eigen",
    "llvm-openmp",
    "vc14_runtime",
    "gcc-runtime",
    "lapack",
)
# Absolute local paths must not leak into the example.
LOCAL_PATH = re.compile(rb"(?<![A-Za-z])[A-Za-z]:[\\/][^\s<>\"]|[\\/](Users|home)[\\/]")


def demo_name(pins: Mapping[str, Any]) -> str:
    return f"msk-pipe-{pins['mw2_demo']['tag']}"


# ---------------------------------------------------------------- PE imports


def pe_imports(path: Path) -> list[str]:
    """DLL names of the import and delay-import tables of a PE file (exe or dll)."""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise BuildError(f"{path.name}: not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        raise BuildError(f"{path.name}: not a PE file")
    nsections = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic not in (0x10B, 0x20B):
        raise BuildError(f"{path.name}: unknown optional header {magic:#x}")
    dirs = opt + (112 if magic == 0x20B else 96)
    ndirs = struct.unpack_from("<I", data, dirs - 4)[0]
    sections = []
    for i in range(nsections):
        vsize, va, raw_size, raw = struct.unpack_from("<IIII", data, opt + opt_size + 40 * i + 8)
        sections.append((va, max(vsize, raw_size), raw))

    def offset(rva: int) -> int:
        for va, size, raw in sections:
            if va <= rva < va + size:
                return rva - va + raw
        raise BuildError(f"{path.name}: RVA {rva:#x} outside the sections")

    def cstring(rva: int) -> str:
        start = offset(rva)
        return data[start : data.index(b"\0", start)].decode("ascii")

    names = []
    # (data directory, descriptor size, offset of the DLL name RVA in the descriptor)
    for index, size, name_at in ((1, 20, 12), (13, 32, 4)):
        if index >= ndirs:
            continue
        rva, _ = struct.unpack_from("<II", data, dirs + 8 * index)
        if not rva:
            continue
        pos = offset(rva)
        while True:
            desc = data[pos : pos + size]
            name_rva = struct.unpack_from("<I", desc, name_at)[0] if len(desc) == size else 0
            if not name_rva:
                break
            names.append(cstring(name_rva))
            pos += size
    return names


def is_system(name: str) -> bool:
    name = name.lower()
    stem = name.rsplit(".", 1)[0]
    return name.startswith(("api-ms-win-", "ext-ms-")) or stem in SYSTEM_DLLS


def closure(
    roots: Iterable[str],
    find: Callable[[str], Path | None],
    imports: Callable[[Path], list[str]] = pe_imports,
) -> dict[str, Path]:
    """Lower-case file name -> path of every non-system DLL the roots need (roots included)."""
    found: dict[str, Path] = {}
    missing: dict[str, set[str]] = {}
    todo = [(r.lower(), "") for r in roots]
    while todo:
        name, parent = todo.pop()
        if name in found or is_system(name):
            continue
        if name in DEBUG_DLLS:
            raise BuildError(f"{parent} imports the Debug runtime {name}: use the Release build")
        path = find(name)
        if path is None:
            missing.setdefault(name, set()).add(parent or "root")
            continue
        found[name] = path
        todo += [(dep.lower(), path.name) for dep in imports(path)]
    if missing:
        listed = ", ".join(
            f"{name} (needed by {', '.join(sorted(parents))})"
            for name, parents in sorted(missing.items())
        )
        raise BuildError(f"Missing DLLs: {listed}")
    return found


def forbidden_code(paths: Iterable[Path]) -> list[str]:
    hits = []
    for path in paths:
        data = path.read_bytes()
        hits += [f"{path.name}: {what}" for marker, what in FORBIDDEN.items() if marker in data]
    return hits


# ---------------------------------------------------------------- conda runtimes


def conda_members(archive: Path, wanted: Callable[[str], bool]) -> dict[str, bytes]:
    """Files of a ``.conda`` package (zip of ``info-*.tar.zst`` and ``pkg-*.tar.zst``)."""
    import zstandard

    out = {}
    with zipfile.ZipFile(archive) as zf:
        for inner in zf.namelist():
            if not inner.endswith(".tar.zst"):
                continue
            raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(zf.read(inner)))
            with tarfile.open(fileobj=raw, mode="r|") as tar:
                for member in tar:
                    if member.isfile() and wanted(member.name):
                        out[member.name] = tar.extractfile(member).read()
    return out


def conda_runtimes(pins: Mapping[str, Any], cache: Path) -> tuple[dict[str, bytes], dict]:
    """DLL name (lower case) -> content, and per package its licence text and record."""
    dlls: dict[str, bytes] = {}
    packages = {}
    for pkg in pins["mw2_demo"]["conda"]:
        archive = fetch(pkg["url"], cache / pkg["url"].rsplit("/", 1)[1], pkg["sha256"])
        folder = pkg["dlls"].rstrip("/") + "/"
        licence = pkg["licence_file"]

        def wanted(n: str, folder: str = folder, licence: str = licence) -> bool:
            return n == licence or (n.startswith(folder) and n.endswith(".dll"))

        files = conda_members(archive, wanted)
        if pkg["licence_file"] not in files:
            raise BuildError(f"{archive.name}: no {pkg['licence_file']}")
        rename = {k.lower(): v.lower() for k, v in pkg.get("rename", {}).items()}
        for member, data in files.items():
            if member.startswith(folder):
                name = member[len(folder) :].lower()
                dlls[rename.get(name, name)] = data
        packages[pkg["name"]] = {
            "licence": files[pkg["licence_file"]],
            "record": {"url": pkg["url"], "sha256": pkg["sha256"]},
        }
    return dlls, packages


# ---------------------------------------------------------------- licences


def source_licences(mw2: Path) -> dict[str, list[Path]]:
    """Licence files of the sources the binaries in bin/ are built from (MW2 source tree)."""
    d = mw2 / DWNLD
    vtk = [d / "VTK" / "Copyright.txt"]
    for top in ("ThirdParty", "Utilities"):
        for path in sorted((d / "VTK" / top).glob("*/**/*")):
            depth = len(path.relative_to(d / "VTK" / top).parts)
            shipped = not NOT_SHIPPED & set(path.parts)
            if depth <= 4 and shipped and path.is_file() and _is_licence(path.name):
                vtk.append(path)
    found = {
        "MuscleWrapping2": [mw2 / "LICENSE"],
        "OpenSim": [d / "OpenSim" / "LICENSE.txt", d / "OpenSim" / "NOTICE"],
        "Simbody": [d / "OpenSim" / "dependencies" / "simbody" / "LICENSE.txt"],
        "spdlog": [d / "OpenSim" / "dependencies" / "spdlog" / "LICENSE"],
        "docopt.cpp": sorted((d / "OpenSim" / "Vendors" / "docopt").glob("LICENSE*")),
        "VTK": vtk,
        "Discregrid": [d / "Discregrid" / "LICENSE"],
        "vtkVisualDebugger": [d / "vtkVisualDebugger" / "LICENSE"],
        "Eigen": sorted((d / "Eigen3").glob("COPYING*")),
    }
    for component, paths in found.items():
        absent = [p for p in paths if not p.is_file()]
        if not paths or absent:
            raise BuildError(f"Licence of {component} not found: {absent or 'no files'}")
    return found


def _is_licence(name: str) -> bool:
    return re.match(r"(licen[cs]e|copying|copyright|notice)", name, re.IGNORECASE) is not None


def _licence_target(component: str, path: Path, mw2: Path) -> str:
    """Path below licenses/<component>/ (VTK files keep their path in the VTK source)."""
    if component == "VTK":
        return path.relative_to(mw2 / DWNLD / "VTK").as_posix()
    return path.name


# ---------------------------------------------------------------- example


def example_files(folder: Path) -> list[Path]:
    """Input files of a Muscle Wrapping setup folder (no outputs): setup, model, motion,
    meshes, attachment areas, Geometry of the model, msk-pipe's export.json."""
    setup = folder / SETUP
    if not setup.is_file():
        raise BuildError(f"No {SETUP} in {folder}")
    root = ET.parse(setup).getroot()
    refs = {
        el.text.strip()
        for el in root.iter()
        if el.tag.endswith("_file") and not el.tag.startswith("output_") and (el.text or "").strip()
    }
    model = root.findtext(".//model_file", "").strip()
    if not model:
        raise BuildError(f"{SETUP} has no model_file")
    geometry = {
        el.text.strip()
        for el in ET.parse(folder / model).getroot().iter("mesh_file")
        if (el.text or "").strip()
    }
    files = {setup}
    for ref in sorted(refs | geometry):
        candidates = [folder / ref] + ([folder / "Geometry" / ref] if ref in geometry else [])
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise BuildError(f"{SETUP} or {model} refers to a missing file: {ref}")
        files.add(path)
    if (folder / "export.json").is_file():
        files.add(folder / "export.json")
    text_files = [
        p for p in files if p.suffix.lower() in (".xml", ".osim", ".mot", ".sto", ".json")
    ]
    for path in text_files:
        if LOCAL_PATH.search(path.read_bytes()):
            raise BuildError(f"{path.name} contains an absolute local path")
    return sorted(files)


def run_record(example: Path) -> dict[str, Any]:
    """What produced the example: msk-pipe version and run identity (no local paths)."""
    manifest = example.parent / "manifest.json"
    if not manifest.is_file():
        return {}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    env = data.get("environment", {})
    inp = data.get("input", {})
    return {
        "mskpipe": env.get("packages", {}).get("mskpipe", {}).get("version"),
        "git": env.get("git"),
        "run_key": data.get("run_key"),
        "config_fingerprint": data.get("config_fingerprint"),
        "modality": inp.get("modality"),
        "subject_id": inp.get("subject_id"),
        "input_sha256": inp.get("sha256"),
    }


# ---------------------------------------------------------------- versions


def _version(path: Path, *patterns: str) -> str | None:
    """Version from a CMake file: one pattern per component (first group each)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    parts = [re.search(p, text) for p in patterns]
    return ".".join(m.group(1) for m in parts) if all(parts) else None


def component_versions(mw2: Path) -> dict[str, str | None]:
    d = mw2 / DWNLD
    num = r"set\({}_{}_VERSION (\d+)\)"
    return {
        "opensim": _version(
            d / "OpenSim" / "CMakeLists.txt",
            *(num.format("OPENSIM", p) for p in ("MAJOR", "MINOR", "PATCH")),
        ),
        "simbody": _version(
            d / "OpenSim" / "dependencies" / "simbody" / "CMakeLists.txt",
            r"project\(Simbody VERSION ([\d.]+)\)",
        ),
        "vtk": _version(
            d / "VTK" / "CMake" / "vtkVersion.cmake",
            *(num.format("VTK", p) for p in ("MAJOR", "MINOR", "BUILD")),
        ),
    }


# ---------------------------------------------------------------- build


def stage(
    folder: Path,
    *,
    mw2: Path,
    example: Path,
    runtimes: Mapping[str, bytes],
    packages: Mapping[str, Mapping[str, Any]],
    pins: Mapping[str, Any],
    mw2_source: str,
) -> Path:
    """Lay out the demo folder (created fresh); returns it."""
    bin_dir = mw2 / "Build" / "Bin"
    if not all((bin_dir / r).is_file() for r in ROOTS):
        raise BuildError(f"{bin_dir} has no {' / '.join(ROOTS)} (Release build of MW2)")
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "bin").mkdir(parents=True)

    by_name = {p.name.lower(): p for p in bin_dir.iterdir() if p.is_file()}
    tmp = folder / ".runtimes"
    tmp.mkdir()
    for name, data in runtimes.items():  # conda runtimes replace the build's copies
        (tmp / name).write_bytes(data)
    found = closure(ROOTS, lambda n: (tmp / n) if n in runtimes else by_name.get(n))
    hits = forbidden_code(found.values())
    if hits:
        raise BuildError("Not redistributable here: " + "; ".join(hits))
    for name, path in sorted(found.items()):
        target = name if path.parent == tmp else path.name
        shutil.copy2(path, folder / "bin" / target)
    shutil.rmtree(tmp)

    lic = folder / "licenses"
    for component, paths in source_licences(mw2).items():
        for path in paths:
            target = lic / component / _licence_target(component, path, mw2)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for name, pkg in packages.items():
        (lic / name).mkdir(parents=True, exist_ok=True)
        (lic / name / "LICENSE.txt").write_bytes(pkg["licence"])
    shutil.copytree(TEMPLATES / "licenses", lic, dirs_exist_ok=True)
    shutil.copy2(mw2 / "LICENSE", folder / "LICENSE")

    for path in example_files(example):
        target = folder / "example" / path.relative_to(example)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    # batch files need CRLF line ends (labels, goto); text files for Notepad too
    for path in TEMPLATES.iterdir():
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name == "NOTICE"):
            _write_crlf(folder / path.name, path.read_text(encoding="utf-8"))
    for name in ("LICENSE", "README.txt"):
        _write_crlf(folder / "example" / name, (TEMPLATES / "example" / name).read_text("utf-8"))

    record = {
        "format": "mskpipe.mw2_demo",
        "version": 1,
        "tag": pins["mw2_demo"]["tag"],
        "muscle_wrapping": {
            "source": mw2_source,
            "plugin_sha256": sha256(folder / "bin" / "OsimMuscleGeneratorTool.dll"),
            **component_versions(mw2),
        },
        "runtimes": {name: pkg["record"] for name, pkg in packages.items()},
        "example": run_record(example),
        "files": {
            p.relative_to(folder).as_posix(): sha256(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        },
    }
    (folder / "demo.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return folder


def make_zip(folder: Path, archive: Path) -> str:
    """Reproducible zip of ``folder`` as its top-level directory; returns its SHA-256."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", allowZip64=True) as zf:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(path.relative_to(folder.parent).as_posix(), DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return hashlib.sha256(archive.read_bytes()).hexdigest()


# ---------------------------------------------------------------- check / verify


def check(folder: Path) -> list[str]:
    """Problems of an unpacked demo folder (empty = fine)."""
    problems = []
    bin_dir = folder / "bin"
    by_name = {p.name.lower(): p for p in bin_dir.glob("*") if p.is_file()}
    try:
        needed = closure(ROOTS, by_name.get)
    except BuildError as exc:
        return [str(exc)]
    extra = sorted(set(by_name) - set(needed))
    if extra:
        problems.append(f"bin/ has files nothing imports: {', '.join(extra)}")
    problems += forbidden_code(needed.values())
    for required in ("LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md", "run_demo.cmd", "demo.json"):
        if not (folder / required).is_file():
            problems.append(f"{required} missing")
    for required in ("LICENSE", "README.txt", SETUP):
        if not (folder / "example" / required).is_file():
            problems.append(f"example/{required} missing")
    lic = folder / "licenses"
    components = {p.name for p in lic.iterdir()} if lic.is_dir() else set()
    for component in REQUIRED_LICENCES:
        if component not in components:
            problems.append(f"licenses/{component} missing")
    return problems


def verify(work: Path) -> list[str]:
    """Problems of a finished Muscle Wrapping run in ``work`` (empty = fine): the output model
    has num_of_lines fibres per muscle and the fibre-length analysis of every muscle exists."""
    root = ET.parse(work / SETUP).getroot()
    tool = next(iter(root))
    model = (tool.findtext("output_model_file") or "").strip()
    prefix = (tool.findtext("output_muscle_analysis_file_prefix") or "").strip()
    expected = {
        g.get("name"): int(g.findtext("num_of_lines") or 0) for g in tool.iter("MuscleGenerator")
    }
    problems = []
    if not expected:
        return [f"{SETUP} has no MuscleGenerator"]
    counts = dict.fromkeys(expected, 0)
    if model and (work / model).is_file():
        for _, el in ET.iterparse(work / model):
            if el.tag.endswith("Muscle") and el.get("name"):
                base = el.get("name").rsplit("_", 1)[0]
                if base in counts:
                    counts[base] += 1
            el.clear()
        for name, n in expected.items():
            if counts[name] != n:
                problems.append(f"{name}: {counts[name]} fibres in {model}, expected {n}")
    elif model:
        problems.append(f"{model} was not written")
    for name in expected:
        if not (work / f"{prefix}{name}_Lengths.sto").is_file():
            problems.append(f"{prefix}{name}_Lengths.sto missing")
    return problems


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the demo zip from a local MW2 build")
    b.add_argument(
        "--mw2", type=Path, required=True, help="MW2 folder (LICENSE, Sources, Build/Bin)"
    )
    b.add_argument("--example", type=Path, required=True, help="06_mw2_input of a msk-pipe run")
    b.add_argument(
        "--source",
        default="Muscle Wrapping 2.x, University of West Bohemia",
        help="where the MW2 source of this build comes from (demo.json, e.g. a commit)",
    )
    b.add_argument("--out", type=Path, default=Path("dist"))
    b.add_argument("--cache", type=Path, default=Path("build/mw2-demo-cache"))
    f = sub.add_parser("fetch", help="download the pinned demo asset and check its SHA-256")
    f.add_argument("--out", type=Path, default=Path("build/demo"))
    f.add_argument("--allow-missing", action="store_true", help="no pin yet: exit 0 without it")
    c = sub.add_parser("check", help="check an unpacked demo folder")
    c.add_argument("folder", type=Path)
    v = sub.add_parser("verify", help="check the result of run_demo.cmd in a work folder")
    v.add_argument("work", type=Path)
    args = parser.parse_args(argv)
    pins = load_pins()
    try:
        if args.cmd == "build":
            return _build(args, pins)
        if args.cmd == "fetch":
            return _fetch(args, pins)
        problems = check(args.folder) if args.cmd == "check" else verify(args.work)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if not problems:
        print("ok")
    return 1 if problems else 0


def _build(args: argparse.Namespace, pins: Mapping[str, Any]) -> int:
    runtimes, packages = conda_runtimes(pins, args.cache)
    folder = stage(
        args.out / "stage" / demo_name(pins),
        mw2=args.mw2,
        example=args.example,
        runtimes=runtimes,
        packages=packages,
        pins=pins,
        mw2_source=args.source,
    )
    problems = check(folder)
    if problems:
        raise BuildError("; ".join(problems))
    record = json.loads((folder / "demo.json").read_text(encoding="utf-8"))
    run = record["example"]
    if not run or "dev" in str(run.get("mskpipe")) or (run.get("git") or {}).get("dirty"):
        print(f"warning: the example was not made by a released msk-pipe ({run.get('mskpipe')})")
    archive = args.out / pins["mw2_demo"]["file"]
    digest = make_zip(folder, archive)
    tag = pins["mw2_demo"]["tag"]
    size = archive.stat().st_size / 2**20
    print(f"{archive} ({size:.1f} MiB, {len(list((folder / 'bin').iterdir()))} files in bin)")
    print(f'sha256 = "{digest}"   -> packaging/release.toml [mw2_demo]')
    print("Publish once:")
    print(
        f"  gh release create {tag} {archive.as_posix()} --latest=false "
        '--title "Muscle Wrapping 2.x demo v1 (Windows x64)" '
        '--notes "Prebuilt Muscle Wrapping 2.x (OsimMuscleGeneratorTool, OpenSim 4.5) with an '
        "example input made by msk-pipe. Not part of msk-pipe. Licences: LICENSE, NOTICE and "
        'THIRD_PARTY_LICENSES.md in the zip; example data CC BY-NC-SA 2.0 BE."'
    )
    return 0


def _fetch(args: argparse.Namespace, pins: Mapping[str, Any]) -> int:
    demo = pins["mw2_demo"]
    if not demo.get("sha256"):
        if args.allow_missing:
            print("[mw2_demo] is not pinned yet (sha256 empty): nothing to test")
            return 0
        raise BuildError("[mw2_demo] sha256 is empty in packaging/release.toml")
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / demo["file"]
    if not dest.is_file():
        argv = ["gh", "release", "download", demo["tag"], "-p", demo["file"], "-D", str(args.out)]
        subprocess.run(argv, check=True)
    digest = sha256(dest)
    if digest != demo["sha256"]:
        raise BuildError(f"Demo checksum mismatch: expected {demo['sha256']}, got {digest}")
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
