# SPDX-License-Identifier: Apache-2.0
"""Write ``src/mskpipe/resources/fast_rnrr_build.json``: what ``mskpipe setup fast-rnrr``
downloads (URL + SHA-256) and how it builds Fast-RNRR.

Run after changing a pinned version below (needs internet; downloads ~100 MB to hash the
files that PyPI does not hash for us)::

    pixi run python tools/fast_rnrr_pins.py            # write the JSON
    pixi run python tools/fast_rnrr_pins.py --check    # only compare with the JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "mskpipe" / "resources" / "fast_rnrr_build.json"

FAST_RNRR_REPO = "https://github.com/Juyong/Fast_RNRR"
FAST_RNRR_COMMIT = "9c6e336e6f1096ae8144d64d9d8009183d622fba"
RAW = "https://raw.githubusercontent.com/Juyong/Fast_RNRR/{commit}/{path}"
FAST_RNRR_FILES = [
    "include/median.h",
    "include/nanoflann.hpp",
    "src/NonRigidreg.cpp",
    "src/NonRigidreg.h",
    "src/Registration.cpp",
    "src/Registration.h",
    "src/geodesic/geodesic_algorithm_base.h",
    "src/geodesic/geodesic_algorithm_exact.h",
    "src/geodesic/geodesic_algorithm_exact_elements.h",
    "src/geodesic/geodesic_constants_and_simple_functions.h",
    "src/geodesic/geodesic_memory.h",
    "src/main.cpp",
    "src/tools/OmpHelper.h",
    "src/tools/Types.h",
    "src/tools/io_mesh.h",
    "src/tools/nanoflann.h",
    "src/tools/nodeSampler.cpp",
    "src/tools/nodeSampler.h",
    "src/tools/tools.cpp",
    "src/tools/tools.h",
    "data/test/source.obj",
    "data/test/target_mesh.obj",
    "data/test/mesh_res.obj",
]
NOTICE = (
    "Fast-RNRR (Yao, Deng, Xu, Zhang; https://github.com/Juyong/Fast_RNRR) is protected "
    "under patent and may only be used for research purposes. For business or for-profit "
    "use contact Juyong Zhang (juyong@ustc.edu.cn). msk-pipe does not distribute it: it is "
    "downloaded from the authors' repository and built on this computer."
)
CITATION = (
    "Yao Y., Deng B., Xu W., Zhang J. (2020). Quasi-Newton Solver for Robust Non-Rigid "
    "Registration. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). "
    "https://arxiv.org/abs/2004.04322"
)
ZIG_VERSION = "0.13.0"
ZIG_PLATFORMS = {  # platform tag -> (wheel tag, zig target)
    "win_amd64": ("win_amd64", "x86_64-windows-gnu"),
    "manylinux_x86_64": ("manylinux_2_12_x86_64", "native"),
}
OPENMESH_PYPI = "1.2.1"  # sdist of the Python bindings, contains the C++ OpenMesh 9.1.0
EIGEN = {
    "version": "3.4.0",
    "url": "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
    # the same hash as the conda-forge eigen recipe
    "sha256": "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _pypi(name: str, version: str) -> list[dict]:
    return json.loads(_get(f"https://pypi.org/pypi/{name}/{version}/json"))["urls"]


def build_spec(verify_eigen: bool) -> dict:
    files = []
    for path in FAST_RNRR_FILES:
        data = _get(RAW.format(commit=FAST_RNRR_COMMIT, path=path))
        files.append({"path": path, "sha256": hashlib.sha256(data).hexdigest()})
        print(f"  {path}", file=sys.stderr)
    sdist = next(f for f in _pypi("openmesh", OPENMESH_PYPI) if f["packagetype"] == "sdist")
    if verify_eigen:
        got = hashlib.sha256(_get(EIGEN["url"])).hexdigest()
        if got != EIGEN["sha256"]:
            raise SystemExit(f"Eigen tarball changed: {got}")
    wheels = _pypi("ziglang", ZIG_VERSION)
    compilers = {}
    for tag, (wheel_tag, target) in ZIG_PLATFORMS.items():
        wheel = next(f for f in wheels if wheel_tag in f["filename"])
        compilers[tag] = {
            "name": "zig",
            "version": ZIG_VERSION,
            "url": wheel["url"],
            "sha256": wheel["digests"]["sha256"],
            "target": target,
        }
    return {
        "format": "mskpipe.fast_rnrr_build",
        "version": 1,
        "notice": NOTICE,
        "citation": CITATION,
        "fast_rnrr": {
            "repository": FAST_RNRR_REPO,
            "commit": FAST_RNRR_COMMIT,
            "raw_url": RAW,
            "files": files,
            "sources": ["src/*.cpp", "src/geodesic/*.cpp", "src/tools/*.cpp"],
            "include_dirs": ["src", "include"],
            "check": {
                "source": "data/test/source.obj",
                "target": "data/test/target_mesh.obj",
                "expected": "data/test/mesh_res.obj",
            },
        },
        "openmesh": {
            "name": "openmesh",
            "version": "9.1.0",
            "licence": "BSD-3-Clause",
            "url": sdist["url"],
            "sha256": sdist["digests"]["sha256"],
            "root": f"openmesh-{OPENMESH_PYPI}/OpenMesh/src",
        },
        "openmesh_sources": ["OpenMesh/Core/**/*.cc"],
        "eigen": {
            "name": "eigen",
            "version": EIGEN["version"],
            "licence": "MPL-2.0",
            "url": EIGEN["url"],
            "sha256": EIGEN["sha256"],
            "root": f"eigen-{EIGEN['version']}",
        },
        # the authors' CMakeLists.txt: -O3 -std=c++14 (Release adds -DNDEBUG); OpenMesh's:
        # _USE_MATH_DEFINES (and NOMINMAX on Windows); OM_STATIC_BUILD = linked in statically
        "flags": [
            "-std=c++14",
            "-O3",
            "-DNDEBUG",
            "-D_USE_MATH_DEFINES",
            "-DOM_STATIC_BUILD",
            "-w",
        ],
        "defines": {"win_amd64": ["NOMINMAX"]},
        "compilers": compilers,
        "tolerance": 1e-4,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Only compare with the JSON.")
    parser.add_argument(
        "--no-eigen", action="store_true", help="Do not download Eigen to verify its hash."
    )
    args = parser.parse_args()
    spec = build_spec(verify_eigen=not args.no_eigen)
    text = json.dumps(spec, indent=2) + "\n"
    if args.check:
        same = OUT.read_text(encoding="utf-8") == text
        print("up to date" if same else f"{OUT} differs, run without --check")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"Written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
