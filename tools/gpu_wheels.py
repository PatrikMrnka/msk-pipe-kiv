# SPDX-License-Identifier: Apache-2.0
"""Write (or check) src/mskpipe/resources/gpu_wheels.json from pixi.lock.

The releases install GPU support as an overlay of the wheels in which the ``gpu``
environment of pixi.lock differs from ``cpu``: the CUDA builds of torch and torchvision
and, on Linux, the CUDA libraries (``nvidia-*``, ``triton``). Run after every ``pixi lock``
that changes torch or adds a platform:

    pixi run python tools/gpu_wheels.py           # rewrite the file
    pixi run python tools/gpu_wheels.py --check   # exit 1 if it is out of date

It fails if the environments differ in anything else (another version of a common
package, a package only in ``cpu``, a conda package), because the overlay would then not
reproduce ``gpu``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from mskpipe.resources.gpu import PLATFORMS, specs_from_lock, unexpected_differences

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "pixi.lock"
SPEC = ROOT / "src" / "mskpipe" / "resources" / "gpu_wheels.json"


def render(lock_path: Path = LOCK) -> str:
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    for platform in PLATFORMS:
        if platform not in lock["environments"]["gpu"]["packages"]:
            continue
        extra = unexpected_differences(lock, platform)
        if extra:
            raise SystemExit(
                f"{platform}: cpu and gpu environments differ beyond torch and CUDA: {extra}"
            )
    return specs_from_lock(lock).model_dump_json(indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only compare, do not write")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = SPEC.read_text(encoding="utf-8") if SPEC.is_file() else ""
        if current != text:
            print(f"{SPEC.relative_to(ROOT)} is out of date: run tools/gpu_wheels.py")
            return 1
        print(f"{SPEC.relative_to(ROOT)} matches pixi.lock")
        return 0
    SPEC.write_text(text, encoding="utf-8", newline="\n")
    print(f"Written {SPEC.relative_to(ROOT)}")
    for platform, spec in specs_from_lock(
        yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    ).platforms.items():
        print(f"  {platform}: {spec.tag}, {len(spec.wheels)} wheels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
