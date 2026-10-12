# SPDX-License-Identifier: Apache-2.0
"""Write (or check) src/mskpipe/resources/gpu_wheels.json from pixi.lock.

The Windows release installs GPU support as an overlay of the CUDA builds of torch and
torchvision pinned in pixi.lock (environment ``gpu``). Run after every ``pixi lock`` that
changes torch:

    pixi run python tools/gpu_wheels.py           # rewrite the file
    pixi run python tools/gpu_wheels.py --check   # exit 1 if it is out of date

It also fails if the ``cpu`` and ``gpu`` environments differ in anything but torch and
torchvision, because the overlay would then not reproduce the ``gpu`` environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from mskpipe.resources.gpu import OVERLAY_PACKAGES, env_differences, spec_from_lock

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "pixi.lock"
SPEC = ROOT / "src" / "mskpipe" / "resources" / "gpu_wheels.json"


def render(lock_path: Path = LOCK) -> str:
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    extra = {k: v for k, v in env_differences(lock).items() if k not in OVERLAY_PACKAGES}
    if extra:
        raise SystemExit(f"cpu and gpu environments differ beyond {OVERLAY_PACKAGES}: {extra}")
    return spec_from_lock(lock).model_dump_json(indent=2) + "\n"


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
    SPEC.write_text(text, encoding="utf-8")
    print(f"Written {SPEC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
