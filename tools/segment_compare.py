# SPDX-License-Identifier: Apache-2.0
"""Check the raw segmentations of a run (step ``segment``), optionally against reference masks.

Without a reference: every configured structure has voxels, ``_r`` lies right of ``_l``
(RAS +x), pelvis > femur > tibia superiorly (catches swapped label IDs/sides, flipped grids).
With ``--reference``: Dice, volume ratio, centroid shift, ASSD and HD95 per mask
(e.g. the BP raw masks: TotalSegmentator ``hip_left``, ``femur_right``, ``tibia``, ...,
MuscleMap muscles under unified names).

    pixi run -e dev-cpu python tools/segment_compare.py runs/<run> `
        --reference C:/msk-pipe-kiv-zcu/reference/labelmap/raw --json segment_vs_bp.json

Exit code 0 = sanity checks passed and every reference mask has Dice >= --min-dice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mskpipe.core.workspace import Workspace
from mskpipe.labelmap.scheme import load_scheme
from mskpipe.validation.segment import (
    RunSegmentation,
    compare_reference,
    format_report,
    sanity_check,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="run folder (runs/<run>)")
    parser.add_argument("--reference", type=Path, help="folder with reference masks (.nii.gz)")
    parser.add_argument("--min-dice", type=float, default=0.95)
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    ws = Workspace.open(args.run)
    config = ws.load_config()
    scheme = load_scheme()
    seg = RunSegmentation(ws.step_dir("segment"))
    sanity = sanity_check(seg, config, scheme)
    reference = compare_reference(seg, args.reference, scheme, config) if args.reference else None
    print(format_report(sanity, reference))
    if args.json:
        report = {"run": str(args.run), "sanity": sanity, "reference": reference}
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    low = []
    if reference is not None:
        low = [n for n, r in reference["structures"].items() if r["dice"] < args.min_dice]
        if low:
            print(f"Dice < {args.min_dice}: {', '.join(low)}")
        if not reference["structures"]:
            print("FAILED: no reference mask matched the run")
            return 1
    return 1 if sanity["problems"] or low else 0


if __name__ == "__main__":
    sys.exit(main())
