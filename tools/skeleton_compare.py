# SPDX-License-Identifier: Apache-2.0
"""Compare two skeletal models built from *different* bone meshes (mm / deg).

Use this to compare a pipeline run with the BP model (MATLAB STAPLE). `skeleton_parity.py`
is the wrong tool for that: it checks bit-level parity of models built from the *same*
bones (tolerance 1e-9) and also compares raw coordinates, which differ by the frame
(BP voxel frame vs. NIfTI world, ~1.1 m for LHDL).

Both models are expressed in their own pelvis anatomical frame (child frame of
ground_pelvis), so any rigid difference between their frames cancels out.

    pixi run -e dev-cpu python tools/skeleton_compare.py `
        runs/<run>/04_skeleton/bone_model.osim `
        C:/msk-pipe-kiv-zcu/reference/skeleton/bone_model.osim `
        --json skeleton_vs_bp.json

Exit code 0 = compared (no pass/fail: the numbers are the result).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mskpipe.io.osim import OsimError
from mskpipe.validation.skeleton import ParityError, compare_anatomical


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("ours", type=Path, help="model to check (.osim)")
    ap.add_argument("reference", type=Path, help="reference model (.osim)")
    ap.add_argument("--side", choices=["r", "l"], default="r")
    ap.add_argument("--json", type=Path, help="write the full report as JSON")
    args = ap.parse_args(argv)

    try:
        report = compare_anatomical(args.ours, args.reference, args.side)
    except (OsimError, ParityError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"frame: {report['frame']}")
    print(f"{'joint':<16}{'centre mm':>11}{'parent deg':>12}{'child deg':>11}  child x/y/z deg")
    for name, j in report["joints"].items():
        axes = "/".join(f"{v:.2f}" for v in j["child_axis_angles_deg"])
        print(
            f"{name:<16}{j['centre_distance_mm']:>11.2f}{j['parent_rotation_deg']:>12.2f}"
            f"{j['child_rotation_deg']:>11.2f}  {axes}"
        )
    if report["markers_distance_mm"]:
        worst = max(report["markers_distance_mm"].items(), key=lambda kv: kv[1])
        print(f"markers: {len(report['markers_distance_mm'])}, max {worst[1]:.2f} mm ({worst[0]})")
    for key, v in report["qc"].items():
        print(f"{key:<26} ours {v['ours']}  reference {v['reference']}")
    for key in ("only_in_ours", "only_in_reference"):
        if report[key]:
            print(f"{key}: {', '.join(report[key])}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
