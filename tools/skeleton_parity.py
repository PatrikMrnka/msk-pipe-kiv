# SPDX-License-Identifier: Apache-2.0
"""Skeleton step vs. MATLAB STAPLE (see mskpipe.validation.skeleton).

    # pystaple's MSKPIPE_CT dataset (LHDL CT, BP pipeline), downloaded for the installed
    # pystaple commit; this is what CI runs
    pixi run -e cpu python tools/skeleton_parity.py --fetch

    # own bones (pelvis_no_sacrum, femur_r, tibia_r) + model built by MATLAB STAPLE
    pixi run -e cpu python tools/skeleton_parity.py --bones D:/bp/bones_STAPLE `
        --reference D:/bp/bone_model.osim --json parity.json

Exit code 0 = both checks passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from mskpipe.validation.skeleton import (
    TOLERANCE,
    ParityError,
    fetch_reference,
    find_bones,
    run_parity,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="use pystaple's reference dataset")
    ap.add_argument("--cache", type=Path, default=Path(".cache/pystaple-reference"))
    ap.add_argument("--bones", type=Path, help="folder with pelvis_no_sacrum, femur_r, tibia_r")
    ap.add_argument("--reference", type=Path, help="model built by MATLAB STAPLE (.osim)")
    ap.add_argument("--out", type=Path, help="keep the built models here (default: temporary)")
    ap.add_argument("--json", type=Path, help="write the report as JSON")
    args = ap.parse_args(argv)
    if not args.fetch and not (args.bones and args.reference):
        ap.error("use --fetch, or --bones and --reference")

    try:
        if args.fetch:
            bones_dir, reference = fetch_reference(args.cache)
        else:
            bones_dir, reference = args.bones, args.reference
        bones = find_bones(bones_dir)
        with tempfile.TemporaryDirectory() as tmp:
            report = run_parity(bones, reference, args.out or Path(tmp))
    except ParityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parity, rigid, qc = report["parity"], report["rigid_frame"], report["qc"]
    commit = (report["pystaple"]["commit"] or "-")[:12]
    print(f"pystaple {report['pystaple']['version']} ({commit}), built in {report['time_s']} s")
    for unit, value in sorted(parity["max_abs_diff"].items()):
        print(f"  parity  max |diff| {value:9.3g} {unit:5s} (tolerance {TOLERANCE[unit]:g})")
    for msg in parity["ignored"]:
        print(f"  parity  ignored: {msg}")
    for msg in parity["failures"]:
        print(f"  parity  FAIL: {msg}")
    print(
        f"  rigid   max |diff| position {rigid['max_position_diff_m']:.3g} m, "
        f"rotation {rigid['max_rotation_diff']:.3g}, QC {rigid['max_qc_diff_mm']:.3g} mm"
    )
    print(
        f"  QC      hip in pelvis {qc.get('hip_center_in_pelvis_mm')} mm, "
        f"femur {qc.get('femur_length_mm')} mm, ASIS {qc.get('asis_width_mm')} mm"
    )
    print("PASSED" if report["passed"] else "FAILED")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
