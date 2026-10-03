# SPDX-License-Identifier: Apache-2.0
"""Label map cleaning vs. BP ``clean_masks.py`` (see mskpipe.validation.labelmap).

    pixi run -e dev-cpu python tools/labelmap_compare.py `
        --raw C:/msk-pipe-kiv-zcu/reference/labelmap/raw `
        --cleaned C:/msk-pipe-kiv-zcu/reference/labelmap/cleaned --json labelmap_compare.json

Exit code 0 = every structure has Dice >= --min-dice and no voxel outside the BP mask.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mskpipe.validation.labelmap import compare_with_bp, format_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="BP masks before cleaning")
    parser.add_argument("--cleaned", type=Path, required=True, help="BP cleaned masks")
    parser.add_argument("--min-dice", type=float, default=0.975)
    parser.add_argument("--json", type=Path, help="Write the full report as JSON")
    args = parser.parse_args(argv)

    report = compare_with_bp(args.raw, args.cleaned)
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    bad = [
        name
        for name, r in report["structures"].items()
        if r["dice"] < args.min_dice or r["only_ours"] > 0
    ]
    if bad or not report["structures"]:
        print(f"FAILED: {', '.join(bad) or 'nothing compared'}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
