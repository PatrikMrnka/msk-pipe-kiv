# SPDX-License-Identifier: Apache-2.0
"""Compare the attachment outlines of a run with reference outlines (mm).

On the atlas subject itself (LHDL CT -> LHDL atlas) the reference is the atlas: the
"transfer without registration". Outlines keep their point order, so point i of the run
corresponds to point i of the reference: reported per area are the mean / max distance of
corresponding points and the centroid shift.

    pixi run -e dev-cpu python tools/attachments_compare.py runs/<run> `
        --reference D:/data/lhdl_atlas --json attachments_vs_atlas.json

Exit code 0 = compared (the numbers are the result).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mskpipe.core.workspace import Workspace
from mskpipe.io.attachment_vtk import read_points
from mskpipe.steps.attachments import read_attachments_index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", type=Path, help="run folder (runs/<run>)")
    ap.add_argument("--reference", type=Path, required=True, help="atlas folder (muscles/...)")
    ap.add_argument("--json", type=Path, help="write the full report as JSON")
    args = ap.parse_args(argv)

    out = Workspace.open(args.run).step_dir("attachments")
    index = read_attachments_index(out)
    rows = []
    for area in index.get("areas", []):
        if area.get("status") != "ok":
            continue
        ours = read_points(out / area["file"])
        ref_file = args.reference / "muscles" / area["file"]
        if not ref_file.is_file():
            continue
        ref = read_points(ref_file)
        row = {"muscle": area["muscle"], "kind": area["kind"], "n_points": len(ours)}
        if len(ref) == len(ours):
            d = np.linalg.norm(ours - ref, axis=1)
            row.update(point_mean_mm=float(d.mean()), point_max_mm=float(d.max()))
        row["centroid_mm"] = float(np.linalg.norm(ours.mean(0) - ref.mean(0)))
        rows.append(row)

    print(f"{'muscle':<26}{'area':<5}{'n':>4}{'mean mm':>10}{'max mm':>9}{'centroid':>10}")
    for r in rows:
        mean, worst = r.get("point_mean_mm", float("nan")), r.get("point_max_mm", float("nan"))
        print(
            f"{r['muscle']:<26}{r['kind']:<5}{r['n_points']:>4}"
            f"{mean:>10.2f}{worst:>9.2f}{r['centroid_mm']:>10.2f}"
        )
    summary = {}
    if rows:
        means = np.array([r.get("point_mean_mm", np.nan) for r in rows])
        cents = np.array([r["centroid_mm"] for r in rows])
        summary = {
            "n_areas": len(rows),
            "point_mean_mm": {"median": float(np.nanmedian(means)), "max": float(np.nanmax(means))},
            "centroid_mm": {"median": float(np.median(cents)), "max": float(cents.max())},
        }
        pm, cm = summary["point_mean_mm"], summary["centroid_mm"]
        print(
            f"\n{len(rows)} areas: point distance median {pm['median']:.2f} mm"
            f" (worst {pm['max']:.2f}), centroid median {cm['median']:.2f} mm"
            f" (worst {cm['max']:.2f})"
        )
    for bone, r in index.get("bones", {}).items():
        if r.get("status") == "ok":
            nr = r.get("nonrigid", r["rigid"])
            print(
                f"{bone:<8} scale {r['scale']:.4f} rot {r['rotation_deg']:.2f} deg "
                f"rigid {r['rigid']['mean_mm']:.2f} -> {nr['mean_mm']:.2f} mm"
            )
    if args.json:
        report = {
            "run": str(args.run),
            "reference": str(args.reference),
            "summary": summary,
            "areas": rows,
            "registration": index.get("bones", {}),
        }
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
