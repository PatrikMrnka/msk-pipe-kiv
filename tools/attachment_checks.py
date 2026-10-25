#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Attachment-area checks on LHDL (paper_results E19).

On LHDL the attachment atlas is the same donor, so E18 measures only the registration /
projection noise floor (0.57 mm). Two checks put the TLEM 2.0 residual (~12 mm) in context:

(a) Other side as atlas - the right atlas mirrored onto the left LHDL leg vs the left
    dissection (bilateral asymmetry + the two sides dissected separately):

    mirror-atlas  write the mirrored atlas (right -> left: bones, outlines, atlas.json)
    gt-bones      register the mirrored atlas onto the left GT bones (no segmentation
                  error) with the pipeline's own plugin; control = the left atlas itself
    pair          gt_compare of pipeline runs with the mirrored atlas
                  (paper/experiments/e19_mirror.yaml) vs the E18 left runs, paired per area

(b) Definition floor - the LHDL areas built as for TLEM 2.0 (convex hull of the fibre end
    points + 4 mm, as data/reference-TLEM20/tlem_to_mskpipe.py) vs the dissection areas,
    no registration:

    fibre-hull    end points of the digitised fascicles / tendons (MscFbr_*, TndFbr_*)
                  that lie on the bone of the area -> outline -> compare

Metrics are those of tools/gt_compare.py (centroid_dist_mm, gt2p/p2gt, hd) plus
area_ratio (outline area / dissection area). Outputs: CSV + JSON + the outlines (.vtk).

Usage (repository root; outputs default to ..\\paper\\stats\\e19\\<check>)::

    pixi run -e dev-cpu python tools/attachment_checks.py mirror-atlas
    pixi run -e dev-cpu python tools/attachment_checks.py gt-bones
    pixi run -e dev-cpu python tools/attachment_checks.py fibre-hull
    pixi run -e dev-cpu python tools/attachment_checks.py pair `
        --mirror ../paper/stats/e19/runs/attachments_LHDL_mesh.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gt_compare import compare_points  # noqa: E402  (same metrics as E18)

AREAS = ("Ori", "Ins")
GT = Path("data/reference-GT")
ATLAS = GT / "lhdl_atlas"
MIRROR = GT / "lhdl_atlas_mirror"
POLYLINES = GT / "lhdl_msf" / "polylines"
E18_AREAS = Path("../paper/stats/e18/attachments_LHDL_mesh.csv")
# TLEM 2.0 areas defined differently from LHDL (paper_results E18), reported separately
OTHER_DEFINITION = {
    ("vastus_lateralis", "Ori"),
    ("vastus_lateralis", "Ins"),
    ("vastus_medialis", "Ins"),
    ("rectus_femoris", "Ins"),
    ("adductor_magnus", "Ins"),
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def other(side: str) -> str:
    return {"r": "l", "l": "r"}[side]


def swap_side(name: str, side: str) -> str:
    """femur_r -> femur_l; names without a side (pelvis_with_sacrum, pelvis) unchanged."""
    for suf in ("_r", "_l"):
        if name.endswith(suf):
            return name[:-2] + "_" + side
    return name


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def area_ratio(pts: np.ndarray, gt: np.ndarray) -> float:
    from mskpipe.geometry.outline import polygon_area

    a = polygon_area(gt)
    return float(polygon_area(pts) / a) if a > 0 else float("nan")


def compare(pts: np.ndarray, gt: np.ndarray) -> dict:
    return compare_points(pts, gt) | {"area_ratio": area_ratio(pts, gt)}


def gt_outline(atlas: Path, muscle: str, kind: str) -> np.ndarray | None:
    from mskpipe.io.attachment_vtk import read_points

    files = sorted((atlas / "muscles" / muscle).glob(f"*_{kind}.vtk"))
    return np.vstack([read_points(f) for f in files]) if files else None


def median_iqr(v: Iterable[float]) -> str:
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(a):
        return "-"
    q1, m, q3 = np.percentile(a, [25, 50, 75])
    return f"{m:.2f} [{q1:.2f}-{q3:.2f}] (n={len(a)})"


# ---------------------------------------------------------------- (a) mirrored atlas


def cmd_mirror_atlas(args: argparse.Namespace) -> int:
    """Right atlas -> left (or reverse): x' = 2 c - x, c = x of the pelvis centroid."""
    from mskpipe.io.attachment_vtk import read_points, write_points
    from mskpipe.io.mesh_io import read_mesh, write_mesh

    src, dst, s = args.atlas, args.out, args.source_side
    t = other(s)
    if (dst / "atlas.json").exists() and not args.force:
        log(f"{dst} exists (use --force)")
        return 1
    index = json.loads((src / "atlas.json").read_text(encoding="utf-8"))
    pelvis = next(
        (
            src / "bones" / f"{n}.obj"
            for n in ("pelvis_with_sacrum", "pelvis_no_sacrum")
            if (src / "bones" / f"{n}.obj").is_file()
        ),
        None,
    )
    if pelvis is None:
        log(f"no pelvis in {src / 'bones'}")
        return 1
    cx = float(read_mesh(pelvis).vertices[:, 0].mean())
    m = np.eye(4)
    m[0, 0], m[0, 3] = -1.0, 2.0 * cx
    log(f"mirror plane x = {cx:.2f} mm (pelvis centroid), {s} -> {t}")

    bones = {}
    for name in (
        "pelvis_with_sacrum",
        "pelvis_no_sacrum",
        f"femur_{s}",
        f"tibia_{s}",
        f"patella_{s}",
    ):
        p = src / "bones" / f"{name}.obj"
        if p.is_file():
            out = dst / "bones" / f"{swap_side(name, t)}.obj"
            out.parent.mkdir(parents=True, exist_ok=True)
            write_mesh(out, read_mesh(p).transformed(m))  # winding reversed by transformed()
            bones[name] = out.name

    muscles = {}
    for name, rec in sorted(index["muscles"].items()):
        if not name.endswith(f"_{s}"):
            continue
        new = swap_side(name, t)
        out_rec = {"stem": rec["stem"], "codes": rec.get("codes", []), "mirrored_from": name}
        for kind in AREAS:
            area = rec.get(kind)
            if not area or "file" not in area:
                continue
            pts = read_points(src / area["file"])
            pts = (pts @ m[:3, :3].T + m[:3, 3])[::-1]  # reversed: still clockwise from outside
            rel = f"muscles/{new}/{Path(area['file']).name}"
            write_points(dst / rel, pts)
            out_rec[kind] = {
                "file": rel,
                "source": f"{area.get('source', '')}_mirrored_{s}{t}",
                "n_points": len(pts),
                "bone": swap_side(area["bone"], t),
                "body": swap_side(area.get("body", ""), t),
                "orientation": area.get("orientation", ""),
            }
        muscles[new] = out_rec
    (dst / "atlas.json").write_text(
        json.dumps(
            {
                "frame": index.get("frame", "ras"),
                "units": index.get("units", "mm"),
                "source": str(src),
                "mirror": {
                    "from_side": s,
                    "to_side": t,
                    "plane_x_mm": round(cx, 3),
                    "bones": bones,
                },
                "muscles": muscles,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lic = src / "LICENSE"
    if lic.is_file():
        (dst / "LICENSE").write_bytes(lic.read_bytes())
    log(f"{len(muscles)} muscles, {len(bones)} bones -> {dst}")
    return 0


class _Ctx:
    """Minimal StepContext for calling the attachments plugin outside a run."""

    def __init__(self, side: str) -> None:
        import logging

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.config = SimpleNamespace(skeleton=SimpleNamespace(side=side))
        self.logger = logging.getLogger("attachment_checks")

    def check_cancel(self) -> None:
        pass

    def run_command(self, args, *, env=None, cwd=None) -> None:
        import os

        subprocess.run(
            [str(a) for a in args],
            check=True,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=subprocess.DEVNULL,
        )


def cmd_gt_bones(args: argparse.Namespace) -> int:
    """Atlas (same side / mirrored) registered onto the GT bones of ``--side``."""
    from mskpipe.plugins.attachments.atlas_registration import (
        AtlasRegistration,
        AtlasRegistrationParams,
    )

    side, gt = args.side, args.atlas
    bones = {
        "pelvis_with_sacrum": gt / "bones" / "pelvis_with_sacrum.obj",
        f"femur_{side}": gt / "bones" / f"femur_{side}.obj",
        f"tibia_{side}": gt / "bones" / f"tibia_{side}.obj",
    }
    for b in bones.values():
        if not b.is_file():
            log(f"missing GT bone {b}")
            return 1
    rows, regs = [], []
    for label, atlas in (("same", gt), ("mirror", args.mirror)):
        if not (atlas / "atlas.json").is_file():
            log(f"skip {label}: no atlas.json in {atlas} (run mirror-atlas first)")
            continue
        index = json.loads((atlas / "atlas.json").read_text(encoding="utf-8"))
        muscles = {k: Path(k) for k in index["muscles"] if k.endswith(f"_{side}")}
        for method in args.nonrigid:
            out = args.out / label / method
            log(f"{label} atlas, {method} -> {out}")
            params = AtlasRegistrationParams(
                atlas_dir=str(atlas), nonrigid=method, fast_rnrr_exe=args.fast_rnrr_exe
            )
            AtlasRegistration().compute(_Ctx(side), bones, muscles, out, params)
            transfer = json.loads((out / "transfer.json").read_text(encoding="utf-8"))
            for group, info in transfer["bones"].items():
                regs.append(
                    {"atlas": label, "nonrigid": method, "bone": group}
                    | {k: info.get(k) for k in ("scale", "rotation_deg", "start")}
                    | {f"rigid_{k}": v for k, v in (info.get("rigid") or {}).items()}
                    | {f"nonrigid_{k}": v for k, v in (info.get("nonrigid") or {}).items()}
                )
            for area in transfer["areas"]:
                if area.get("status") != "ok":
                    continue
                ref = gt_outline(gt, area["muscle"], area["kind"])
                if ref is None:
                    continue
                from mskpipe.io.attachment_vtk import read_points

                pts = read_points(out / area["file"])
                rows.append(
                    {
                        "atlas": label,
                        "nonrigid": method,
                        "side": side,
                        "muscle": area["muscle"],
                        "area": area["kind"],
                    }
                    | compare(pts, ref)
                )
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "gt_bones_areas.csv", rows)
    write_csv(args.out / "gt_bones_registration.csv", regs)
    print(f"\nGT bones of side '{side}' vs dissection {side}: centroid_dist_mm median [IQR]")
    for label in ("same", "mirror"):
        for method in args.nonrigid:
            sel = [r for r in rows if r["atlas"] == label and r["nonrigid"] == method]
            if sel:
                print(
                    f"  {label:6s} {method:13s} {median_iqr(r['centroid_dist_mm'] for r in sel)}"
                    f"  gt2p {median_iqr(r['gt2p_mean'] for r in sel)}"
                )
    _paired(rows, "atlas", "same", "mirror", ("nonrigid",))
    return 0


# ------------------------------------------------------------ (a) pipeline runs, paired


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _paired(rows: list[dict], col: str, a: str, b: str, by: tuple[str, ...]) -> list[dict]:
    """Paired b - a per area (Wilcoxon) within each group ``by``; prints and returns."""
    from scipy.stats import wilcoxon

    key = lambda r: (r["muscle"], r["area"])  # noqa: E731
    groups = sorted({tuple(r[k] for k in by) for r in rows})
    out = []
    for g in groups:
        sel = [r for r in rows if tuple(r[k] for k in by) == g]
        A = {key(r): float(r["centroid_dist_mm"]) for r in sel if r[col] == a}
        B = {key(r): float(r["centroid_dist_mm"]) for r in sel if r[col] == b}
        common = sorted(set(A) & set(B))
        if len(common) < 3:
            continue
        va, vb = np.array([A[k] for k in common]), np.array([B[k] for k in common])
        d = vb - va
        p = float(wilcoxon(vb, va).pvalue) if np.any(d != 0) else 1.0
        rec = dict(zip(by, g, strict=True)) | {
            "n": len(common),
            f"{a}_median": float(np.median(va)),
            f"{b}_median": float(np.median(vb)),
            "delta_median": float(np.median(d)),
            "p_wilcoxon": p,
        }
        out.append(rec)
        print(
            f"  {' '.join(g):38s} n={len(common):2d}  {a} {np.median(va):6.2f}  "
            f"{b} {np.median(vb):6.2f}  delta {np.median(d):+6.2f}  p={p:.2g}"
        )
    return out


def cmd_pair(args: argparse.Namespace) -> int:
    """E18 left runs (same-side atlas) vs runs with the mirrored atlas (gt_compare CSVs)."""
    base = [r for r in _read(args.base) if r.get("side") == args.side]
    mirror = [r for r in _read(args.mirror) if r.get("side") == args.side]
    for r in base:
        r["atlas"] = "same"
    for r in mirror:
        r["atlas"] = "mirror"
    base = [r for r in base if not r["subject"].upper().endswith("_MIRROR")]
    mirror = [r for r in mirror if r["subject"].upper().endswith("_MIRROR")]
    if not base or not mirror:
        log(
            f"no rows: base {len(base)}, mirror {len(mirror)} (side {args.side}; mirror runs "
            "need a subject ending in _MIRROR, see e19_mirror.yaml)"
        )
        return 2
    rows = base + mirror
    by = ("modality", "bones_source", "nonrigid")
    print(f"Side '{args.side}': centroid_dist_mm, same-side atlas vs mirrored atlas (paired)")
    res = _paired(rows, "atlas", "same", "mirror", by)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "pair_summary.csv", res)
    # per-area table of the main configuration (CT, TS, default non-rigid stage)
    main = [r for r in rows if r["modality"] == "ct" and r["bones_source"] == "totalsegmentator"]
    per = {}
    for r in main:
        k = (r["nonrigid"], r["muscle"], r["area"])
        per.setdefault(k, {})[r["atlas"]] = float(r["centroid_dist_mm"])
    write_csv(
        args.out / "pair_areas_ct_ts.csv",
        [
            {"nonrigid": k[0], "muscle": k[1], "area": k[2]}
            | v
            | ({"delta": v["mirror"] - v["same"]} if len(v) == 2 else {})
            for k, v in sorted(per.items())
        ],
    )
    log(f"-> {args.out}")
    return 0


# ---------------------------------------------------------------- (b) fibre-hull floor


def _polylines(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    import vtk
    from vtk.util import numpy_support as ns

    r = vtk.vtkPolyDataReader()
    r.SetFileName(str(path))
    r.Update()
    pd = r.GetOutput()
    pts = ns.vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    off = ns.vtk_to_numpy(pd.GetLines().GetOffsetsArray())
    con = ns.vtk_to_numpy(pd.GetLines().GetConnectivityArray())
    return pts, [con[off[i] : off[i + 1]] for i in range(len(off) - 1)]


def chain_ends(path: Path) -> np.ndarray:
    """End points of the digitised chains (nodes of degree != 2; coincident points merged)."""
    pts, lines = _polylines(path)
    deg: dict[tuple, int] = {}
    edges = set()
    for line in lines:
        ks = [tuple(np.round(pts[i], 3)) for i in line]
        for a, b in itertools.pairwise(ks):
            e = (min(a, b), max(a, b))
            if a == b or e in edges:
                continue
            edges.add(e)
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
    return np.array([k for k, d in deg.items() if d != 2], float).reshape(-1, 3)


class Bone:
    """Bone mesh with a cell locator (closest point + triangle normal)."""

    def __init__(self, mesh) -> None:
        import vtk

        from mskpipe.geometry.surface import to_polydata

        self.mesh = mesh
        t = mesh.vertices[mesh.faces]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        self.normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self.centers = t.mean(1)
        self._vtk = vtk
        self._keep: list = []  # polydata of the locators (kept alive)
        self.loc = self._locator(to_polydata(mesh))

    def _locator(self, pd):
        loc = self._vtk.vtkStaticCellLocator()
        loc.SetDataSet(pd)
        loc.BuildLocator()
        self._keep.append(pd)
        return loc

    @staticmethod
    def _query(loc, vtk, points):
        n = len(points)
        out, dist, cells = np.empty((n, 3)), np.empty(n), np.empty(n, int)
        c = [0.0, 0.0, 0.0]
        cid, sid, d2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        for i, p in enumerate(np.asarray(points, float)):
            loc.FindClosestPoint(p, c, cid, sid, d2)
            out[i], dist[i], cells[i] = c, np.sqrt(d2.get()), cid.get()
        return out, dist, cells

    def closest(self, points):
        return self._query(self.loc, self._vtk, points)

    def closest_on_side(self, points, n, c, extent):
        """Closest point on the faces facing ``n`` near ``c`` (not the opposite cortex)."""
        from mskpipe.geometry.surface import to_polydata
        from mskpipe.io.mesh_io import TriMesh

        keep = (self.normals @ n > 0.0) & (np.linalg.norm(self.centers - c, axis=1) < extent)
        if keep.sum() < 4:
            return self.closest(points)[0]
        loc = self._locator(to_polydata(TriMesh(self.mesh.vertices, self.mesh.faces[keep])))
        return self._query(loc, self._vtk, points)[0]


def hull_outline(
    points: np.ndarray, bone: Bone, radius: float, step: float, max_points: int = 48
) -> np.ndarray:
    """Closed outline on ``bone`` around ``points``: tlem_to_mskpipe.outline() on TriMesh/VTK
    (convex hull of the points buffered by ``radius`` in the tangent plane, resampled every
    ``step`` mm, 8..max_points points, projected onto the bone, clockwise from outside)."""
    from scipy.spatial import ConvexHull

    proj, _, cells = bone.closest(points)
    normals = bone.normals[cells]
    n = normals.mean(0)
    if np.linalg.norm(n) < 0.3:  # points on opposite sides: plane of the points
        _, _, vt = np.linalg.svd(proj - proj.mean(0), full_matrices=False)
        ax = vt[2]
        n = ax * (np.sign(ax @ normals.sum(0)) or 1.0)
    n /= np.linalg.norm(n)
    c = proj.mean(0)
    flat = (proj - c) - np.outer((proj - c) @ n, n)
    if len(proj) > 1 and np.linalg.norm(flat, axis=1).max() > 1e-6:
        u = np.linalg.svd(flat, full_matrices=False)[2][0]
    else:
        u = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    u = u - (u @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    q = np.c_[flat @ u, flat @ v]
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    disc = np.c_[np.cos(ang), np.sin(ang)] * radius
    cloud = (q[:, None, :] + disc[None, :, :]).reshape(-1, 2)
    hull = cloud[ConvexHull(cloud).vertices]  # counter-clockwise seen from +n
    seg = np.linalg.norm(np.roll(hull, -1, 0) - hull, axis=1)
    per = float(seg.sum())
    k = int(np.clip(round(per / step), 8, max_points))
    s = np.r_[0, np.cumsum(seg)]
    loop = np.vstack([hull, hull[:1]])
    ts = np.linspace(0, per, k, endpoint=False)
    p2 = np.c_[np.interp(ts, s, loop[:, 0]), np.interp(ts, s, loop[:, 1])][::-1]  # clockwise
    p3 = c + p2[:, :1] * u + p2[:, 1:] * v
    extent = np.linalg.norm(proj - c, axis=1).max() + 3 * radius + 10
    return bone.closest_on_side(p3, n, c, extent)


def cmd_fibre_hull(args: argparse.Namespace) -> int:
    from mskpipe.io.attachment_vtk import write_points
    from mskpipe.io.mesh_io import read_mesh

    index = json.loads((args.atlas / "atlas.json").read_text(encoding="utf-8"))
    bones: dict[str, Bone] = {}

    def bone(name: str) -> Bone:
        if name not in bones:
            bones[name] = Bone(read_mesh(args.atlas / "bones" / f"{name}.obj"))
        return bones[name]

    rows = []
    for name, rec in sorted(index["muscles"].items()):
        side = name[-1]
        if side not in args.sides:
            continue
        codes = rec.get("codes") or []
        files = [
            f
            for code in codes
            for kind in ("MscFbr", "TndFbr")
            for f in sorted(args.polylines.glob(f"{kind}_{side.upper()}_{code}_*.vtk"))
        ]
        if not files:
            log(f"  {name}: no dissection fibres ({', '.join(codes) or 'no code'}) - skipped")
            continue
        ends = np.unique(np.vstack([chain_ends(f) for f in files]), axis=0)
        kinds = [k for k in AREAS if (rec.get(k) or {}).get("file")]
        if len(kinds) < 2:
            continue
        d = np.stack([bone(rec[k]["bone"]).closest(ends)[1] for k in kinds], 1)
        for j, kind in enumerate(kinds):
            ref = gt_outline(args.atlas, name, kind)
            base = {
                "side": side,
                "muscle": name,
                "area": kind,
                "bone": rec[kind]["bone"],
                "fibre_files": ";".join(f.name for f in files),
                "other_definition_in_tlem2": (name[:-2], kind) in OTHER_DEFINITION,
            }
            sel = ends[(d[:, j] <= args.max_dist) & (d[:, j] <= d[:, 1 - j])]
            if ref is None or len(sel) == 0:
                rows.append(base | {"status": "no_points" if ref is not None else "no_gt"})
                log(f"  {name:24s} {kind}: no fibre end on {rec[kind]['bone']}")
                continue
            on_outline = float(
                np.mean(np.min(np.linalg.norm(sel[:, None] - ref[None], axis=2), axis=1) < 0.5)
            )
            for radius in args.radius:
                pts = hull_outline(sel, bone(rec[kind]["bone"]), radius, args.step)
                write_points(
                    args.out / f"r{radius:g}" / "muscles" / name / f"{rec['stem']}_{kind}.vtk", pts
                )
                m = compare(pts, ref)
                rows.append(
                    base
                    | {
                        "status": "ok",
                        "radius_mm": radius,
                        "fibre_points": len(sel),
                        "fibre_points_on_outline": on_outline,
                    }
                    | m
                )
                log(
                    f"  {name:24s} {kind} r={radius:g}: {len(sel):2d} pts, centroid "
                    f"{m['centroid_dist_mm']:6.2f}, gt2p {m['gt2p_mean']:6.2f}, "
                    f"area x{m['area_ratio']:.2f}"
                )
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "fibre_hull_areas.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    summary = []
    print("\nFibre hull (TLEM 2.0 definition) vs LHDL dissection, median [IQR]:")
    for radius in args.radius:
        for label, flt in (
            ("all", lambda r: True),
            ("without TLEM2 other-definition areas", lambda r: not r["other_definition_in_tlem2"]),
        ):
            for side in [*args.sides, "both"]:
                sel = [
                    r
                    for r in ok
                    if r["radius_mm"] == radius and flt(r) and (side == "both" or r["side"] == side)
                ]
                if not sel:
                    continue
                rec = {"radius_mm": radius, "subset": label, "side": side, "n": len(sel)}
                for mtr in ("centroid_dist_mm", "gt2p_mean", "p2gt_mean", "hd", "area_ratio"):
                    v = np.array([r[mtr] for r in sel], float)
                    v = v[np.isfinite(v)]
                    rec |= {
                        f"{mtr}_median": float(np.median(v)),
                        f"{mtr}_q1": float(np.percentile(v, 25)),
                        f"{mtr}_q3": float(np.percentile(v, 75)),
                    }
                summary.append(rec)
                print(
                    f"  r={radius:g} mm  {side:4s} {label:38s} centroid "
                    f"{median_iqr(r['centroid_dist_mm'] for r in sel)}  "
                    f"area x{median_iqr(r['area_ratio'] for r in sel)}"
                )
    write_csv(args.out / "fibre_hull_summary.csv", summary)
    skipped = sorted({r["muscle"] + " " + r["area"] for r in rows if r.get("status") != "ok"})
    if skipped:
        print("  no fibre points: " + ", ".join(skipped))
    log(f"-> {args.out}")
    return 0


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mirror-atlas", help="mirror the atlas of one side to the other")
    p.add_argument("--atlas", type=Path, default=ATLAS)
    p.add_argument("--out", type=Path, default=MIRROR)
    p.add_argument("--source-side", choices=("r", "l"), default="r")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_mirror_atlas)

    p = sub.add_parser("gt-bones", help="(a) same-side / mirrored atlas on the GT bones")
    p.add_argument("--atlas", type=Path, default=ATLAS, help="LHDL atlas = GT bones + dissection")
    p.add_argument("--mirror", type=Path, default=MIRROR)
    p.add_argument("--side", choices=("r", "l"), default="l")
    p.add_argument(
        "--nonrigid",
        nargs="+",
        default=["none", "coherent_icp"],
        choices=("none", "coherent_icp", "fast_rnrr", "cpd"),
    )
    p.add_argument("--fast-rnrr-exe", default=None)
    p.add_argument("--out", type=Path, default=Path("../paper/stats/e19/gt_bones"))
    p.set_defaults(fn=cmd_gt_bones)

    p = sub.add_parser("pair", help="(a) E18 left runs vs runs with the mirrored atlas")
    p.add_argument("--base", type=Path, default=E18_AREAS, help="gt_compare CSV of E18 (LHDL)")
    p.add_argument("--mirror", type=Path, required=True, help="gt_compare CSV of the mirror runs")
    p.add_argument("--side", choices=("r", "l"), default="l")
    p.add_argument("--out", type=Path, default=Path("../paper/stats/e19/pair"))
    p.set_defaults(fn=cmd_pair)

    p = sub.add_parser("fibre-hull", help="(b) TLEM 2.0-style areas from the LHDL fibres")
    p.add_argument("--atlas", type=Path, default=ATLAS)
    p.add_argument("--polylines", type=Path, default=POLYLINES)
    p.add_argument("--sides", nargs="+", choices=("r", "l"), default=["r", "l"])
    p.add_argument(
        "--radius",
        type=float,
        nargs="+",
        default=[4.0],
        help="outline buffer (mm); TLEM 2.0 conversion used 4",
    )
    p.add_argument("--step", type=float, default=4.0, help="outline point spacing (mm)")
    p.add_argument(
        "--max-dist",
        type=float,
        default=3.0,
        help="fibre end points closer than this to the area's bone are used (mm)",
    )
    p.add_argument("--out", type=Path, default=Path("../paper/stats/e19/fibre_hull"))
    p.set_defaults(fn=cmd_fibre_hull)

    args = ap.parse_args(argv)
    if args.cmd == "fibre-hull" and min(args.radius) <= 0:
        ap.error("--radius must be > 0 (a hull of 1-2 points needs the buffer)")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
