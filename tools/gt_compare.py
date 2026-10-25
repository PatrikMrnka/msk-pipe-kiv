#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compare pipeline outputs with ground-truth (GT) surfaces and attachment areas.

One GT map (YAML) describes a dataset: which GT files belong to which pipeline structure,
the GT coordinate frame and the attachment references. Every run given on the command
line whose subject matches the map is compared; the results go to CSV files ready for
``pandas`` / the paper.

Metrics per structure (bones and muscles):

* ``p2gt_*`` - distance of the pipeline surface to the GT surface (mean, p95, max);
  small = the segmented shape is correct where it exists,
* ``gt2p_*`` - distance of the GT surface to the pipeline surface; large where the
  pipeline misses part of the structure (tendons of muscles),
* ``assd`` (mean of both directions), ``hd95`` (max of both p95), ``hd``,
* ``dice``/``jaccard`` after voxelising both closed surfaces on a common grid,
  ``volume_ratio`` = pipeline / GT,
* ``gap_proximal_mm``/``gap_distal_mm`` (muscles) - how far the GT extends beyond the
  pipeline surface along the long axis of the GT muscle (positive = missing part).

Distances are evaluated on area-uniform random samples of the surfaces (fixed seed), so
coarse or decimated meshes are not over-weighted at dense vertices.

Attachments: every ``<muscle>/*_<Ori|Ins>.vtk`` of ``05_attachments`` against GT point
sets (contours or single points): centroid distance and nearest-point distances in both
directions. On LHDL the attachment atlas *is* LHDL, so these numbers measure the
registration/projection error only.

Usage::

    python tools/gt_compare.py --gt paper/experiments/gt/lhdl.yaml RUN [RUN ...] --out DIR

A RUN can also be a folder with several runs (e.g. ``runs``); runs are found one level
down. ``--stage mw2`` compares the repaired muscle meshes of ``06_mw2_input`` instead of
``03_mesh`` (rotated back from the OpenSim frame to RAS).
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

FRAMES = {"ras": np.eye(3), "lps": np.diag([-1.0, -1.0, 1.0])}
# 06_mw2_input is written in the OpenSim Y-up frame: X = anterior (y RAS), Y = superior
# (z RAS), Z = right (x RAS). Row vectors: opensim = ras @ RAS_TO_OPENSIM.T.
RAS_TO_OPENSIM = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
AREAS = ("Ori", "Ins")
FRAME_WARN_MM = 20.0


# --------------------------------------------------------------------------- meshes


@dataclass
class Mesh:
    vertices: np.ndarray  # (n, 3) float64, mm
    faces: np.ndarray  # (m, 3) int64

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def transformed(self, rotation: np.ndarray, translation: np.ndarray | None = None) -> Mesh:
        v = self.vertices @ rotation.T
        if translation is not None:
            v = v + translation
        f = self.faces if np.linalg.det(rotation) > 0 else self.faces[:, ::-1].copy()
        return Mesh(v, f)


def merge(meshes: list[Mesh]) -> Mesh:
    verts, faces, offset = [], [], 0
    for m in meshes:
        verts.append(m.vertices)
        faces.append(m.faces + offset)
        offset += len(m.vertices)
    return Mesh(np.vstack(verts), np.vstack(faces))


def _weld(points: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    key = np.round(points / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return points[first], inverse.reshape(-1)


def read_obj(path: Path) -> Mesh:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                faces.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    return Mesh(np.asarray(verts, float).reshape(-1, 3), np.asarray(faces, np.int64).reshape(-1, 3))


def read_stl(path: Path) -> Mesh:
    data = path.read_bytes()
    n = int.from_bytes(data[80:84], "little") if len(data) >= 84 else -1
    if n >= 0 and len(data) == 84 + 50 * n:
        rec = np.frombuffer(
            data,
            dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
            count=n,
            offset=84,
        )
        tri = rec["v"].astype(float)
    else:
        nums = [
            line.split()[1:4]
            for line in data.decode("utf-8", "replace").splitlines()
            if line.strip().startswith("vertex")
        ]
        tri = np.asarray(nums, float).reshape(-1, 3, 3)
    verts, inverse = _weld(tri.reshape(-1, 3))
    return Mesh(verts, inverse.reshape(-1, 3).astype(np.int64))


def _vtk_polydata(path: Path):
    import vtk

    suffix = path.suffix.lower()
    if suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif suffix == ".ply":
        reader = vtk.vtkPLYReader()
    else:
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def read_mesh(path: Path) -> Mesh:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return read_obj(path)
    if suffix == ".stl":
        return read_stl(path)
    import vtk
    from vtk.util import numpy_support as ns

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(_vtk_polydata(path))
    tri.Update()
    pd = tri.GetOutput()
    verts = ns.vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    cells = ns.vtk_to_numpy(pd.GetPolys().GetConnectivityArray()).astype(np.int64)
    return Mesh(verts, cells.reshape(-1, 3))


def read_points(path: Path) -> np.ndarray:
    """Points of an attachment reference: legacy VTK, CSV with x,y,z columns, or a mesh."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header = [c.strip().lower() for c in rows[0]]
        if {"x", "y", "z"} <= set(header):
            ix = [header.index(c) for c in ("x", "y", "z")]
            return np.asarray([[float(r[i]) for i in ix] for r in rows[1:] if r], float)
        return np.asarray([[float(c) for c in r[:3]] for r in rows if r], float)
    if suffix in (".obj", ".stl"):
        return read_mesh(path).vertices
    from vtk.util import numpy_support as ns

    pd = _vtk_polydata(path)
    if pd.GetPoints() is None:
        return np.zeros((0, 3))
    return ns.vtk_to_numpy(pd.GetPoints().GetData()).astype(float)


# ------------------------------------------------------------------------ geometry


def face_areas(m: Mesh) -> np.ndarray:
    t = m.vertices[m.faces]
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def volume(m: Mesh) -> float:
    t = m.vertices[m.faces]
    return float(abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6.0)


def surface_centroid(m: Mesh) -> np.ndarray:
    a = face_areas(m)
    return (m.vertices[m.faces].mean(axis=1) * a[:, None]).sum(0) / a.sum()


def edge_counts(m: Mesh) -> tuple[int, int]:
    """Boundary edges (1 face = hole) and non-manifold edges (> 2 faces)."""
    e = np.sort(m.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts > 2).sum())


def boundary_edges(m: Mesh) -> int:
    return edge_counts(m)[0]


def sample_surface(m: Mesh, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = face_areas(m)
    idx = rng.choice(len(a), size=n, p=a / a.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    t = m.vertices[m.faces[idx]]
    return (
        (1 - s)[:, None] * t[:, 0] + (s * (1 - r2))[:, None] * t[:, 1] + (s * r2)[:, None] * t[:, 2]
    )


def to_vtk(m: Mesh):
    import vtk
    from vtk.util import numpy_support as ns

    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(m.vertices), deep=1))
    cells = vtk.vtkCellArray()
    offsets = np.arange(0, 3 * m.n_faces + 1, 3, dtype=np.int64)
    cells.SetData(
        ns.numpy_to_vtkIdTypeArray(offsets, deep=1),
        ns.numpy_to_vtkIdTypeArray(np.ascontiguousarray(m.faces.reshape(-1)), deep=1),
    )
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    return pd


def sequential_vtk() -> None:
    """Run VTK's SMP loops on one thread.

    ``vtkImplicitPolyDataDistance`` is not thread-safe, but ``FunctionValue`` on an array
    runs through vtkSMPTools: with the STDThread backend (default of the Windows wheels) it
    crashes the process without a message or returns wrong distances.
    """
    import vtk

    if vtk.vtkSMPTools.GetBackend() != "Sequential":
        vtk.vtkSMPTools.SetBackend("Sequential")


def distance_to_surface(points: np.ndarray, m: Mesh) -> np.ndarray:
    import vtk
    from vtk.util import numpy_support as ns

    sequential_vtk()
    impl = vtk.vtkImplicitPolyDataDistance()
    impl.SetInput(to_vtk(m))
    out = vtk.vtkDoubleArray()
    impl.FunctionValue(ns.numpy_to_vtk(np.ascontiguousarray(points), deep=1), out)
    return np.abs(ns.vtk_to_numpy(out))


VOXEL_VOLUME_TOL = 0.03


def voxel_overlap(a: Mesh, b: Mesh, spacing: float) -> tuple[float, float]:
    """Dice and Jaccard of two closed surfaces voxelised on a common grid.

    A few non-manifold edges (common in the LHDL GT) do not disturb the voxelisation; a
    mask whose volume differs from the mesh volume by more than 3 % is not trusted (NaN).
    """
    import vtk
    from vtk.util import numpy_support as ns

    lo = np.minimum(a.vertices.min(0), b.vertices.min(0)) - 2 * spacing
    hi = np.maximum(a.vertices.max(0), b.vertices.max(0)) + 2 * spacing
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1
    masks = []
    for m in (a, b):
        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputData(to_vtk(m))
        stencil.SetOutputOrigin(*lo)
        stencil.SetOutputSpacing(spacing, spacing, spacing)
        stencil.SetOutputWholeExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
        img = vtk.vtkImageStencilToImage()
        img.SetInputConnection(stencil.GetOutputPort())
        img.SetInsideValue(1)
        img.SetOutsideValue(0)
        img.SetOutputScalarTypeToUnsignedChar()
        img.Update()
        mask = ns.vtk_to_numpy(img.GetOutput().GetPointData().GetScalars()).astype(bool)
        mesh_volume = volume(m)
        if not mesh_volume or abs(mask.sum() * spacing**3 / mesh_volume - 1) > VOXEL_VOLUME_TOL:
            return math.nan, math.nan
        masks.append(mask)
    inter = float(np.logical_and(*masks).sum())
    total = float(masks[0].sum() + masks[1].sum())
    union = float(np.logical_or(*masks).sum())
    return (2 * inter / total if total else math.nan), (inter / union if union else math.nan)


def _stats(prefix: str, d: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(d.mean()),
        f"{prefix}_p95": float(np.percentile(d, 95)),
        f"{prefix}_max": float(d.max()),
    }


def end_gaps(gt_pts: np.ndarray, p_pts: np.ndarray) -> dict[str, float]:
    """Overhang of GT beyond the pipeline surface at both ends of the GT long axis."""
    c = gt_pts.mean(0)
    _, _, vt = np.linalg.svd(gt_pts - c, full_matrices=False)
    axis = vt[0] if vt[0][2] >= 0 else -vt[0]  # towards superior (RAS z)
    g, p = (gt_pts - c) @ axis, (p_pts - c) @ axis
    return {
        "gap_proximal_mm": float(g.max() - p.max()),
        "gap_distal_mm": float(p.min() - g.min()),
        "gt_length_mm": float(g.max() - g.min()),
        "length_mm": float(p.max() - p.min()),
    }


def compare_surfaces(pipe: Mesh, gt: Mesh, kind: str, n: int, spacing: float, seed: int) -> dict:
    sp, sg = sample_surface(pipe, n, seed), sample_surface(gt, n, seed + 1)
    d_pg, d_gp = distance_to_surface(sp, gt), distance_to_surface(sg, pipe)
    (b_p, nm_p), (b_g, nm_g) = edge_counts(pipe), edge_counts(gt)
    out: dict[str, Any] = {
        "n_faces": pipe.n_faces,
        "gt_n_faces": gt.n_faces,
        "boundary_edges": b_p,
        "gt_boundary_edges": b_g,
        "non_manifold_edges": nm_p,
        "gt_non_manifold_edges": nm_g,
    }
    out |= _stats("p2gt", d_pg) | _stats("gt2p", d_gp)
    out["assd"] = float((d_pg.sum() + d_gp.sum()) / (len(d_pg) + len(d_gp)))
    out["hd95"] = max(out["p2gt_p95"], out["gt2p_p95"])
    out["hd"] = max(out["p2gt_max"], out["gt2p_max"])
    out["volume_ml"], out["gt_volume_ml"] = volume(pipe) / 1000, volume(gt) / 1000
    out["volume_ratio"] = (
        out["volume_ml"] / out["gt_volume_ml"] if out["gt_volume_ml"] else math.nan
    )
    out["centroid_shift_mm"] = float(np.linalg.norm(surface_centroid(pipe) - surface_centroid(gt)))
    if out["boundary_edges"] == 0 and out["gt_boundary_edges"] == 0:
        out["dice"], out["jaccard"] = voxel_overlap(pipe, gt, spacing)
    else:
        out["dice"] = out["jaccard"] = math.nan
    if kind == "muscle":
        out |= end_gaps(sg, sp)
    return out


def compare_points(pipe: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    from scipy.spatial import cKDTree

    d_gp = cKDTree(pipe).query(gt)[0]
    d_pg = cKDTree(gt).query(pipe)[0]
    return {
        "n_points": len(pipe),
        "gt_n_points": len(gt),
        "centroid_dist_mm": float(np.linalg.norm(pipe.mean(0) - gt.mean(0))),
        "gt2p_mean": float(d_gp.mean()),
        "gt2p_max": float(d_gp.max()),
        "p2gt_mean": float(d_pg.mean()),
        "p2gt_max": float(d_pg.max()),
        "hd": float(max(d_gp.max(), d_pg.max())),
    }


# --------------------------------------------------------------------------- GT map


@dataclass
class GtMap:
    path: Path
    dataset: str
    root: Path
    rotation: np.ndarray
    translation: np.ndarray
    subjects: list[str]
    structures: dict[str, dict[str, Any]]
    attachments: dict[str, dict[str, list[str]]]
    attachments_glob: dict[str, str]
    sample_points: int
    voxel_mm: float

    @classmethod
    def load(cls, path: Path) -> GtMap:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = Path(raw.get("root", "."))
        if not root.is_absolute():
            root = (path.parent / root).resolve()
        frame = str(raw.get("frame", "ras")).lower()
        if frame not in FRAMES:
            sys.exit(f"{path}: frame must be one of {sorted(FRAMES)}")
        rot, trans = FRAMES[frame].copy(), np.zeros(3)
        if raw.get("transform") is not None:  # 4x4, applied after the frame conversion
            t = np.asarray(raw["transform"], float)
            rot, trans = t[:3, :3] @ rot, t[:3, 3]
        return cls(
            path,
            str(raw.get("dataset", path.stem)),
            root,
            rot,
            trans,
            list(raw.get("subjects") or ["*"]),
            dict(raw.get("structures") or {}),
            dict(raw.get("attachments") or {}),
            dict(raw.get("attachments_glob") or {}),
            int(raw.get("sample_points", 30000)),
            float(raw.get("voxel_mm", 0.75)),
        )

    def resolve(self, patterns: list[str] | str) -> list[Path]:
        if isinstance(patterns, str):
            patterns = [patterns]
        files: list[Path] = []
        for pat in patterns:
            hits = sorted(self.root.glob(pat))
            if len(hits) > 1:
                log(
                    f"  note: '{pat}' matches {len(hits)} files, all used: "
                    + ", ".join(h.name for h in hits)
                )
            files += hits
        return files

    def gt_mesh(self, name: str) -> tuple[Mesh | None, list[Path]]:
        files = self.resolve(self.structures[name]["files"])
        if not files:
            return None, []
        mesh = merge([read_mesh(f) for f in files])
        return mesh.transformed(self.rotation, self.translation), files

    def gt_area(self, muscle: str, area: str) -> tuple[np.ndarray | None, list[Path]]:
        spec = self.attachments.get(muscle, {}).get(area)
        if spec is None and area in self.attachments_glob:
            spec = self.attachments_glob[area].replace("{muscle}", muscle)
        if spec is None:
            return None, []
        if isinstance(spec, dict) and "points" in spec:
            pts, files = np.asarray(spec["points"], float).reshape(-1, 3), []
        else:
            files = self.resolve(spec)
            if not files:
                return None, []
            pts = np.vstack([read_points(f) for f in files])
        return pts @ self.rotation.T + self.translation, files


# ----------------------------------------------------------------------------- runs


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def find_runs(paths: list[Path]) -> list[Path]:
    runs: list[Path] = []
    for p in paths:
        if (p / "03_mesh").is_dir():
            runs.append(p)
        elif p.is_dir():
            runs += sorted(c for c in p.iterdir() if (c / "03_mesh").is_dir())
        else:
            log(f"skip {p}: not a run folder")
    return runs


def _get(d: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def run_info(run: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"run": run.name}
    inp = run / "input.json"
    spec = json.loads(inp.read_text(encoding="utf-8")) if inp.is_file() else {}
    spec = spec.get("input", spec) if isinstance(spec, dict) else {}
    info["subject"] = spec.get("subject_id") or spec.get("subject") or run.name.split("_", 2)[-1]
    info["modality"] = spec.get("modality")
    cfg_path = run / "config.resolved.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    info["side"] = _get(cfg, "skeleton.side")
    info["bones_source"] = _get(cfg, "segmentation.bones")
    info["muscles_source"] = _get(cfg, "segmentation.muscles")
    info["attachments_method"] = _get(cfg, "attachments.method")
    info["nonrigid"] = _get(cfg, "attachments.params.nonrigid")
    return info


def pipeline_mesh(run: Path, name: str, kind: str, stage: str) -> Path | None:
    if stage == "mw2":
        if kind != "muscle":
            return None
        objs = sorted((run / "06_mw2_input" / "muscles" / name).glob("*.obj"))
        return max(objs, key=lambda p: p.stat().st_size) if objs else None
    sub = "bones" if kind == "bone" else "muscles"
    for d in (sub, "muscles" if sub == "bones" else "bones"):
        for ext in (".stl", ".obj"):
            p = run / "03_mesh" / d / f"{name}{ext}"
            if p.is_file():
                return p
    return None


def compare_run(run: Path, gt: GtMap, stage: str, seed: int) -> tuple[list[dict], list[dict]]:
    info = run_info(run)
    base = {"dataset": gt.dataset, "stage": stage} | info
    rows, area_rows = [], []
    bone_shifts = []
    side = info.get("side")
    for name, spec in gt.structures.items():
        # the label map holds only the modelled side (+ pelvis, sacrum)
        if side in ("l", "r") and name[-2:] in ("_l", "_r") and name[-1] != side:
            continue
        kind = spec.get("kind", "muscle")
        row = base | {"structure": name, "kind": kind, "note": spec.get("note", "")}
        p = pipeline_mesh(run, name, kind, stage)
        if p is None:
            if stage == "mesh" or kind == "muscle":
                rows.append(row | {"status": "missing_in_run"})
            continue
        gmesh, files = gt.gt_mesh(name)
        if gmesh is None:
            rows.append(row | {"status": "missing_gt"})
            continue
        pmesh = read_mesh(p)
        if stage == "mw2":
            pmesh = pmesh.transformed(RAS_TO_OPENSIM.T)
        m = compare_surfaces(pmesh, gmesh, kind, gt.sample_points, gt.voxel_mm, seed)
        rows.append(row | {"status": "ok"} | m | {"gt_files": ";".join(f.name for f in files)})
        if kind == "bone":
            bone_shifts.append(m["centroid_shift_mm"])
        log(
            f"  {name:28s} assd {m['assd']:6.2f}  hd95 {m['hd95']:6.2f}  "
            f"dice {m['dice']:.3f}  V/Vgt {m['volume_ratio']:.3f}"
        )
    if bone_shifts and min(bone_shifts) > FRAME_WARN_MM:
        log(
            f"  WARNING: every bone centroid is > {FRAME_WARN_MM:.0f} mm from GT - GT is "
            "probably not in the image frame (check 'frame'/'transform' in the GT map)"
        )
    att = run / "05_attachments"
    if stage == "mesh" and att.is_dir():
        for mdir in sorted(d for d in att.iterdir() if d.is_dir()):
            for area in AREAS:
                files = sorted(mdir.glob(f"*_{area}.vtk"))
                if not files:
                    continue
                gpts, gfiles = gt.gt_area(mdir.name, area)
                if gpts is None or len(gpts) == 0:
                    continue
                pts = np.vstack([read_points(f) for f in files])
                area_rows.append(
                    base
                    | {"muscle": mdir.name, "area": area}
                    | compare_points(pts, gpts)
                    | {"gt_files": ";".join(f.name for f in gfiles)}
                )
    return rows, area_rows


# --------------------------------------------------------------------------- output

GROUP = ("dataset", "subject", "modality", "side", "bones_source", "nonrigid", "stage")


def summarize(rows: list[dict], area_rows: list[dict]) -> list[dict]:
    out = []

    def add(src, keys, metrics, label):
        groups: dict[tuple, list[dict]] = {}
        for r in src:
            if r.get("status", "ok") == "ok":
                groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
        for g, items in groups.items():
            rec = dict(zip(keys, g, strict=True)) | {"group": label, "n": len(items)}
            for mtr in metrics:
                v = np.array([float(i[mtr]) for i in items if i.get(mtr) is not None], float)
                v = v[np.isfinite(v)]
                if len(v):
                    q1, med, q3 = np.percentile(v, [25, 50, 75])
                    rec |= {f"{mtr}_median": med, f"{mtr}_q1": q1, f"{mtr}_q3": q3}
            out.append(rec)

    for kind in ("bone", "muscle"):
        add(
            [r for r in rows if r.get("kind") == kind],
            GROUP,
            (
                "dice",
                "assd",
                "hd95",
                "p2gt_mean",
                "gt2p_mean",
                "volume_ratio",
                "gap_proximal_mm",
                "gap_distal_mm",
            ),
            kind,
        )
    add(area_rows, GROUP, ("centroid_dist_mm", "gt2p_mean", "p2gt_mean", "hd"), "attachment")
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def image_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """World (RAS, mm) bounding box of a NIfTI volume."""
    import nibabel as nib

    img = nib.load(str(path))
    n = np.asarray(img.shape[:3]) - 1
    corners = np.array([[i, j, k] for i in (0, n[0]) for j in (0, n[1]) for k in (0, n[2])])
    world = corners @ img.affine[:3, :3].T + img.affine[:3, 3]
    return world.min(0), world.max(0)


def check_gt(gt: GtMap, image: Path | None) -> int:
    """List what every entry of the GT map resolves to (and where it lies)."""
    bounds = image_bounds(image) if image else None
    missing = 0
    for name, spec in gt.structures.items():
        mesh, files = gt.gt_mesh(name)
        if mesh is None:
            missing += 1
            print(f"{name:28s} MISSING  {spec['files']}")
            continue
        lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
        where = ""
        if bounds is not None:
            inside = np.all(lo >= bounds[0] - 5) and np.all(hi <= bounds[1] + 5)
            where = "  in image" if inside else "  OUTSIDE image (frame?)"
        print(
            f"{name:28s} {len(files)} file(s), {mesh.n_faces} faces, "
            f"holes/non-manifold edges {edge_counts(mesh)}, "
            f"centre {np.round((lo + hi) / 2, 1)}{where}"
            f"  <- {', '.join(f.name for f in files)}"
        )
    for muscle, areas in gt.attachments.items():
        for area in areas:
            pts, _ = gt.gt_area(muscle, area)
            print(f"{muscle + ' ' + area:28s} " + ("MISSING" if pts is None else f"{len(pts)} pt"))
    if gt.attachments_glob:
        print(f"attachments by pattern: {gt.attachments_glob} (resolved per run)")
    if bounds is not None:
        print(f"image bounds RAS: {np.round(bounds[0], 1)} .. {np.round(bounds[1], 1)}")
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", type=Path, help="run folders or folders of runs")
    ap.add_argument("--gt", required=True, type=Path, help="GT map (YAML)")
    ap.add_argument("--out", type=Path, default=Path("gt_compare"), help="output folder")
    ap.add_argument("--stage", choices=("mesh", "mw2"), default="mesh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="only list what the GT map resolves to")
    ap.add_argument("--image", type=Path, help="with --check: is the GT inside this volume?")
    args = ap.parse_args(argv)
    sequential_vtk()

    gt = GtMap.load(args.gt)
    if args.check:
        return check_gt(gt, args.image)
    if not args.runs:
        ap.error("give at least one run folder (or --check)")
    runs = []
    for r in find_runs(args.runs):
        subject = str(run_info(r)["subject"])
        if any(fnmatch.fnmatch(subject, pat) for pat in gt.subjects):
            runs.append(r)
    if not runs:
        log(f"no run matches subjects {gt.subjects} of {args.gt}")
        return 2
    rows, area_rows = [], []
    for run in runs:
        log(f"{run.name}")
        r, a = compare_run(run, gt, args.stage, args.seed)
        rows += r
        area_rows += a
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{gt.dataset}_{args.stage}"
    write_csv(args.out / f"structures_{tag}.csv", rows)
    write_csv(args.out / f"attachments_{tag}.csv", area_rows)
    write_csv(args.out / f"summary_{tag}.csv", summarize(rows, area_rows))
    (args.out / f"gt_compare_{tag}.json").write_text(
        json.dumps(
            {
                "gt_map": str(args.gt),
                "dataset": gt.dataset,
                "stage": args.stage,
                "seed": args.seed,
                "runs": [str(r) for r in runs],
                "structures": rows,
                "attachments": area_rows,
            },
            indent=1,
            default=float,
        ),
        encoding="utf-8",
    )
    log(
        f"{len(runs)} runs, {len(rows)} structure rows, {len(area_rows)} attachment rows "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
