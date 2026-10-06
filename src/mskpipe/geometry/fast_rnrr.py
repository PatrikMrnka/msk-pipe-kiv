# SPDX-License-Identifier: Apache-2.0
"""Fast-RNRR (Yao et al., CVPR 2020) as an external non-rigid registration.

Fast-RNRR (https://github.com/Juyong/Fast_RNRR) is research-only and patented, and has
no licence that allows redistribution: mskpipe neither ships nor builds it. The user
points to an executable built by themselves (``<exe> <src.obj> <tar.obj> <out_prefix>``,
result ``<out_prefix>res.obj``).

The executable deforms the vertices of the source mesh (embedded deformation graph,
robust Welsch metric, quasi-Newton solver; its own rigid stage is off in the reference
``main.cpp``). :class:`MeshWarp` turns the deformed mesh into a field usable for any
point near the source surface: barycentric transport on the closest source triangle.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk

from mskpipe.geometry.surface import to_polydata
from mskpipe.io.mesh_io import TriMesh, read_mesh, write_mesh

FAST_RNRR_ENV = "MSKPIPE_FAST_RNRR"
EXECUTABLE_NAMES = ("Fast_RNRR.exe", "Fast_RNRR")


class FastRnrrError(RuntimeError):
    """Executable missing or no usable result."""


def find_executable(value: str | os.PathLike[str] | None = None) -> Path:
    """``value``, else ``$MSKPIPE_FAST_RNRR``, else ``Fast_RNRR`` on PATH."""
    candidate = value or os.environ.get(FAST_RNRR_ENV)
    if candidate:
        path = Path(candidate)
        if path.is_file():
            return path
        raise FastRnrrError(f"Fast-RNRR executable not found: {path}")
    for name in EXECUTABLE_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    raise FastRnrrError(
        "Fast-RNRR is not distributed with mskpipe (research-only licence): build it "
        "(https://github.com/Juyong/Fast_RNRR) and set attachments.params.fast_rnrr_exe "
        f"or the {FAST_RNRR_ENV} environment variable"
    )


def compact(mesh: TriMesh) -> TriMesh:
    """Drop vertices not used by any face (Fast-RNRR fails on isolated points)."""
    used = np.unique(mesh.faces)
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return TriMesh(mesh.vertices[used], remap[mesh.faces])


def decimate(mesh: TriMesh, max_faces: int) -> TriMesh:
    """Quadric decimation to at most ``max_faces`` triangles (unchanged if smaller)."""
    if len(mesh.faces) <= max_faces:
        return mesh
    dec = vtk.vtkQuadricDecimation()
    dec.SetInputData(to_polydata(mesh))
    dec.SetTargetReduction(1.0 - max_faces / len(mesh.faces))
    dec.VolumePreservationOn()
    dec.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(dec.GetOutputPort())
    clean.Update()
    poly = clean.GetOutput()
    from vtk.util.numpy_support import vtk_to_numpy

    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    faces = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(np.int64)
    return compact(TriMesh(points, faces))


def run_fast_rnrr(
    executable: Path,
    source: TriMesh,
    target: TriMesh,
    work_dir: Path,
    run: Callable[[Sequence[str | Path]], None],
) -> TriMesh:
    """Deform ``source`` onto ``target``; returns ``source`` with the moved vertices.

    Both meshes are centred on the target centroid first (Fast-RNRR scales by the joint
    bounding box and rejects pairs farther than 5 % of its diagonal); ``run`` executes
    the command (e.g. ``StepContext.run_command``).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    source = compact(source)
    shift = target.vertices.mean(0)
    src_path = write_mesh(work_dir / "source.obj", TriMesh(source.vertices - shift, source.faces))
    tar_path = write_mesh(work_dir / "target.obj", TriMesh(target.vertices - shift, target.faces))
    prefix = work_dir / "fast_rnrr_"
    result = Path(f"{prefix}res.obj")
    result.unlink(missing_ok=True)
    run([executable, src_path, tar_path, prefix])
    if not result.is_file():
        raise FastRnrrError(f"Fast-RNRR wrote no result ({result})")
    moved = read_mesh(result).vertices
    if len(moved) < len(source.vertices):  # OpenMesh may only append (split) vertices
        raise FastRnrrError(f"Fast-RNRR result has {len(moved)} < {len(source.vertices)} vertices")
    return TriMesh(moved[: len(source.vertices)] + shift, source.faces)


@dataclass
class MeshWarp:
    """Field given by a mesh and its deformed copy (same topology).

    A point is attached to its closest point on ``source`` (barycentric coordinates on
    that triangle) and moves with it; its offset from the surface is kept.
    """

    source: TriMesh
    deformed: np.ndarray  # (V, 3)

    def __post_init__(self) -> None:
        self._locator = vtk.vtkStaticCellLocator()
        self._poly = to_polydata(self.source)
        self._locator.SetDataSet(self._poly)
        self._locator.BuildLocator()

    def displacement(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        closest = [0.0, 0.0, 0.0]
        cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        cells = np.empty(len(points), dtype=np.int64)
        foot = np.empty_like(points)
        for i, p in enumerate(points):
            self._locator.FindClosestPoint(p, closest, cell_id, sub_id, dist2)
            cells[i] = cell_id.get()
            foot[i] = closest
        tri = self.source.faces[cells]
        bary = _barycentric(foot, self.source.vertices[tri])
        disp = self.deformed[tri] - self.source.vertices[tri]
        return np.einsum("nk,nkd->nd", bary, disp)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, float) + self.displacement(points)


def _barycentric(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = np.sum(v0 * v0, 1)
    d01 = np.sum(v0 * v1, 1)
    d11 = np.sum(v1 * v1, 1)
    d20 = np.sum(v2 * v0, 1)
    d21 = np.sum(v2 * v1, 1)
    den = np.where(np.abs(d00 * d11 - d01**2) > 1e-30, d00 * d11 - d01**2, 1e-30)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    out = np.c_[1.0 - v - w, v, w]
    return np.clip(out, 0.0, 1.0) / np.clip(out, 0.0, 1.0).sum(1, keepdims=True)


__all__ = [
    "FAST_RNRR_ENV",
    "FastRnrrError",
    "MeshWarp",
    "compact",
    "decimate",
    "find_executable",
    "run_fast_rnrr",
]
