# SPDX-License-Identifier: Apache-2.0
"""Fast-RNRR (Yao et al., CVPR 2020) as an external non-rigid registration.

Fast-RNRR (https://github.com/Juyong/Fast_RNRR) is research-only and patented, and has
no licence that allows redistribution: mskpipe does not ship it. With the user's consent
``mskpipe setup fast-rnrr`` downloads the authors' source and builds it on the user's
computer (:mod:`mskpipe.resources.fast_rnrr`); an executable built otherwise can be given
instead (``<exe> <src.obj> <tar.obj> <out_prefix>``, result ``<out_prefix>res.obj``).

The executable deforms the vertices of the source mesh (embedded deformation graph,
robust Welsch metric, quasi-Newton solver; its own rigid stage is off in the reference
``main.cpp``). :class:`MeshWarp` turns the deformed mesh into a field usable for any
point near the source surface: barycentric transport on the closest source triangle.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk

from mskpipe.geometry.surface import to_polydata
from mskpipe.io.mesh_io import TriMesh, read_mesh, write_mesh
from mskpipe.resources.fast_rnrr import FAST_RNRR_ENV


class FastRnrrError(RuntimeError):
    """Executable missing or no usable result."""


def find_executable(value: str | os.PathLike[str] | None = None) -> Path:
    """``value``, else ``$MSKPIPE_FAST_RNRR``, else the executable built by
    ``mskpipe setup fast-rnrr``, else ``Fast_RNRR`` on PATH."""
    from mskpipe.resources.fast_rnrr import locate

    path, source = locate(value)
    if path is not None:
        return path
    if source == "config":
        raise FastRnrrError(f"Fast-RNRR executable not found: {value}")
    if source == "env":
        raise FastRnrrError(f"Fast-RNRR executable not found: {os.environ[FAST_RNRR_ENV]}")
    raise FastRnrrError(
        "Fast-RNRR is not distributed with msk-pipe (research-only licence). Build it on "
        "this computer: mskpipe setup fast-rnrr --accept-research-licence (GUI: Setup… -> "
        "Fast-RNRR), or set attachments.params.fast_rnrr_exe or the "
        f"{FAST_RNRR_ENV} environment variable"
    )


def compact(mesh: TriMesh) -> TriMesh:
    """Drop vertices not used by any face (Fast-RNRR fails on isolated points)."""
    used = np.unique(mesh.faces)
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return TriMesh(mesh.vertices[used], remap[mesh.faces])


def _nonmanifold_edge_faces(faces: np.ndarray) -> np.ndarray:
    """Faces on an edge shared by > 2 faces or traversed twice in the same direction."""
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, inv, counts = np.unique(
        np.sort(directed, 1), axis=0, return_inverse=True, return_counts=True
    )
    _, dinv, dcounts = np.unique(directed, axis=0, return_inverse=True, return_counts=True)
    bad = (counts[inv.ravel()] > 2) | (dcounts[dinv.ravel()] > 1)
    out = np.zeros(len(faces), dtype=bool)
    out[np.repeat(np.arange(len(faces)), 3)[bad]] = True
    return out


def _split_nonmanifold_vertices(mesh: TriMesh) -> tuple[TriMesh, int]:
    """Give every fan of faces around a vertex its own copy of the vertex ("bow-ties")."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = mesh.faces
    n_inc = 3 * len(f)  # incidence i = corner i % 3 of face i // 3
    v = f.reshape(-1)
    nxt = f[:, [1, 2, 0]].reshape(-1)
    prv = f[:, [2, 0, 1]].reshape(-1)
    n = np.int64(len(mesh.vertices))
    inc = np.arange(n_inc)
    keys = np.concatenate([v * n + nxt, v * n + prv])  # an edge seen from its vertex v
    owners = np.concatenate([inc, inc])
    order = np.argsort(keys, kind="stable")
    keys, owners = keys[order], owners[order]
    same = keys[1:] == keys[:-1]  # edge-manifold: at most two incidences per key
    graph = coo_matrix(
        (np.ones(int(same.sum())), (owners[:-1][same], owners[1:][same])), shape=(n_inc, n_inc)
    )
    _, fan = connected_components(graph, directed=False)
    pairs = np.unique(np.stack([v, fan], 1), axis=0)  # (vertex, fan), sorted by vertex
    first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
    extra = pairs[~first]
    if not len(extra):
        return mesh, 0
    new_index = np.full(int(fan.max()) + 1, -1, dtype=np.int64)
    new_index[extra[:, 1]] = len(mesh.vertices) + np.arange(len(extra))
    flat = np.where(new_index[fan] >= 0, new_index[fan], v)
    vertices = np.vstack([mesh.vertices, mesh.vertices[extra[:, 0]]])
    return TriMesh(vertices, flat.reshape(-1, 3)), len(extra)


def prepare_mesh(mesh: TriMesh) -> tuple[TriMesh, list[str]]:
    """A mesh OpenMesh (inside Fast-RNRR) reads unchanged; returns it and the applied fixes.

    OpenMesh cannot represent non-manifold edges and vertices: on reading it drops such
    triangles or re-adds them on duplicated vertices, which breaks the vertex
    correspondence and can make Fast-RNRR return NaN for the whole mesh. Degenerate and
    duplicate triangles and triangles on non-manifold or inconsistently oriented edges
    are removed and non-manifold vertices split; vertex order and components are kept
    (a pelvis may be two hip bones), so a prepared mesh comes back unchanged.
    """
    from mskpipe.geometry.topology import DEGENERATE_AREA

    fixes: list[str] = []
    v, f = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    bad = (
        (f[:, 0] == f[:, 1])
        | (f[:, 1] == f[:, 2])
        | (f[:, 0] == f[:, 2])
        | (area < DEGENERATE_AREA)
    )
    if bad.any():
        fixes.append(f"removed {int(bad.sum())} degenerate faces")
        f = f[~bad]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    if len(first) < len(f):
        fixes.append(f"removed {len(f) - len(first)} duplicate faces")
        f = f[np.sort(first)]
    removed = 0
    while len(f):
        bad = _nonmanifold_edge_faces(f)
        if not bad.any():
            break
        removed += int(bad.sum())
        f = f[~bad]
    if removed:
        fixes.append(f"removed {removed} faces on non-manifold edges")
    if not len(f):
        raise FastRnrrError("No usable triangles left for Fast-RNRR")
    out, split = _split_nonmanifold_vertices(compact(TriMesh(v, f)))
    if split:
        fixes.append(f"split {split} non-manifold vertices")
    return out, fixes


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

    Both meshes go through :func:`prepare_mesh` first; pass ``source`` already prepared
    when its vertices must match the result (as :class:`MeshWarp` needs).

    Both meshes are centred on the target centroid first (Fast-RNRR scales by the joint
    bounding box and rejects pairs farther than 5 % of its diagonal); ``run`` executes
    the command (e.g. ``StepContext.run_command``).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    source, _ = prepare_mesh(source)
    target, _ = prepare_mesh(target)
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
    bad = ~np.isfinite(moved[: len(source.vertices)]).all(axis=1)
    if bad.any():
        raise FastRnrrError(
            f"Fast-RNRR returned {int(bad.sum())} of {len(source.vertices)} vertices that are "
            "not finite (NaN/inf); the registration diverged. Try nonrigid: coherent_icp or cpd"
        )
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
    "prepare_mesh",
    "run_fast_rnrr",
]
