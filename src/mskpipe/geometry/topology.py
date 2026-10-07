# SPDX-License-Identifier: Apache-2.0
"""Topology checks and repair of closed surfaces (Muscle Wrapping 2.x muscle meshes).

Muscle Wrapping 2.x decomposes a muscle with a harmonic field on the surface between the
origin and insertion outlines; it needs a closed, consistently oriented 2-manifold of a
single component and genus 0 (one iso-contour per level), without degenerate triangles.
Violations crash it, usually without a message.

:func:`check_surface` reports these properties; :func:`repair_surface` fixes what can be
fixed on the mesh itself (duplicate/degenerate triangles, extra components, orientation).
Holes, non-manifold edges and handles (genus > 0) need a new surface from the mask
(:func:`remesh_mask`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mskpipe.io.mesh_io import TriMesh

DEGENERATE_AREA = 1e-10  # mm^2


def _edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique undirected edges, their face counts and directed duplicates."""
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    undirected = np.sort(directed, axis=1)
    edges, counts = np.unique(undirected, axis=0, return_counts=True)
    _, directed_counts = np.unique(directed, axis=0, return_counts=True)
    return edges, counts, directed_counts


def face_areas(mesh: TriMesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def components(mesh: TriMesh) -> np.ndarray:
    """Component label of every face (faces sharing a vertex are connected)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = mesh.n_vertices
    f = mesh.faces
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels[f[:, 0]]


def mean_edge_length(mesh: TriMesh) -> float:
    edges, _, _ = _edges(mesh.faces)
    lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    return float(lengths.mean())


def check_surface(mesh: TriMesh) -> dict[str, Any]:
    """Properties Muscle Wrapping 2.x needs; ``ok`` = all of them hold.

    ``genus`` is computed from the Euler characteristic and is meaningful only for a
    closed manifold of one component (``None`` otherwise).
    """
    if mesh.n_faces == 0:
        return {"ok": False, "problems": ["empty"], "n_faces": 0}
    f = mesh.faces
    repeated = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    degenerate = int((repeated | (face_areas(mesh) < DEGENERATE_AREA)).sum())
    _, dup_counts = np.unique(np.sort(f, axis=1), axis=0, return_counts=True)
    edges, counts, directed_counts = _edges(f)
    n_comp = len(np.unique(components(mesh)))
    used = len(np.unique(f))
    out: dict[str, Any] = {
        "n_vertices": mesh.n_vertices,
        "n_faces": mesh.n_faces,
        "components": n_comp,
        "boundary_edges": int((counts == 1).sum()),
        "non_manifold_edges": int((counts > 2).sum()),
        "misoriented_edges": int((directed_counts > 1).sum()),
        "degenerate_faces": degenerate,
        "duplicate_faces": int((dup_counts > 1).sum()),
        "unused_vertices": mesh.n_vertices - used,
    }
    closed = out["boundary_edges"] == 0 and out["non_manifold_edges"] == 0
    euler = used - len(edges) + mesh.n_faces
    out["euler"] = int(euler)
    out["genus"] = int((2 - euler) // 2) if closed and n_comp == 1 else None
    vol = _signed_volume(mesh)
    out["volume"] = float(vol)
    problems = [
        f"{k.replace('_', ' ')}: {out[k]}"
        for k in (
            "boundary_edges",
            "non_manifold_edges",
            "misoriented_edges",
            "degenerate_faces",
            "duplicate_faces",
        )
        if out[k]
    ]
    if n_comp != 1:
        problems.append(f"components: {n_comp}")
    if out["genus"] not in (0, None):
        problems.append(f"genus: {out['genus']}")
    if closed and out["misoriented_edges"] == 0 and vol < 0:
        problems.append("inward orientation")
    out["problems"] = problems
    out["ok"] = not problems
    return out


def _signed_volume(mesh: TriMesh) -> float:
    tri = mesh.vertices[mesh.faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def compact(vertices: np.ndarray, faces: np.ndarray) -> TriMesh:
    """Drop vertices not used by any face."""
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return TriMesh(vertices[used], remap[faces])


def repair_surface(mesh: TriMesh) -> tuple[TriMesh, list[str]]:
    """Mesh-level repairs; returns the mesh and the list of applied fixes.

    Coincident vertices are merged, degenerate and duplicate triangles removed, only the
    largest component kept and the orientation made consistent and outward.
    """
    fixes: list[str] = []
    v, f = mesh.vertices, mesh.faces
    uniq, inverse = np.unique(v, axis=0, return_inverse=True)
    if len(uniq) < len(v):
        fixes.append(f"merged {len(v) - len(uniq)} coincident vertices")
        v, f = uniq, inverse.reshape(-1)[f]
    repeated = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    bad = repeated | (area < DEGENERATE_AREA)
    if bad.any():
        fixes.append(f"removed {int(bad.sum())} degenerate faces")
        f = f[~bad]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    if len(first) < len(f):
        fixes.append(f"removed {len(f) - len(first)} duplicate faces")
        f = f[np.sort(first)]
    out = compact(v, f)
    labels = components(out)
    sizes = np.bincount(labels)
    if len(np.flatnonzero(sizes)) > 1:
        keep = labels == np.argmax(sizes)
        fixes.append(f"removed {int(len(np.flatnonzero(sizes)) - 1)} small components")
        out = compact(out.vertices, out.faces[keep])
    oriented, flipped = orient(out)
    if flipped:
        fixes.append(f"reoriented {flipped} faces")
    return oriented, fixes


def orient(mesh: TriMesh) -> tuple[TriMesh, int]:
    """Consistent winding over manifold edges (BFS), outward for a positive volume.

    Returns the mesh and the number of flipped faces. Non-manifold edges are not crossed.
    """
    f = mesh.faces.copy()
    n = len(f)
    if n == 0:
        return mesh, 0
    directed = f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    key = np.sort(directed, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    k_sorted = key[order]
    same = np.all(k_sorted[1:] == k_sorted[:-1], axis=1)
    # manifold edges: exactly two half-edges with this key
    starts = np.flatnonzero(np.r_[True, ~same])
    lengths = np.diff(np.r_[starts, len(order)])
    pairs = order[starts[lengths == 2]], order[starts[lengths == 2] + 1]
    a, b = pairs[0] // 3, pairs[1] // 3
    neighbours: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for i, j, ha, hb in zip(a, b, pairs[0], pairs[1], strict=True):
        consistent = int(not np.array_equal(directed[ha], directed[hb]))
        neighbours[i].append((j, consistent))
        neighbours[j].append((i, consistent))
    flip = np.full(n, -1, dtype=np.int8)
    for seed in range(n):
        if flip[seed] >= 0:
            continue
        flip[seed] = 0
        stack = [seed]
        while stack:
            i = stack.pop()
            for j, consistent in neighbours[i]:
                want = flip[i] if consistent else 1 - flip[i]
                if flip[j] < 0:
                    flip[j] = want
                    stack.append(j)
    f[flip == 1] = f[flip == 1][:, ::-1]
    out = TriMesh(mesh.vertices, f)
    flipped = int((flip == 1).sum())
    if _signed_volume(out) < 0:
        out = TriMesh(mesh.vertices, f[:, ::-1])
        flipped = n - flipped
    return out, flipped


def remesh_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    params: Any,
    closing_mm: float = 0.0,
    opening_mm: float = 0.0,
    *,
    offset: Sequence[int],
    full_shape: Sequence[int],
) -> TriMesh:
    """New surface from a binary mask: closing (``closing_mm``) -> 3D hole filling ->
    opening (``opening_mm``) -> largest 6-connected component -> the ``mesh`` filter
    chain with topology-preserving decimation.

    Closing fills tunnels; opening cuts thin bridges (e.g. a muscle sheet one to three
    voxels thick left next to a bone, connected only through edges and corners, which
    gives handles that closing cannot remove). ``mask`` must have a zero margin of at
    least the larger radius (in voxels) + 1.
    """
    from scipy import ndimage

    from mskpipe.geometry.surface import mask_to_surface
    from mskpipe.labelmap.clean import structuring_element

    spacing = np.linalg.norm(np.asarray(affine, dtype=float)[:3, :3], axis=0)
    work = np.asarray(mask, dtype=bool)
    if closing_mm > 0:
        work = ndimage.binary_closing(work, structure=structuring_element(closing_mm, spacing))
    work = ndimage.binary_fill_holes(work)
    if opening_mm > 0:
        work = ndimage.binary_opening(work, structure=structuring_element(opening_mm, spacing))
    labels, n = ndimage.label(work)
    if n == 0:
        from mskpipe.geometry.surface import SurfaceError

        raise SurfaceError("mask is empty after the morphological operations")
    if n > 1:
        work = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    params = params.model_copy(update={"min_component_fraction": 1.0})
    result = mask_to_surface(
        work, affine, params, offset=offset, full_shape=full_shape, decimation="topology"
    )
    return result.mesh


OPENINGS_MM = (1.0, 1.5, 2.0, 3.0)
CLOSING_STEP_MM = 1.5


def repair_candidates(max_closing_mm: float, max_opening_mm: float) -> list[tuple[float, float]]:
    """(closing_mm, opening_mm) in the order they are tried: nothing, openings, closings,
    then closing + opening."""
    n = int(np.floor(max_closing_mm / CLOSING_STEP_MM + 1e-9))
    closings = [round(i * CLOSING_STEP_MM, 6) for i in range(1, n + 1)]
    openings = [o for o in OPENINGS_MM if o <= max_opening_mm + 1e-9]
    return [
        (0.0, 0.0),
        *((0.0, o) for o in openings),
        *((c, 0.0) for c in closings),
        *((c, o) for c in closings for o in openings),
    ]


@dataclass
class MaskRepair:
    """Result of :func:`repair_mask`: the accepted mesh (``ok``) or the best attempt."""

    mesh: TriMesh | None
    ok: bool
    attempts: list[dict[str, Any]]
    best: int | None  # index of ``mesh`` in ``attempts``


def _rank(check: dict[str, Any], volume_ok: bool, ratio: float) -> tuple:
    """Smaller is better: components, genus, open/non-manifold edges, orientation, volume."""
    genus = check.get("genus")
    return (
        check.get("components", 99),
        99 if genus is None else genus,
        check.get("boundary_edges", 0) + check.get("non_manifold_edges", 0),
        check.get("misoriented_edges", 0),
        0 if volume_ok else 1,
        abs(1.0 - ratio),
    )


def repair_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    params: Any,
    *,
    offset: Sequence[int],
    full_shape: Sequence[int],
    reference_volume: float,
    max_closing_mm: float,
    max_opening_mm: float,
    max_volume_change: float,
    check_cancel: Any = None,
) -> MaskRepair:
    """Remesh ``mask`` with the candidates of :func:`repair_candidates` until one passes
    :func:`check_surface` and keeps its volume within ``max_volume_change`` (fraction) of
    ``reference_volume``. Without such a candidate the best attempt is returned with
    ``ok = False``."""
    from mskpipe.geometry.surface import SurfaceError

    attempts: list[dict[str, Any]] = []
    best: tuple[tuple, int, TriMesh] | None = None
    for closing, opening in repair_candidates(max_closing_mm, max_opening_mm):
        if check_cancel is not None:
            check_cancel()
        rec: dict[str, Any] = {"method": "remesh", "closing_mm": closing, "opening_mm": opening}
        try:
            mesh = remesh_mask(
                mask,
                affine,
                params,
                closing,
                opening,
                offset=offset,
                full_shape=full_shape,
            )
        except SurfaceError as exc:
            attempts.append({**rec, "fixes": [], "problems": [f"remesh failed: {exc}"]})
            continue
        mesh, fixes = repair_surface(mesh)
        check = check_surface(mesh)
        ratio = abs(check["volume"]) / reference_volume if reference_volume > 0 else 1.0
        volume_ok = abs(1.0 - ratio) <= max_volume_change
        problems = list(check["problems"])
        if not volume_ok:
            problems.append(
                f"volume change {100 * (ratio - 1):+.1f} % (limit {100 * max_volume_change:.0f} %)"
            )
        accepted = check["ok"] and volume_ok
        attempts.append(
            {
                **rec,
                "fixes": fixes,
                "genus": check.get("genus"),
                "components": check.get("components"),
                "volume_ratio": round(ratio, 4),
                "problems": problems,
                "accepted": accepted,
            }
        )
        if accepted:
            return MaskRepair(mesh, True, attempts, len(attempts) - 1)
        key = _rank(check, volume_ok, ratio)
        if best is None or key < best[0]:
            best = (key, len(attempts) - 1, mesh)
    if best is None:
        return MaskRepair(None, False, attempts, None)
    return MaskRepair(best[2], False, attempts, best[1])


__all__ = [
    "MaskRepair",
    "check_surface",
    "compact",
    "components",
    "face_areas",
    "mean_edge_length",
    "orient",
    "remesh_mask",
    "repair_candidates",
    "repair_mask",
    "repair_surface",
]
