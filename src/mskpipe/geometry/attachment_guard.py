# SPDX-License-Identifier: Apache-2.0
"""Guard against attachment outlines that collapse on the muscle surface.

Muscle Wrapping 2.x projects every attachment point onto the nearest place of the muscle
and cuts the attachment patch out along that outline. When the bone outline lies far from
the muscle (missing tendon), all points land on one or two vertices; the cut then yields
an empty mesh and MW2 corrupts its heap (``vtkStaticPointLocator: No points to locate``,
exit 0xC0000374; seen for gracilis and sartorius on LHDL CT).

:func:`inflate_outline` replaces such an outline by the boundary of a small connected
patch of the muscle surface around the collapse point: a closed, non-crossing outline
with at least three distinct points lying on mesh vertices. The muscle is kept (the
export only warns); its fibres still end where the tendon would start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mskpipe.io.mesh_io import TriMesh


class GuardError(ValueError):
    """No valid outline could be built on the muscle surface."""


@dataclass(frozen=True)
class OutlineStats:
    n_points: int
    distinct: int
    span_mm: float

    def as_dict(self) -> dict[str, Any]:
        return {"n_points": self.n_points, "distinct": self.distinct, "span_mm": self.span_mm}


@dataclass(frozen=True)
class InflatedOutline:
    points: np.ndarray  # (K, 3), ordered closed outline on mesh vertices
    radius_mm: float  # radius that produced it
    center: np.ndarray  # (3,) seed vertex position
    patch_fraction: float  # patch area / muscle area
    before: OutlineStats
    after: OutlineStats = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "after", outline_stats(self.points))

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius_mm": round(self.radius_mm, 3),
            "center_mm": [round(float(v), 3) for v in self.center],
            "patch_fraction": round(self.patch_fraction, 4),
            "before": self.before.as_dict(),
            "after": self.after.as_dict(),
        }


def outline_stats(points: np.ndarray, tol_mm: float = 1e-3) -> OutlineStats:
    """Number of points, of distinct points (closer than ``tol_mm`` = same) and span (mm)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return OutlineStats(0, 0, 0.0)
    distinct = len(np.unique(np.round(pts / tol_mm).astype(np.int64), axis=0))
    diff = pts[:, None, :] - pts[None, :, :]
    span = float(np.sqrt((diff**2).sum(-1)).max())
    return OutlineStats(len(pts), distinct, round(span, 3))


def is_degenerate(stats: OutlineStats, min_distinct: int = 3, min_span_mm: float = 4.0) -> bool:
    """True when a projected outline would not give MW2 a usable patch."""
    return stats.distinct < min_distinct or stats.span_mm < min_span_mm


def mean_edge_length(mesh: TriMesh) -> float:
    tri = mesh.vertices[mesh.faces]
    edges = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2)
    return float(edges.mean())


def inflate_outline(
    mesh: TriMesh,
    projected: np.ndarray,
    radius_mm: float = 5.0,
    *,
    max_points: int = 40,
    max_fraction: float = 0.25,
    growth: float = 1.5,
    max_steps: int = 8,
) -> InflatedOutline:
    """Outline of a connected surface patch of ``mesh`` around the projected points.

    ``projected`` are the attachment points as MW2 would place them on the muscle; their
    centroid picks the seed vertex. The patch holds the faces whose three vertices lie
    within ``radius_mm`` of the seed and are connected to it; the radius grows by
    ``growth`` until the patch is a disc with a simple boundary of at least three
    vertices. The boundary follows the face orientation and is subsampled to at most
    ``max_points`` points.
    """
    if mesh.n_faces == 0:
        raise GuardError("empty muscle mesh")
    pts = np.asarray(projected, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        raise GuardError("no attachment points")
    before = outline_stats(pts)
    seed = int(np.argmin(np.linalg.norm(mesh.vertices - pts.mean(0), axis=1)))
    center = mesh.vertices[seed]
    dist = np.linalg.norm(mesh.vertices - center, axis=1)
    areas = _face_areas(mesh)
    total = float(areas.sum())

    radius = float(radius_mm)
    for _ in range(max_steps):
        inside = np.flatnonzero(dist[mesh.faces].max(axis=1) <= radius)
        patch = _component(mesh.faces, inside, seed)
        if len(patch):
            fraction = float(areas[patch].sum() / total) if total > 0 else 1.0
            if fraction > max_fraction:
                raise GuardError(
                    f"patch of radius {radius:.1f} mm covers {fraction:.0%} of the muscle"
                )
            loop = _boundary_loop(mesh.faces[patch])
            if loop is not None and len(loop) >= 3:
                loop = _subsample(loop, max_points)
                return InflatedOutline(mesh.vertices[loop], radius, center, fraction, before)
        radius *= growth
    raise GuardError(f"no disc-shaped patch up to radius {radius:.1f} mm")


# ------------------------------------------------------------------------------ helpers


def _face_areas(mesh: TriMesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _component(faces: np.ndarray, candidates: np.ndarray, seed: int) -> np.ndarray:
    """Faces of ``candidates`` edge-connected to a candidate face touching ``seed``."""
    if len(candidates) == 0:
        return candidates
    sub = faces[candidates]
    start = np.flatnonzero((sub == seed).any(axis=1))
    if len(start) == 0:
        return np.zeros(0, dtype=np.int64)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for i, (a, b, c) in enumerate(sub):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(u, v), max(u, v)), []).append(i)
    seen = np.zeros(len(sub), dtype=bool)
    stack = [int(start[0])]
    seen[start[0]] = True
    while stack:
        i = stack.pop()
        a, b, c = sub[i]
        for u, v in ((a, b), (b, c), (c, a)):
            for j in edge_faces[(min(u, v), max(u, v))]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
    return candidates[seen]


def _boundary_loop(faces: np.ndarray) -> np.ndarray | None:
    """Longest boundary loop of a patch, or None if the boundary is not a set of simple loops."""
    half = {(int(u), int(v)) for a, b, c in faces for u, v in ((a, b), (b, c), (c, a))}
    nxt: dict[int, int] = {}
    for u, v in half:
        if (v, u) in half:
            continue
        if u in nxt:  # two boundary edges leave one vertex: pinched patch
            return None
        nxt[u] = v
    if not nxt:
        return None  # closed surface, no boundary
    loops: list[list[int]] = []
    left = set(nxt)
    while left:
        start = left.pop()
        loop, v = [start], nxt[start]
        while v != start:
            if v not in left:
                return None  # open chain: inconsistent orientation
            left.remove(v)
            loop.append(v)
            v = nxt[v]
        loops.append(loop)
    if len(loops) != 1:
        return None  # annulus or several holes: not a disc
    return np.asarray(loops[0], dtype=np.int64)


def _subsample(loop: np.ndarray, max_points: int) -> np.ndarray:
    if len(loop) <= max_points:
        return loop
    idx = np.round(np.linspace(0, len(loop), max_points, endpoint=False)).astype(np.int64)
    return loop[idx]


__all__ = [
    "GuardError",
    "InflatedOutline",
    "OutlineStats",
    "inflate_outline",
    "is_degenerate",
    "mean_edge_length",
    "outline_stats",
]
