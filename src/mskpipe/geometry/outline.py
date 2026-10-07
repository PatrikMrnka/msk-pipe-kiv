# SPDX-License-Identifier: Apache-2.0
"""Attachment outlines on the muscle surface, as Muscle Wrapping 2.x will see them.

Muscle Wrapping 2.x projects every outline point onto the closest point of the muscle
surface (no distance limit), joins the projected points in file order into a closed
contour and cuts the surface along it. The cut fails (crash) when the projected contour
collapses (most points land on the same spot, typically when the muscle mesh lacks the
tendon and the attachment is far away), crosses itself, or touches the other attachment.

The checks here are geometric pre-flight tests of these conditions:

* ``distinct``: projected points that stay apart after merging points closer than
  ``MERGE_FRACTION`` of the mean edge length (the insertion tolerance of Muscle Wrapping),
  at least 3 are needed;
* ``span_mm``: largest distance between projected points, at least ``MIN_SPAN_EDGES`` mean
  edge lengths (the cut needs a ring of triangles); ``shrink`` = projected / original span;
* ``crossing``: the projected contour, flattened onto its best-fit plane, crosses itself;
* distance between the origin and insertion contours, at least one mean edge length;
* ``patch_fraction``: area enclosed by the contour / muscle area, must stay below 0.5
  (Muscle Wrapping takes the smaller side of the cut as the attachment patch).

``gap_mm`` (outline point -> muscle surface) measures how far the attachment lies from
the segmented muscle, i.e. the tendon the segmentation did not capture.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MERGE_FRACTION = 0.05
MIN_SPAN_EDGES = 3.0
MAX_PATCH_FRACTION = 0.5


def merge_close(points: np.ndarray, tol: float) -> np.ndarray:
    """Points left after greedily merging points closer than ``tol`` (order kept)."""
    kept: list[np.ndarray] = []
    for p in np.asarray(points, dtype=float):
        if all(np.linalg.norm(p - q) >= tol for q in kept):
            kept.append(p)
    return np.array(kept).reshape(-1, 3)


def polygon_area(points: np.ndarray) -> float:
    """Area of a closed (possibly non-planar) polygon (Newell's method)."""
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return 0.0
    return float(0.5 * np.linalg.norm(np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)))


def _plane_coords(points: np.ndarray) -> np.ndarray:
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Proper intersection of 2D segments ab and cd (shared end points do not count)."""

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def self_crossing(points: np.ndarray, tol: float) -> bool:
    """Does the closed contour (consecutive duplicates within ``tol`` merged) cross itself?"""
    pts = [p for i, p in enumerate(points) if i == 0 or np.linalg.norm(p - points[i - 1]) >= tol]
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < tol:
        pts.pop()
    if len(pts) < 4:
        return False
    flat = _plane_coords(np.array(pts))
    n = len(flat)
    for i in range(n):
        a, b = flat[i], flat[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent through the closing segment
            if _segments_cross(a, b, flat[j], flat[(j + 1) % n]):
                return True
    return False


def contour_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between two closed polylines (point-to-segment, both ways)."""

    def to_polyline(points: np.ndarray, poly: np.ndarray) -> float:
        p0, p1 = poly, np.roll(poly, -1, axis=0)
        seg = p1 - p0
        len2 = np.maximum((seg**2).sum(axis=1), 1e-30)
        best = np.inf
        for p in points:
            t = np.clip(((p - p0) * seg).sum(axis=1) / len2, 0.0, 1.0)
            best = min(best, float(np.linalg.norm(p0 + t[:, None] * seg - p, axis=1).min()))
        return best

    a, b = np.asarray(a, float), np.asarray(b, float)
    return min(to_polyline(a, b), to_polyline(b, a))


def _span(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(points[:, None] - points[None], axis=2)))


def outline_report(
    points: np.ndarray, projected: np.ndarray, gaps: np.ndarray, edge: float, area: float
) -> dict[str, Any]:
    """Pre-flight report of one outline (``projected`` = points on the muscle surface)."""
    tol = MERGE_FRACTION * edge
    distinct = merge_close(projected, tol)
    span = _span(distinct)
    original = _span(np.asarray(points, dtype=float))
    patch = polygon_area(projected) / area if area > 0 else 0.0
    rep: dict[str, Any] = {
        "n_points": len(points),
        "distinct": len(distinct),
        "span_mm": round(span, 3),
        "shrink": round(span / original, 3) if original > 0 else None,
        "crossing": self_crossing(projected, tol),
        "patch_fraction": round(patch, 4),
        "gap_mean_mm": round(float(gaps.mean()), 3),
        "gap_max_mm": round(float(gaps.max()), 3),
    }
    problems = []
    if len(distinct) < 3 or span < MIN_SPAN_EDGES * edge:
        problems.append("collapsed")
    if rep["crossing"]:
        problems.append("crossing")
    if patch >= MAX_PATCH_FRACTION:
        problems.append("patch too large")
    rep["problems"] = problems
    return rep


__all__ = [
    "MAX_PATCH_FRACTION",
    "MERGE_FRACTION",
    "MIN_SPAN_EDGES",
    "contour_distance",
    "merge_close",
    "outline_report",
    "polygon_area",
    "self_crossing",
]
