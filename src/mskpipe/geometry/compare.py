# SPDX-License-Identifier: Apache-2.0
"""Surface-to-surface distances (validation against reference meshes)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import vtk

from mskpipe.geometry.surface import to_polydata
from mskpipe.io.mesh_io import TriMesh


@dataclass(frozen=True)
class DistanceStats:
    """Symmetric vertex-to-surface distances (units of the meshes)."""

    mean: float
    rms: float
    p95: float
    hausdorff: float
    mean_a_to_b: float
    mean_b_to_a: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class SurfaceLocator:
    """Closest-point queries on one surface (the locator is built once)."""

    def __init__(self, surface: TriMesh) -> None:
        self._poly = to_polydata(surface)  # keep the data set alive with the locator
        self._locator = vtk.vtkStaticCellLocator()
        self._locator.SetDataSet(self._poly)
        self._locator.BuildLocator()

    def closest(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Closest surface point to each point, and its distance."""
        closest = [0.0, 0.0, 0.0]
        cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.empty_like(points)
        dist = np.empty(len(points))
        for i, p in enumerate(points):
            self._locator.FindClosestPoint(p, closest, cell_id, sub_id, dist2)
            out[i] = closest
            dist[i] = dist2.get()
        return out, np.sqrt(dist)


def point_to_surface(points: np.ndarray, surface: TriMesh) -> np.ndarray:
    """Unsigned distance of each point to the closest point of ``surface``."""
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(to_polydata(surface))
    locator.BuildLocator()
    closest = [0.0, 0.0, 0.0]
    cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    out = np.empty(len(points))
    for i, p in enumerate(np.asarray(points, dtype=np.float64)):
        locator.FindClosestPoint(p, closest, cell_id, sub_id, dist2)
        out[i] = dist2.get()
    return np.sqrt(out)


def closest_on_surface(points: np.ndarray, surface: TriMesh) -> tuple[np.ndarray, np.ndarray]:
    """Closest point of ``surface`` to each point, and its distance."""
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(to_polydata(surface))
    locator.BuildLocator()
    closest = [0.0, 0.0, 0.0]
    cell_id, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    points = np.asarray(points, dtype=np.float64)
    out = np.empty_like(points)
    dist = np.empty(len(points))
    for i, p in enumerate(points):
        locator.FindClosestPoint(p, closest, cell_id, sub_id, dist2)
        out[i] = closest
        dist[i] = dist2.get()
    return out, np.sqrt(dist)


def surface_distance(a: TriMesh, b: TriMesh) -> DistanceStats:
    d_ab = point_to_surface(a.vertices, b)
    d_ba = point_to_surface(b.vertices, a)
    both = np.concatenate([d_ab, d_ba])
    return DistanceStats(
        mean=float(both.mean()),
        rms=float(np.sqrt(np.mean(both**2))),
        p95=float(np.percentile(both, 95)),
        hausdorff=float(both.max()),
        mean_a_to_b=float(d_ab.mean()),
        mean_b_to_a=float(d_ba.mean()),
    )


__all__ = [
    "DistanceStats",
    "SurfaceLocator",
    "closest_on_surface",
    "point_to_surface",
    "surface_distance",
]
