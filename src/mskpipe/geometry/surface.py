# SPDX-License-Identifier: Apache-2.0
"""Binary mask -> smoothed, decimated triangle surface (VTK).

Port of the C++ ``mask_to_mesh`` tool of the BP pipeline (Mrnka 2026). With the same
VTK version (9.4) the filter chain gives identical meshes:

1. ``vtkFlyingEdges3D`` isosurface at 0.5,
2. ``vtkCleanPolyData`` + ``vtkPolyDataConnectivityFilter`` (small components removed),
3. ``vtkWindowedSincPolyDataFilter`` (non-manifold smoothing, normalized coordinates),
4. ``vtkTriangleFilter`` + ``vtkCleanPolyData`` (point merging),
5. ``vtkQuadricDecimation``.

Differences to the BP tool, all intentional:

* Output is in world coordinates (NIfTI RAS+, mm). The BP tool used
  ``vtkNIFTIImageReader``, which ignores the image origin and orientation, so its meshes
  were in ``voxel index x spacing``; images with a mirroring affine produced mirrored
  meshes. Filtering still runs in that frame (smoothing and decimation see the same
  numbers as in BP); the rigid part of the affine is applied at the end.
* The mask is zero-padded, so structures cut by the field of view give closed surfaces
  (reported as ``touches_border``).
* Instead of "largest component only" (with a name-based exception for tibia/fibula),
  components smaller than ``min_component_fraction`` of the largest are removed. With a
  single dominant component this is identical to BP; structures made of two similar
  parts (e.g. both hip bones of ``pelvis_no_sacrum``) keep both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from mskpipe.config.schema import MeshParams
from mskpipe.io.mesh_io import TriMesh

ISO_VALUE = 0.5
_PAD = 1


class SurfaceError(ValueError):
    """A surface cannot be extracted (e.g. empty mask)."""


@dataclass(frozen=True, eq=False)
class SurfaceResult:
    mesh: TriMesh
    info: dict[str, Any] = field(default_factory=dict)


def vtk_version() -> str:
    return vtk.vtkVersion.GetVTKVersion()


def set_threads(threads: int | None) -> None:
    """Limit VTK's SMP backend (used by Flying Edges); ``None`` = all cores."""
    vtk.vtkSMPTools.Initialize(threads or 0)


def mask_to_surface(
    mask: np.ndarray,
    affine: np.ndarray,
    params: MeshParams,
    *,
    offset: Sequence[int] = (0, 0, 0),
    full_shape: Sequence[int] | None = None,
) -> SurfaceResult:
    """Extract the surface of a binary ``mask`` (indexed ``[i, j, k]``).

    ``mask`` may be a crop of a larger volume: ``offset`` is the index of its first voxel
    and ``full_shape`` the shape of the whole volume (used for ``touches_border``).
    ``affine`` is the voxel-to-world matrix of the whole volume.
    """
    mask = np.asarray(mask) > 0
    if mask.ndim != 3:
        raise SurfaceError(f"mask must be 3D, got shape {mask.shape}")
    n_voxels = int(mask.sum())
    if n_voxels == 0:
        raise SurfaceError("mask is empty")

    affine = np.asarray(affine, dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    offset_arr = np.asarray(offset, dtype=np.int64)
    shape = np.asarray(full_shape if full_shape is not None else mask.shape)
    nz = np.argwhere(mask)
    touches = bool(
        (nz.min(0) + offset_arr == 0).any() or (nz.max(0) + offset_arr == shape - 1).any()
    )

    image = _to_vtk_image(mask, spacing, offset_arr)
    surface = _isosurface(image)
    cleaned = _clean(surface)
    kept, n_found, n_kept = _filter_components(cleaned, params.min_component_fraction)
    smoothed = _smooth(kept, params.smooth_iterations, params.passband)
    final = _decimate(smoothed, params.target_reduction)

    # Filtering ran in "index x spacing" (the BP frame); apply the rest of the affine.
    to_world = np.eye(4)
    to_world[:3, :3] = affine[:3, :3] / spacing
    to_world[:3, 3] = affine[:3, 3]
    mesh = _to_trimesh(final).transformed(to_world)
    if mesh.n_faces == 0:
        raise SurfaceError("surface is empty after processing")

    return SurfaceResult(
        mesh,
        {
            "n_voxels": n_voxels,
            "mask_volume": float(n_voxels * np.prod(spacing)),
            "touches_border": touches,
            "components_found": n_found,
            "components_kept": n_kept,
            "faces_before_decimation": int(smoothed.GetNumberOfPolys()),
        },
    )


# ---------------------------------------------------------------------------- VTK stages


def _to_vtk_image(mask: np.ndarray, spacing: np.ndarray, offset: np.ndarray) -> vtk.vtkImageData:
    padded = np.pad(mask.astype(np.uint8), _PAD)
    image = vtk.vtkImageData()
    image.SetDimensions(*padded.shape)
    image.SetSpacing(*spacing)
    image.SetOrigin(*((offset - _PAD) * spacing))
    scalars = numpy_to_vtk(padded.ravel(order="F"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    scalars.SetName("mask")
    image.GetPointData().SetScalars(scalars)
    return image


def _isosurface(image: vtk.vtkImageData) -> vtk.vtkPolyData:
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(image)
    fe.SetValue(0, ISO_VALUE)
    fe.ComputeNormalsOff()
    fe.Update()
    return fe.GetOutput()


def _clean(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.Update()
    return clean.GetOutput()


def _filter_components(poly: vtk.vtkPolyData, fraction: float) -> tuple[vtk.vtkPolyData, int, int]:
    """Drop connected components with fewer cells than ``fraction`` x the largest one."""
    probe = vtk.vtkPolyDataConnectivityFilter()
    probe.SetInputData(poly)
    probe.SetExtractionModeToAllRegions()
    probe.Update()
    sizes = vtk_to_numpy(probe.GetRegionSizes()).astype(np.int64)
    if sizes.size == 0:
        raise SurfaceError("isosurface is empty")
    keep = np.flatnonzero(sizes >= fraction * sizes.max())
    if len(keep) == len(sizes):
        return poly, len(sizes), len(sizes)

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(poly)
    if len(keep) == 1:  # the BP behaviour, bit for bit
        conn.SetExtractionModeToLargestRegion()
    else:
        conn.SetExtractionModeToSpecifiedRegions()
        for region in keep:
            conn.AddSpecifiedRegion(int(region))
    conn.Update()
    out = _clean(conn.GetOutput())  # drop points of removed regions
    return out, len(sizes), len(keep)


def _smooth(poly: vtk.vtkPolyData, iterations: int, passband: float) -> vtk.vtkPolyData:
    source: vtk.vtkPolyData = poly
    if iterations > 0:
        ws = vtk.vtkWindowedSincPolyDataFilter()
        ws.SetInputData(poly)
        ws.SetNumberOfIterations(iterations)
        ws.SetPassBand(passband)
        ws.BoundarySmoothingOff()
        ws.FeatureEdgeSmoothingOff()
        ws.NonManifoldSmoothingOn()
        ws.NormalizeCoordinatesOn()
        ws.Update()
        source = ws.GetOutput()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(source)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.PointMergingOn()
    clean.Update()
    return clean.GetOutput()


def _decimate(poly: vtk.vtkPolyData, target_reduction: float) -> vtk.vtkPolyData:
    if target_reduction <= 0.0:
        return poly
    dec = vtk.vtkQuadricDecimation()
    dec.SetInputData(poly)
    dec.SetTargetReduction(target_reduction)
    dec.Update()
    return dec.GetOutput()


def _to_trimesh(poly: vtk.vtkPolyData) -> TriMesh:
    if poly.GetNumberOfPolys() == 0:
        return TriMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    polys = poly.GetPolys()
    offsets = vtk_to_numpy(polys.GetOffsetsArray())
    if not np.all(np.diff(offsets) == 3):
        raise SurfaceError("expected a triangle mesh")
    faces = vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int64).reshape(-1, 3)
    used = np.unique(faces)
    if len(used) != len(points):  # drop points not referenced by any triangle
        remap = np.full(len(points), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        points, faces = points[used], remap[faces]
    return TriMesh(points, faces)


def to_polydata(mesh: TriMesh) -> vtk.vtkPolyData:
    """TriMesh -> vtkPolyData (for VTK-based analysis)."""
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(mesh.vertices, deep=True))
    cells = vtk.vtkCellArray()
    offsets = numpy_to_vtk(np.arange(0, 3 * mesh.n_faces + 1, 3, dtype=np.int64), deep=True)
    conn = numpy_to_vtk(mesh.faces.ravel(), deep=True)
    cells.SetData(offsets, conn)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    return poly
