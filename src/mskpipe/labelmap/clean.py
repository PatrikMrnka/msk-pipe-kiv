# SPDX-License-Identifier: Apache-2.0
"""Cleaning of one binary mask (part of the ``labelmap`` step).

Order: closing -> 3D hole filling -> opening -> removal of small connected components.

* Radii are in millimetres; the structuring element is the set of voxel offsets whose
  physical distance is <= radius, so anisotropic voxels (MRI) get an ellipsoid in voxels.
  An ITK/SimpleITK ball of radius ``r`` voxels (used by the BP pipeline) is the set of
  offsets with distance <= ``r + 0.5`` voxels, i.e. BP radius 1 at 1 mm = 1.5 mm here.
* Closing and hole filling only add voxels that are not ``blocked`` (voxels of other
  structures), so a structure never grows into its neighbours.
* Components use 6-connectivity (as BP); components smaller than ``min_component_fraction``
  of the largest one are removed (1 = keep only the largest).

Masks are processed in a crop padded by the element radius, so results equal processing
the whole volume (SimpleITK ``SafeBorder``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Box = tuple[slice, slice, slice]


@dataclass(frozen=True)
class CleanParams:
    closing_mm: float = 1.5
    opening_mm: float = 1.5
    fill_holes: bool = True
    min_component_fraction: float = 1.0


def element_radius(radius_mm: float, spacing: np.ndarray) -> np.ndarray:
    """Half-size of the structuring element in voxels per axis."""
    return np.floor(radius_mm / np.asarray(spacing, dtype=float) + 1e-9).astype(int)


def structuring_element(radius_mm: float, spacing: np.ndarray) -> np.ndarray:
    """Ellipsoidal (in voxels) structuring element of a ball of ``radius_mm``."""
    spacing = np.asarray(spacing, dtype=float)
    half = element_radius(radius_mm, spacing)
    grid = np.indices(2 * half + 1) - half[:, None, None, None]
    dist2 = ((grid * spacing[:, None, None, None]) ** 2).sum(axis=0)
    return dist2 <= radius_mm**2 * (1 + 1e-9)


def padded_box(shape: tuple[int, ...], box: Box, pad: np.ndarray) -> tuple[Box, list[tuple]]:
    """Box grown by ``pad`` (clipped to the volume) and the zero padding outside the volume.

    ``np.pad(volume[grown], padding)`` is the box grown by ``pad`` on every side.
    """
    grown, padding = [], []
    for s, p, n in zip(box, pad, shape, strict=True):
        lo, hi = s.start - int(p), s.stop + int(p)
        grown.append(slice(max(lo, 0), min(hi, n)))
        padding.append((max(-lo, 0), max(hi - n, 0)))
    return tuple(grown), padding  # type: ignore[return-value]


def unpad(array: np.ndarray, padding: list[tuple]) -> np.ndarray:
    return array[tuple(slice(a, array.shape[i] - b) for i, (a, b) in enumerate(padding))]


def clean_mask(
    mask: np.ndarray,
    spacing: np.ndarray,
    params: CleanParams,
    blocked: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clean a binary mask that has at least ``element_radius + 1`` voxels of zero margin.

    Returns the cleaned mask and voxel counts of each operation.
    """
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    free = np.ones_like(mask) if blocked is None else ~np.asarray(blocked, dtype=bool)
    stats: dict[str, Any] = {}

    out = mask
    if params.closing_mm > 0:
        se = structuring_element(params.closing_mm, spacing)
        added = ndimage.binary_closing(out, se) & ~out & free
        stats["closing_added"] = int(added.sum())
        out = out | added
    if params.fill_holes:
        added = ndimage.binary_fill_holes(out) & ~out & free
        stats["fill_added"] = int(added.sum())
        out = out | added
    if params.opening_mm > 0:
        se = structuring_element(params.opening_mm, spacing)
        opened = ndimage.binary_opening(out, se)
        stats["opening_removed"] = int((out & ~opened).sum())
        out = opened

    labels, n = ndimage.label(out)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if n:
        keep = np.flatnonzero(sizes >= params.min_component_fraction * sizes.max())
        keep = keep[keep > 0]  # label 0 = background
        removed = n - len(keep)
        if removed:
            kept = np.isin(labels, keep)
            stats["components_removed_voxels"] = int((out & ~kept).sum())
            out = kept
    else:
        removed = 0
    stats["components"] = int(n)
    stats["components_removed"] = int(removed)
    return out, stats
