# SPDX-License-Identifier: Apache-2.0
"""NIfTI volumes via nibabel.

Arrays are indexed ``[i, j, k]`` and ``affine`` maps voxel indices to world coordinates
(NIfTI RAS+, millimetres). Label maps keep their integer dtype: never ``get_fdata()``,
which would need 8 bytes per voxel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


class NiftiError(ValueError):
    """A NIfTI file cannot be used."""


@dataclass(frozen=True, eq=False)
class Volume:
    data: np.ndarray
    affine: np.ndarray

    @property
    def spacing(self) -> np.ndarray:
        """Voxel size in mm, taken from the affine (column norms)."""
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def rotation(self) -> np.ndarray:
        """Direction part of the affine, spacing removed (may contain a reflection)."""
        return self.affine[:3, :3] / self.spacing


def load_labelmap(path: str | Path) -> Volume:
    """Load an integer label map (a binary mask is a label map with one label)."""
    path = Path(path)
    try:
        img = nib.load(path)
    except FileNotFoundError:
        raise NiftiError(f"NIfTI file not found: {path}") from None
    except (OSError, nib.filebasedimages.ImageFileError) as exc:
        raise NiftiError(f"Cannot read NIfTI {path}: {exc}") from exc

    data = np.asanyarray(img.dataobj)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise NiftiError(f"{path.name}: expected a 3D volume, got shape {data.shape}")
    if not np.issubdtype(data.dtype, np.integer):
        rounded = np.rint(data)
        if not np.array_equal(rounded, data):
            raise NiftiError(f"{path.name}: label map contains non-integer values")
        data = rounded
    if data.min(initial=0) < 0 or data.max(initial=0) > np.iinfo(np.uint16).max:
        raise NiftiError(f"{path.name}: label values must be in 0..65535")
    affine = np.asarray(img.affine, dtype=np.float64)
    if not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise NiftiError(f"{path.name}: invalid affine")
    if data.dtype not in (np.uint8, np.uint16):
        data = data.astype(np.uint16)
    return Volume(np.asarray(data), affine)
