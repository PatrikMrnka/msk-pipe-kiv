# SPDX-License-Identifier: Apache-2.0
"""Guess whether a NIfTI volume is CT or MRI.

Evidence, strongest first:

1. ``Modality`` in the dcm2niix/BIDS JSON sidecar next to the image (``CT`` / ``MR``);
2. ``TE=``/``TR=`` in the header description (dcm2niix writes them for MR series);
3. intensities: CT is stored in Hounsfield units, so air around and inside the body is
   about -1000 and the padding outside the field of view -1024 or lower; MRI intensities are
   arbitrary and non-negative.

The intensity rule fails for CT stored without the HU offset (unsigned 0..4095 with the
rescale intercept lost), which then looks like MRI: the user can always override.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

__all__ = ["ModalityGuess", "detect_modality", "guess_from_intensities", "sidecar_path"]

AIR_HU = -500.0  # voxels at or below this are air (or padding) in HU
MIN_AIR_FRACTION = 0.01  # CT shows at least 1 % air voxels (around the body, lungs, bowel)
CT_MINIMUM = -900.0  # the minimum of a CT is air or padding
MRI_MIN_TOLERANCE = -50.0  # MRI may dip slightly below 0 after bias correction
MAX_SAMPLES = 4_000_000  # voxels read for the statistics (subsampled grid)


@dataclass(frozen=True)
class ModalityGuess:
    modality: Literal["ct", "mri"] | None  # None = undecided
    source: Literal["sidecar", "header", "intensities"] | None
    reason: str

    def __str__(self) -> str:
        if self.modality is None:
            return f"unknown ({self.reason})"
        return f"{self.modality.upper()} ({self.source}: {self.reason})"


def sidecar_path(image: Path) -> Path:
    name = image.name
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return image.with_name(f"{name}.json")


def detect_modality(path: str | Path) -> ModalityGuess:
    """Best guess of the modality of a NIfTI volume (reads the voxel data if needed)."""
    import nibabel as nib

    path = Path(path)
    guess = _from_sidecar(sidecar_path(path))
    if guess is not None:
        return guess
    img = nib.load(str(path))
    guess = _from_header(img.header)
    if guess is not None:
        return guess
    data = img.dataobj
    shape = img.shape[:3]
    step = max(1, round((np.prod(shape) / MAX_SAMPLES) ** (1 / 3)))
    sample = np.asanyarray(data[::step, ::step, ::step] if len(img.shape) == 3 else data)
    return guess_from_intensities(np.asarray(sample, dtype=np.float32))


def guess_from_intensities(values: np.ndarray) -> ModalityGuess:
    """CT (Hounsfield units) vs MRI from the voxel values (scaling already applied)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return ModalityGuess(None, None, "no finite voxel values")
    low = float(v.min())
    air = float(np.mean(v <= AIR_HU))
    if low <= CT_MINIMUM and air >= MIN_AIR_FRACTION:
        return ModalityGuess(
            "ct",
            "intensities",
            f"{air:.0%} of voxels at or below {AIR_HU:.0f} HU, minimum {low:.0f}",
        )
    if low >= MRI_MIN_TOLERANCE:
        high = float(np.percentile(v, 99.5))
        return ModalityGuess(
            "mri", "intensities", f"no negative values (minimum {low:.0f}, 99.5th pct {high:.0f})"
        )
    return ModalityGuess(
        None, None, f"negative values (minimum {low:.0f}) but only {air:.1%} below {AIR_HU:.0f}"
    )


def _from_sidecar(path: Path) -> ModalityGuess | None:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = str(meta.get("Modality", "")).upper() if isinstance(meta, dict) else ""
    if value == "CT":
        return ModalityGuess("ct", "sidecar", f"Modality CT in {path.name}")
    if value in ("MR", "MRI"):
        return ModalityGuess("mri", "sidecar", f"Modality {value} in {path.name}")
    return None


_MR_DESCRIP = re.compile(rb"\b(TE|TR)=\d")


def _from_header(header) -> ModalityGuess | None:
    try:
        descrip = bytes(header["descrip"].tobytes()).rstrip(b"\0")
    except (KeyError, AttributeError, TypeError):
        return None
    if _MR_DESCRIP.search(descrip):
        text = descrip.decode("ascii", "replace")[:40]
        return ModalityGuess("mri", "header", f"MR timing in the description '{text}'")
    return None
