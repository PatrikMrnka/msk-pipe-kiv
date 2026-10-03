# SPDX-License-Identifier: Apache-2.0
"""Validation of the raw segmentations of the ``segment`` step (tests and the paper).

Two independent checks of ``01_segment/``:

* :func:`sanity_check` needs no reference (usable for any dataset, e.g. TLEM2): every
  structure the configured sources should deliver has voxels, ``_r`` structures lie on the
  patient's right of their ``_l`` counterparts (+x in NIfTI RAS+ world coordinates) and,
  per leg, pelvis > femur > tibia in the superior direction. Swapped label IDs, swapped
  sides or a flipped/rotated output grid break these.
* :func:`compare_reference` compares tool labels with reference binary masks on the same
  grid (e.g. the BP raw masks ``bone_segmentations/raw``, ``muscle_segmentations/extracted``):
  Dice, volume ratio, centroid shift, average symmetric surface distance and HD95 in mm.

Reference mask names may be tool label names (``hip_left``, ``femur_right``, ``ilium_r``),
BP names (``tibia``/``fibula`` = both legs) or unified structure names
(``gluteus_maximus_r``, ``biceps_femoris_r``); see :func:`resolve_reference`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mskpipe.config import PipelineConfig
    from mskpipe.labelmap.scheme import Scheme

# BP / TotalSegmentator mask names -> alternative tool label names (other sources)
REFERENCE_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "hip_left": (("ilium_l",),),
    "hip_right": (("ilium_r",),),
    "femur_left": (("femur_l",),),
    "femur_right": (("femur_r",),),
    "tibia": (("tibia_r", "tibia_l"),),
    "fibula": (("fibula_r", "fibula_l"),),
}
_AFFINE_ATOL = 1e-3  # mm
_PAD = 2  # voxels around the compared region (surface distances)


class SegmentationValidationError(ValueError):
    """The run or the reference cannot be compared."""


# ------------------------------------------------------------------------- label access


@dataclass
class _Output:
    tool: str
    task: str
    path: Path
    labels: dict[str, int]
    _data: np.ndarray | None = None
    _affine: np.ndarray | None = None
    _boxes: list[Any] | None = field(default=None, repr=False)

    def load(self) -> None:
        if self._data is None:
            from scipy import ndimage

            from mskpipe.io.nifti import load_labelmap

            vol = load_labelmap(self.path)
            self._data, self._affine = vol.data, np.asarray(vol.affine, float)
            self._boxes = ndimage.find_objects(self._data)

    @property
    def data(self) -> np.ndarray:
        self.load()
        assert self._data is not None
        return self._data

    @property
    def affine(self) -> np.ndarray:
        self.load()
        assert self._affine is not None
        return self._affine

    def bbox(self, names: Iterable[str]) -> tuple[slice, ...] | None:
        """Bounding box of the given labels (None = no voxels)."""
        self.load()
        assert self._boxes is not None
        boxes = []
        for name in names:
            value = self.labels[name]
            if value <= len(self._boxes) and self._boxes[value - 1] is not None:
                boxes.append(self._boxes[value - 1])
        return _union(boxes)

    def mask(self, names: Iterable[str], region: tuple[slice, ...]) -> np.ndarray:
        values = [self.labels[n] for n in names]
        return np.isin(self.data[region], values)


class RunSegmentation:
    """The tool outputs listed in ``01_segment/segmentation.json``."""

    def __init__(self, segment_dir: str | Path) -> None:
        from mskpipe.io.segmentation import SEGMENTATION_FILE, SegmentationIndex

        self.dir = Path(segment_dir)
        index = SegmentationIndex.load(self.dir / SEGMENTATION_FILE)
        self.outputs = [
            _Output(o.tool, o.task, self.dir / o.file, dict(o.labels)) for o in index.outputs
        ]
        if not self.outputs:
            raise SegmentationValidationError(f"No tool outputs listed in {self.dir}")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.outputs[0].data.shape)

    @property
    def affine(self) -> np.ndarray:
        return self.outputs[0].affine

    def find(self, names: tuple[str, ...], tool: str | None = None) -> _Output | None:
        """First output (of ``tool``) that knows all ``names`` and has voxels for them."""
        for out in self.outputs:
            if tool is not None and out.tool != tool:
                continue
            if all(n in out.labels for n in names) and out.bbox(names) is not None:
                return out
        return None


def _union(boxes: list[tuple[slice, ...]]) -> tuple[slice, ...] | None:
    if not boxes:
        return None
    return tuple(
        slice(min(b[d].start for b in boxes), max(b[d].stop for b in boxes))
        for d in range(len(boxes[0]))
    )


def _pad(box: tuple[slice, ...], shape: tuple[int, ...], pad: int = _PAD) -> tuple[slice, ...]:
    return tuple(
        slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(box, shape, strict=True)
    )


def _centroid(mask: np.ndarray, region: tuple[slice, ...], affine: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask).mean(axis=0) + [s.start for s in region]
    return (affine @ np.append(idx, 1.0))[:3]


# --------------------------------------------------------------------------- sanity check


def structure_sources(
    config: PipelineConfig, scheme: Scheme
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Unified structure -> (tool, tool label names) of the configured sources."""
    chosen = scheme.selection(config)
    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for name, struct in scheme.structures.items():
        src = scheme.sources[chosen[struct.group]]
        labels = src.labels.get(name)
        if labels:
            out[name] = (src.tool, tuple(labels))
    return out


def sanity_check(seg: RunSegmentation, config: PipelineConfig, scheme: Scheme) -> dict[str, Any]:
    """Reference-free plausibility of the raw segmentations (both legs)."""
    sources = structure_sources(config, scheme)
    rows: dict[str, Any] = {}
    centroids: dict[str, np.ndarray] = {}
    problems: list[str] = []
    for name, (tool, labels) in sources.items():
        out = seg.find(labels, tool)
        if out is None:
            rows[name] = {"tool": tool, "labels": list(labels), "status": "missing"}
            problems.append(f"{name}: no voxels for {', '.join(labels)} ({tool})")
            continue
        box = out.bbox(labels)
        assert box is not None
        mask = out.mask(labels, box)
        voxel_ml = abs(np.linalg.det(out.affine[:3, :3])) / 1000.0
        centroids[name] = _centroid(mask, box, out.affine)
        rows[name] = {
            "tool": tool,
            "task": out.task,
            "labels": list(labels),
            "status": "ok",
            "voxels": int(mask.sum()),
            "volume_ml": round(float(mask.sum()) * voxel_ml, 2),
            "centroid_ras_mm": [round(float(v), 1) for v in centroids[name]],
        }

    # left/right: only for structures whose sides come from different tool labels
    side_rows = {}
    for name in sources:
        if not name.endswith("_r"):
            continue
        left = name[:-2] + "_l"
        if left not in sources or sources[name] == sources[left]:
            continue  # one label for both legs (split later by the labelmap step)
        if name in centroids and left in centroids:
            dx = float(centroids[name][0] - centroids[left][0])
            side_rows[name[:-2]] = round(dx, 1)
            if dx <= 0:
                problems.append(
                    f"{name} is not right of {left} (x_r - x_l = {dx:.1f} mm): swapped sides "
                    "or label IDs?"
                )

    # superior ordering per leg
    order_rows = {}
    for side in ("r", "l"):
        chain = [
            n for n in ("pelvis_no_sacrum", f"femur_{side}", f"tibia_{side}") if n in centroids
        ]
        zs = [float(centroids[n][2]) for n in chain]
        order_rows[side] = {n: round(z, 1) for n, z in zip(chain, zs, strict=True)}
        if any(a <= b for a, b in pairwise(zs)):
            problems.append(
                f"leg {side}: expected pelvis > femur > tibia (superior), got "
                + ", ".join(f"{n} z={z:.0f}" for n, z in zip(chain, zs, strict=True))
            )
    return {
        "structures": rows,
        "side_dx_mm": side_rows,
        "superior_z_mm": order_rows,
        "problems": problems,
    }


# ------------------------------------------------------------------- reference comparison


def resolve_reference(
    name: str, seg: RunSegmentation, scheme: Scheme, config: PipelineConfig | None = None
) -> tuple[_Output, tuple[str, ...]] | None:
    """Tool output and labels corresponding to a reference mask ``name``."""
    candidates: list[tuple[str | None, tuple[str, ...]]] = [(None, (name,))]
    candidates += [(None, alias) for alias in REFERENCE_ALIASES.get(name, ())]
    if name in scheme.structures:
        if config is not None:
            chosen = structure_sources(config, scheme).get(name)
            if chosen:
                candidates.append(chosen)
        candidates += [
            (src.tool, tuple(src.labels[name]))
            for src in scheme.sources.values()
            if name in src.labels
        ]
    for tool, labels in candidates:
        out = seg.find(labels, tool)
        if out is not None:
            return out, labels
    return None


def _surface(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return mask & ~ndimage.binary_erosion(mask)


def surface_distances(a: np.ndarray, b: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Distances (mm) of the surface voxels of ``a`` to the surface of ``b`` and back."""
    from scipy import ndimage

    sa, sb = _surface(a), _surface(b)
    if not sa.any() or not sb.any():
        return np.array([np.inf])
    da = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    return np.concatenate([da, db])


def compare_masks(
    ours: np.ndarray, ref: np.ndarray, region: tuple[slice, ...], affine: np.ndarray
) -> dict[str, Any]:
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    n_ours, n_ref = int(ours.sum()), int(ref.sum())
    inter = int((ours & ref).sum())
    dist = surface_distances(ours, ref, spacing)
    shift = np.linalg.norm(_centroid(ours, region, affine) - _centroid(ref, region, affine))
    return {
        "dice": round(2 * inter / (n_ours + n_ref), 5),
        "voxels_ours": n_ours,
        "voxels_ref": n_ref,
        "volume_ratio": round(n_ours / n_ref, 4),
        "only_ours": n_ours - inter,
        "only_ref": n_ref - inter,
        "centroid_shift_mm": round(float(shift), 3),
        "assd_mm": round(float(dist.mean()), 3),
        "hd95_mm": round(float(np.percentile(dist, 95)), 3),
    }


def compare_reference(
    seg: RunSegmentation,
    reference_dir: str | Path,
    scheme: Scheme,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Compare tool labels with every reference mask ``<reference_dir>/<name>.nii[.gz]``."""
    from mskpipe.io.nifti import load_labelmap

    paths = sorted(Path(reference_dir).glob("*.nii*"))
    if not paths:
        raise SegmentationValidationError(f"No reference masks in {reference_dir}")
    rows: dict[str, Any] = {}
    unmatched: list[str] = []
    affine_ok = True
    for path in paths:
        name = path.name.removesuffix(".gz").removesuffix(".nii")
        found = resolve_reference(name, seg, scheme, config)
        if found is None:
            unmatched.append(name)
            continue
        out, labels = found
        ref_vol = load_labelmap(path)
        if tuple(ref_vol.data.shape[:3]) != tuple(out.data.shape):
            raise SegmentationValidationError(
                f"{path.name}: shape {ref_vol.data.shape} differs from the run "
                f"{out.data.shape}; the reference must be on the input grid"
            )
        if not np.allclose(ref_vol.affine, out.affine, atol=_AFFINE_ATOL):
            affine_ok = False
        ref_full = ref_vol.data > 0
        ref_box = _bbox(ref_full)
        box = _union([b for b in (out.bbox(labels), ref_box) if b is not None])
        if box is None or ref_box is None:
            unmatched.append(name)
            continue
        region = _pad(box, out.data.shape)
        row = compare_masks(out.mask(labels, region), ref_full[region], region, out.affine)
        rows[name] = {"tool": out.tool, "task": out.task, "labels": list(labels), **row}
    return {"structures": rows, "unmatched": unmatched, "affine_matches": affine_ok}


def _bbox(mask: np.ndarray) -> tuple[slice, ...] | None:
    from scipy import ndimage

    boxes = ndimage.find_objects(mask.astype(np.uint8))
    return boxes[0] if boxes else None


# --------------------------------------------------------------------------------- report


def format_report(sanity: dict[str, Any], reference: dict[str, Any] | None = None) -> str:
    lines = ["Structures (configured sources):"]
    for name, r in sanity["structures"].items():
        if r["status"] != "ok":
            lines.append(f"  {name:26s} MISSING ({r['tool']}: {', '.join(r['labels'])})")
            continue
        c = ", ".join(f"{v:7.1f}" for v in r["centroid_ras_mm"])
        lines.append(
            f"  {name:26s} {r['tool']:16s} {r['volume_ml']:9.1f} ml  centroid RAS ({c}) mm"
        )
    if sanity["side_dx_mm"]:
        lines.append("Right - left centroid x (mm, must be > 0):")
        lines += [f"  {n:26s} {dx:8.1f}" for n, dx in sanity["side_dx_mm"].items()]
    if reference is not None:
        lines.append("Reference masks:")
        lines.append(
            f"  {'mask':26s} {'tool':16s} {'Dice':>7s} {'V/Vref':>7s} {'dC mm':>7s} "
            f"{'ASSD':>6s} {'HD95':>6s}"
        )
        for name, r in reference["structures"].items():
            lines.append(
                f"  {name:26s} {r['tool']:16s} {r['dice']:7.4f} {r['volume_ratio']:7.3f} "
                f"{r['centroid_shift_mm']:7.2f} {r['assd_mm']:6.2f} {r['hd95_mm']:6.2f}"
            )
        if reference["unmatched"]:
            lines.append(f"  not in the run: {', '.join(reference['unmatched'])}")
        if not reference["affine_matches"]:
            lines.append("  WARNING: reference affine differs from the run (compared voxel-wise)")
    if sanity["problems"]:
        lines.append("PROBLEMS:")
        lines += [f"  - {p}" for p in sanity["problems"]]
    else:
        lines.append("Sanity checks: OK")
    return "\n".join(lines)
