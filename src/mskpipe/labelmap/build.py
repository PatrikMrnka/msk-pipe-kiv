# SPDX-License-Identifier: Apache-2.0
"""Assembly of the unified label map from raw tool segmentations.

1. Each configured source label becomes a *part* of a unified structure (two hip bones ->
   two parts of ``pelvis_no_sacrum``). A label covering both legs (TotalSegmentator
   ``appendicular_bones``) is split by side: each connected component goes to ``_r`` if its
   centroid lies right of the midline (pelvis centroid; image centre without a pelvis).
   Only planned structures are built (by default the ``skeleton.side`` leg and the pelvis).
2. Every part is cleaned separately (:mod:`mskpipe.labelmap.clean`); growth is blocked by
   the raw voxels of all other parts.
3. Parts are written to one label map: bones first, then muscles. A voxel claimed by two
   structures of the same kind becomes background (e.g. a gap between two muscles or the
   joint space); bones win over muscles.
4. Structures with fewer than ``min_voxels`` voxels are removed (status ``too_small``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mskpipe.config.schema import LabelmapConfig
from mskpipe.io.labels import Label, LabelTable
from mskpipe.io.nifti import Volume
from mskpipe.io.segmentation import ToolOutput
from mskpipe.labelmap.clean import (
    Box,
    CleanParams,
    clean_mask,
    element_radius,
    padded_box,
    unpad,
)
from mskpipe.labelmap.scheme import Scheme

KIND_PARAMS = {"bone": "bones", "muscle": "muscles"}
_AFFINE_ATOL = 1e-3  # mm


class LabelmapError(ValueError):
    """The raw segmentations cannot be combined."""


@dataclass
class _Part:
    structure: str
    kind: str
    source_label: str
    box: Box
    mask: np.ndarray


@dataclass
class LabelmapResult:
    data: np.ndarray
    affine: np.ndarray
    table: LabelTable
    structures: list[dict[str, Any]]
    info: dict[str, Any] = field(default_factory=dict)


def clean_params(config: LabelmapConfig, kind: str) -> CleanParams:
    p = getattr(config, KIND_PARAMS[kind])
    return CleanParams(
        closing_mm=p.closing_radius_mm,
        opening_mm=p.opening_radius_mm,
        fill_holes=p.fill_holes,
        min_component_fraction=p.min_component_fraction,
    )


def build_labelmap(
    outputs: Sequence[tuple[ToolOutput, Volume]],
    scheme: Scheme,
    plan: dict[str, tuple[str, tuple[str, ...]]],
    config: LabelmapConfig,
    log: Any = None,
) -> LabelmapResult:
    """Combine raw tool outputs into the unified label map (see module docstring)."""
    from scipy import ndimage

    if not outputs:
        raise LabelmapError("No segmentation outputs")
    shape, affine = _common_grid(outputs)
    spacing = outputs[0][1].spacing
    voxel_ml = float(np.prod(spacing)) / 1000.0

    # ---------------------------------------------------------------- raw parts
    boxes = {id(vol): ndimage.find_objects(vol.data) for _, vol in outputs}
    parts: list[_Part] = []
    shared: dict[tuple[str, str], list[str]] = {}  # (source, label) -> structures
    for name, (source, labels) in plan.items():
        for label in labels:
            shared.setdefault((source, label), []).append(name)

    midline = _midline(outputs, scheme, plan, boxes, affine, shape)
    absent: dict[str, list[str]] = {}
    for (source, label), names in shared.items():
        found = _find_label(outputs, scheme, source, label)
        box = None if found is None else _box(boxes[id(found[0])], found[1])
        if found is None or box is None:
            absent.setdefault(source, []).append(label)
            continue
        vol, value = found
        mask = vol.data[box] == value
        kind = scheme.structures[names[0]].kind
        if label in scheme.sources[source].split_names():
            parts.extend(_split_sides(names, kind, label, box, mask, affine, midline["x_mm"]))
        else:
            parts.extend(_Part(name, kind, label, box, mask) for name in names)

    for source, labels in absent.items():
        _warn(log, "%s: no voxels for %s", source, ", ".join(labels))

    # raw coverage: how many parts claim each voxel (growth blocker, raw overlaps)
    coverage = np.zeros(shape, dtype=np.uint8)
    for part in parts:
        coverage[part.box] += part.mask

    # ---------------------------------------------------------------- clean parts
    stats: dict[str, dict[str, Any]] = {}
    cleaned: list[tuple[_Part, Box, np.ndarray]] = []
    for part in parts:
        start = time.perf_counter()
        params = clean_params(config, part.kind)
        pad = np.maximum(
            element_radius(params.closing_mm, spacing), element_radius(params.opening_mm, spacing)
        )
        pad = pad + 1
        grown, padding = padded_box(shape, part.box, pad)
        local = (
            slice(part.box[0].start - grown[0].start, part.box[0].stop - grown[0].start),
            slice(part.box[1].start - grown[1].start, part.box[1].stop - grown[1].start),
            slice(part.box[2].start - grown[2].start, part.box[2].stop - grown[2].start),
        )
        own = np.zeros([s.stop - s.start for s in grown], dtype=bool)
        own[local] = part.mask
        blocked = (coverage[grown] - own) > 0
        mask, op_stats = clean_mask(
            np.pad(own, padding), spacing, params, blocked=np.pad(blocked, padding)
        )
        mask = unpad(mask, padding)
        cleaned.append((part, grown, mask))

        s = stats.setdefault(part.structure, {"parts": [], "raw_voxels": 0, "raw_overlap": 0})
        s["parts"].append(part.source_label)
        s["raw_voxels"] += int(own.sum())
        s["raw_overlap"] += int((own & blocked).sum())
        for key, val in op_stats.items():
            s[key] = s.get(key, 0) + val
        s["time_s"] = s.get("time_s", 0.0) + time.perf_counter() - start

    # ---------------------------------------------------------------- assemble
    data = np.zeros(shape, dtype=np.uint8)
    blocked_all = np.zeros(shape, dtype=bool)
    is_bone = np.zeros(256, dtype=bool)
    for struct in scheme.structures.values():
        is_bone[struct.value] = struct.kind == "bone"
    names_by_value = scheme.by_value()
    lost_to_bone: dict[str, int] = {}
    contested: dict[str, int] = {}
    for kind in ("bone", "muscle"):
        for part, box, mask in cleaned:
            if part.kind != kind:
                continue
            value = scheme.structures[part.structure].value
            region, block = data[box], blocked_all[box]  # views
            current = region.copy()
            other = mask & (current != 0) & (current != value)
            if kind == "muscle":
                bone = other & is_bone[current]
                lost_to_bone[part.structure] = lost_to_bone.get(part.structure, 0) + int(bone.sum())
                other &= ~bone
            region[mask & (current == 0) & ~block] = value
            region[other] = 0
            block[other] = True
            losers, counts = np.unique(current[other], return_counts=True)
            for loser, count in zip(losers, counts, strict=True):
                loser_name = names_by_value[int(loser)]
                contested[loser_name] = contested.get(loser_name, 0) + int(count)
            contested[part.structure] = contested.get(part.structure, 0) + int(other.sum())

    # ---------------------------------------------------------------- finish
    counts = np.bincount(data.ravel(), minlength=256)
    entries: list[dict[str, Any]] = []
    labels: list[Label] = []
    for name in plan:
        struct = scheme.structures[name]
        labels.append(Label(name=name, value=struct.value, kind=struct.kind))
        entry: dict[str, Any] = {"name": name, "kind": struct.kind, "value": struct.value}
        entry["source"], entry["source_labels"] = plan[name][0], list(plan[name][1])
        if name not in stats:
            entries.append({**entry, "status": "missing"})
            continue
        s = stats[name]
        n = int(counts[struct.value])
        min_voxels = getattr(config, KIND_PARAMS[struct.kind]).min_voxels
        entry["status"] = "ok" if n >= max(min_voxels, 1) else "too_small"
        if entry["status"] == "too_small":
            data[data == struct.value] = 0
            _warn(log, "%s: %d voxels < min_voxels %d, removed", name, n, min_voxels)
        entry.update(
            parts=s["parts"],
            raw_voxels=s["raw_voxels"],
            voxels=n,
            raw_volume_ml=round(s["raw_voxels"] * voxel_ml, 3),
            volume_ml=round(n * voxel_ml, 3),
            volume_change=round(n / s["raw_voxels"] - 1, 5) if s["raw_voxels"] else None,
            raw_overlap=s["raw_overlap"],
            closing_added=s.get("closing_added", 0),
            fill_added=s.get("fill_added", 0),
            opening_removed=s.get("opening_removed", 0),
            components=s["components"],
            components_removed=s["components_removed"],
            components_removed_voxels=s.get("components_removed_voxels", 0),
            contested=contested.get(name, 0),
            lost_to_bone=lost_to_bone.get(name, 0),
            time_s=round(s["time_s"], 3),
        )
        entries.append(entry)

    info = {
        "shape": list(shape),
        "spacing_mm": [round(float(v), 6) for v in spacing],
        "midline": midline,
    }
    return LabelmapResult(data, affine, LabelTable(labels=tuple(labels)), entries, info)


# ------------------------------------------------------------------------------ helpers


def _warn(log: Any, msg: str, *args: Any) -> None:
    if log is not None:
        log.warning("[labelmap] " + msg, *args)


def _common_grid(outputs: Sequence[tuple[ToolOutput, Volume]]) -> tuple[tuple, np.ndarray]:
    first_out, first = outputs[0]
    for out, vol in outputs[1:]:
        if vol.data.shape != first.data.shape or not np.allclose(
            vol.affine, first.affine, atol=_AFFINE_ATOL
        ):
            raise LabelmapError(
                f"'{out.file}' and '{first_out.file}' are not on the same voxel grid "
                f"(shape {vol.data.shape} vs {first.data.shape})"
            )
    return first.data.shape, first.affine


def _find_label(
    outputs: Sequence[tuple[ToolOutput, Volume]], scheme: Scheme, source: str, label: str
) -> tuple[Volume, int] | None:
    src = scheme.sources[source]
    for out, vol in outputs:
        if src.accepts(out.tool, out.task) and label in out.labels:
            return vol, out.labels[label]
    return None


def _box(boxes: list, value: int) -> Box | None:
    return boxes[value - 1] if value - 1 < len(boxes) else None


def _world_x(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """RAS x (mm) of voxel index coordinates (N x 3)."""
    return np.asarray(ijk, dtype=float) @ affine[0, :3] + affine[0, 3]


def _midline(outputs, scheme, plan, boxes, affine, shape) -> dict[str, Any]:
    """Left/right boundary: x of the pelvis centroid, else of the image centre."""
    from scipy import ndimage

    total, weight = 0.0, 0
    for name, (source, labels) in plan.items():
        if name != "pelvis_no_sacrum":
            continue
        for label in labels:
            found = _find_label(outputs, scheme, source, label)
            if found is None:
                continue
            vol, value = found
            box = _box(boxes[id(vol)], value)
            if box is None:
                continue
            mask = vol.data[box] == value
            n = int(mask.sum())
            com = np.array(ndimage.center_of_mass(mask)) + [s.start for s in box]
            total += float(_world_x(affine, com[None])[0]) * n
            weight += n
    if weight:
        return {"x_mm": round(total / weight, 3), "from": "pelvis_no_sacrum"}
    centre = (np.array(shape) - 1) / 2.0
    return {"x_mm": round(float(_world_x(affine, centre[None])[0]), 3), "from": "image_centre"}


def _split_sides(
    names: list[str],
    kind: str,
    label: str,
    box: Box,
    mask: np.ndarray,
    affine: np.ndarray,
    midline_x: float,
) -> list[_Part]:
    """Split a label covering both legs into ``_r``/``_l`` parts by component centroids.

    ``names`` are the planned structures of the label; components of an unplanned side are
    dropped.
    """
    from scipy import ndimage

    comps, n = ndimage.label(mask)
    centres = np.array(ndimage.center_of_mass(mask, comps, range(1, n + 1))).reshape(-1, 3)
    centres += [s.start for s in box]
    right = _world_x(affine, centres) > midline_x
    by_side = {name[-1]: name for name in names}
    parts = []
    for side, ids in (("r", np.flatnonzero(right) + 1), ("l", np.flatnonzero(~right) + 1)):
        if len(ids) and side in by_side:
            side_mask = np.isin(comps, ids)
            sub = ndimage.find_objects(side_mask.astype(np.uint8))[0]
            sub_box = tuple(
                slice(b.start + s.start, b.start + s.stop) for b, s in zip(box, sub, strict=True)
            )
            parts.append(_Part(by_side[side], kind, label, sub_box, side_mask[sub]))
    return parts
