# SPDX-License-Identifier: Apache-2.0
"""Label map cleaning vs. the BP pipeline (``clean_masks.py``), for tests and the paper.

BP cleaned every binary mask separately: largest component -> closing -> opening ->
``VotingBinaryIterativeHoleFilling`` (radius 1, 3 iterations; despite its name it mostly
grows concave boundaries, ~1 % of the volume, and leaves enclosed cavities). msk-pipe:
closing -> 3D hole filling -> opening -> components, growth only into background, one
label per voxel. Expected result: Dice > 0.99 and our masks inside BP's (no extra voxels).

Reference layout (BP mask names, binary NIfTI on one grid)::

    <dir>/raw/<name>.nii.gz       BP bone_segmentations/raw, muscle_segmentations/extracted
    <dir>/cleaned/<name>.nii.gz   BP .../cleaned of the same run

Names: ``hip_left``, ``hip_right``, ``femur_left``, ``femur_right`` (TotalSegmentator
``total``), ``tibia``, ``fibula`` (``appendicular_bones``, both legs), muscles as in
MuscleMap (``gluteus_maximus_r``, ...).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

TS_TOTAL = ("hip_left", "hip_right", "femur_left", "femur_right")
TS_APPENDICULAR = ("tibia", "fibula")
# unified structure(s) <- BP mask(s)
_BP_GROUPS = {
    "pelvis_no_sacrum": (("pelvis_no_sacrum",), ("hip_left", "hip_right")),
    "femur_r": (("femur_r",), ("femur_right",)),
    "femur_l": (("femur_l",), ("femur_left",)),
    "tibia": (("tibia_r", "tibia_l"), ("tibia",)),
    "fibula": (("fibula_r", "fibula_l"), ("fibula",)),
}


def _name(path: Path) -> str:
    return path.name.removesuffix(".nii.gz").removesuffix(".nii")


def _tool_label(scheme: Any, tool: str, name: str) -> str:
    """MuscleMap label name of a BP muscle mask (BP used unified names, e.g. biceps_femoris_r)."""
    if tool != "musclemap":
        return name
    return scheme.sources["musclemap"].labels.get(name, (name,))[0]


def find_masks(directory: Path) -> dict[str, Path]:
    return {_name(p): p for p in sorted(Path(directory).glob("*.nii*"))}


def compare_with_bp(raw_dir: Path, cleaned_dir: Path) -> dict[str, Any]:
    """Run the msk-pipe label map on BP raw masks and compare with BP cleaned masks."""
    from mskpipe.config import load_config
    from mskpipe.io.nifti import Volume, load_labelmap
    from mskpipe.io.segmentation import ToolOutput
    from mskpipe.labelmap.build import build_labelmap
    from mskpipe.labelmap.scheme import load_scheme

    raw = find_masks(raw_dir)
    cleaned = find_masks(cleaned_dir)
    if not raw:
        raise FileNotFoundError(f"No masks in {raw_dir}")

    scheme = load_scheme()
    muscles = {n for n, s in scheme.structures.items() if s.kind == "muscle"}
    groups = {
        ("totalsegmentator", "total"): [n for n in raw if n in TS_TOTAL],
        ("totalsegmentator", "appendicular_bones"): [n for n in raw if n in TS_APPENDICULAR],
        ("musclemap", "default"): [n for n in raw if n in muscles],
    }
    unknown = set(raw) - {n for names in groups.values() for n in names}
    if unknown:
        raise ValueError(f"Unknown BP mask names: {sorted(unknown)}")

    outputs: list[tuple[ToolOutput, Volume]] = []
    affine = None
    for (tool, task), names in groups.items():
        if not names:
            continue
        data, labels = None, {}
        for value, name in enumerate(names, start=1):
            vol = load_labelmap(raw[name])
            if data is None:
                data, affine = np.zeros(vol.data.shape, np.uint8), vol.affine
            data[vol.data > 0] = value
            labels[_tool_label(scheme, tool, name)] = value
        out = ToolOutput(tool=tool, task=task, file=f"{tool}_{task}.nii.gz", labels=labels)
        outputs.append((out, Volume(data, affine)))

    config = load_config(overrides=["segmentation.tibia_fibula=ts_appendicular"])
    plan = scheme.plan(config, sides=("r", "l"))  # BP masks cover both legs
    provided = {label for out, _ in outputs for label in out.labels}
    plan = {n: p for n, p in plan.items() if all(label in provided for label in p[1])}
    start = time.perf_counter()
    result = build_labelmap(outputs, scheme, plan, config.labelmap)
    elapsed = time.perf_counter() - start

    bp_groups = dict(_BP_GROUPS)
    bp_groups.update({m: ((m,), (m,)) for m in muscles})
    rows: dict[str, Any] = {}
    bp_union = None
    for key, (ours_names, bp_names) in bp_groups.items():
        if not all(n in cleaned and n in raw for n in bp_names):
            continue
        values = [scheme.structures[n].value for n in ours_names]
        ours = np.isin(result.data, values)
        bp = np.zeros_like(ours)
        raw_mask = np.zeros_like(ours)
        for n in bp_names:
            bp |= load_labelmap(cleaned[n]).data > 0
            raw_mask |= load_labelmap(raw[n]).data > 0
        bp_count = bp.astype(np.uint8)
        bp_union = bp_count if bp_union is None else bp_union + bp_count
        rows[key] = {
            "bp_masks": list(bp_names),
            "dice": float(2 * (ours & bp).sum() / (ours.sum() + bp.sum())),
            "dice_raw_vs_bp": float(2 * (raw_mask & bp).sum() / (raw_mask.sum() + bp.sum())),
            "voxels_raw": int(raw_mask.sum()),
            "voxels_ours": int(ours.sum()),
            "voxels_bp": int(bp.sum()),
            "only_ours": int((ours & ~bp).sum()),
            "only_bp": int((bp & ~ours).sum()),
        }
    return {
        "structures": rows,
        "bp_overlap_voxels": int((bp_union > 1).sum()) if bp_union is not None else 0,
        "labelmap_time_s": round(elapsed, 3),
        "midline": result.info["midline"],
        "labelmap": result.structures,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"{'structure':<22}{'Dice':>9}{'raw/BP':>9}{'ours':>10}{'BP':>10}"
        f"{'only ours':>11}{'only BP':>9}"
    ]
    for name, r in report["structures"].items():
        lines.append(
            f"{name:<22}{r['dice']:>9.5f}{r['dice_raw_vs_bp']:>9.5f}{r['voxels_ours']:>10}"
            f"{r['voxels_bp']:>10}{r['only_ours']:>11}{r['only_bp']:>9}"
        )
    lines.append(f"BP voxels claimed by two masks: {report['bp_overlap_voxels']}")
    lines.append(f"label map built in {report['labelmap_time_s']:.1f} s")
    return "\n".join(lines)
