# SPDX-License-Identifier: Apache-2.0
"""Sanity checks of a valid configuration: values that work but are probably not meant.

The schema (:mod:`mskpipe.config.schema`) rejects impossible values. This module reports
the rest as :class:`Issue` objects, without changing anything:

* **ranges** - a numeric field outside its recommended range (``json_schema_extra``
  ``recommended`` in the schema or a plugin's ``Params``),
* **settings** - combinations known to give poor results (fast TotalSegmentator model,
  repair switched off, an opening larger than the closing, ...),
* **environment** - threads vs CPU cores, segmentation workers vs RAM, run folder writable,
  free disk space, a requested GPU that cannot be used,
* **input** - voxel spacing, modality of the volume vs the one given, closing/opening
  radii smaller than a voxel.

:func:`check` runs them all; ``mskpipe config check``, the pre-run checks of ``mskpipe
run`` / ``mskpipe batch`` and the GUI call it. Checks that need intermediate results (side
of the leg in the image, triangles after decimation, coordinate range of the model) are
done by the steps themselves.
"""

from __future__ import annotations

import math
import os
import shutil
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from mskpipe.config.schema import PipelineConfig, Segmenter

if TYPE_CHECKING:
    from mskpipe.core.registry import Registry

__all__ = [
    "ANATOMICAL_RANGES_DEG",
    "MIN_FREE_DISK",
    "Issue",
    "Recommended",
    "check",
    "check_config",
    "check_environment",
    "check_input",
    "recommended_ranges",
]

Severity = Literal["error", "warning", "info"]

GIB = 2**30
MIN_FREE_DISK = 5 * GIB  # one LHDL run with intermediate outputs needs about 2-3 GB
WORKER_MEMORY = 2 * GIB  # one TotalSegmentator saving worker (loads PyTorch)
MAX_FRAMES = 1000  # Muscle Wrapping time grows with the frames of the motion
IN_PLANE_MM = (0.3, 2.0)
MAX_SLICE_MM = 3.0

# Anatomical range of the swept coordinates [deg], OpenSim sign conventions (knee flexion
# negative); key = coordinate name without the side suffix
ANATOMICAL_RANGES_DEG: dict[str, tuple[float, float]] = {
    "hip_flexion": (-30.0, 120.0),
    "hip_adduction": (-50.0, 30.0),
    "hip_rotation": (-45.0, 45.0),
    "knee_angle": (-120.0, 10.0),
}


@dataclass(frozen=True)
class Issue:
    """A problem found before a run. ``error`` would make the run fail, ``warning`` is a
    setting that works but is probably not meant, ``info`` is a note."""

    severity: Severity
    where: str
    message: str
    field: str | None = None  # dotted config path the issue is about (GUI marks the field)

    def __str__(self) -> str:
        return f"{self.severity}: {self.where}: {self.message}"


@dataclass(frozen=True)
class Recommended:
    path: str
    low: float | None
    high: float | None
    why: str = ""

    def contains(self, value: float) -> bool:
        return (self.low is None or value >= self.low) and (self.high is None or value <= self.high)

    def text(self) -> str:
        lo = "-inf" if self.low is None else f"{self.low:g}"
        hi = "inf" if self.high is None else f"{self.high:g}"
        return f"{lo} - {hi}"


# ---------------------------------------------------------------------------- ranges


def recommended_ranges(model: type[BaseModel], prefix: str = "") -> dict[str, Recommended]:
    """Recommended ranges of all numeric fields of ``model`` (nested models included),
    keyed by dotted path."""
    out: dict[str, Recommended] = {}
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        extra = field.json_schema_extra
        if isinstance(extra, Mapping) and "recommended" in extra:
            low, high = extra["recommended"]  # type: ignore[misc]
            out[path] = Recommended(path, low, high, str(extra.get("recommended_why", "")))
        for sub in _models(field.annotation):
            out.update(recommended_ranges(sub, f"{path}."))
    return out


def _models(annotation: Any) -> list[type[BaseModel]]:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return [m for a in typing.get_args(annotation) for m in _models(a)]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return []


def _value(data: Any, path: str) -> Any:
    for part in path.split("."):
        data = data.get(part) if isinstance(data, Mapping) else getattr(data, part, None)
        if data is None:
            return None
    return data


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def range_issues(data: Any, ranges: Mapping[str, Recommended], prefix: str = "") -> list[Issue]:
    issues = []
    for path, rec in ranges.items():
        value = _value(data, path)
        if _is_number(value) and not rec.contains(value):
            why = f" ({rec.why})" if rec.why else ""
            full = f"{prefix}{path}"
            issues.append(
                Issue(
                    "warning",
                    full.split(".", 1)[0],
                    f"{full} = {value:g} is outside the recommended range {rec.text()}{why}",
                    full,
                )
            )
    return issues


# ---------------------------------------------------------------------------- config


def check_config(config: PipelineConfig, registry: Registry | None = None) -> list[Issue]:
    """Recommended ranges and known poor combinations of settings (no input, no I/O)."""
    issues = range_issues(config, recommended_ranges(PipelineConfig))
    issues += _plugin_ranges(config, registry)
    issues += _settings(config)
    issues += _motion(config)
    return issues


def _plugin_ranges(config: PipelineConfig, registry: Registry | None) -> list[Issue]:
    from mskpipe.core.registry import PluginError, default_registry

    reg = registry or default_registry()
    try:
        plugin = reg.get("attachments", config.attachments.method)
    except PluginError:
        return []  # reported by the pre-run checks
    params = {**plugin.Params().model_dump(mode="json"), **config.attachments.params}
    issues = range_issues(params, recommended_ranges(plugin.Params), "attachments.params.")
    n_points, target = params.get("n_points"), params.get("target_points")
    if _is_number(n_points) and _is_number(target) and target < n_points:
        issues.append(
            Issue(
                "warning",
                "attachments",
                f"attachments.params.target_points ({target}) < n_points ({n_points}): the "
                "subject surface is sampled more sparsely than the atlas (biased ICP)",
                "attachments.params.target_points",
            )
        )
    return issues


def _settings(config: PipelineConfig) -> list[Issue]:
    seg, issues = config.segmentation, []
    ts_used = Segmenter.TOTALSEGMENTATOR in (seg.bones, seg.muscles)
    if ts_used and seg.totalsegmentator.fast:
        issues.append(
            Issue(
                "warning",
                "segmentation",
                "segmentation.totalsegmentator.fast uses the 3 mm model: coarse masks, not "
                "suitable for surface meshes",
                "segmentation.totalsegmentator.fast",
            )
        )
    if ts_used and not seg.totalsegmentator.roi_subset:
        issues.append(
            Issue(
                "info",
                "segmentation",
                "segmentation.totalsegmentator.roi_subset=false predicts all structures: "
                "slower, masks differ by about 1 % (E12)",
                "segmentation.totalsegmentator.roi_subset",
            )
        )
    if seg.muscles == Segmenter.TOTALSEGMENTATOR:
        issues.append(
            Issue(
                "warning",
                "segmentation",
                "segmentation.muscles=totalsegmentator gives only the gluteal muscles "
                "(3 muscles for Muscle Wrapping)",
                "segmentation.muscles",
            )
        )
    if seg.muscles == Segmenter.MUSCLEMAP or seg.bones == Segmenter.MUSCLEMAP:
        from mskpipe.plugins.segmenters.musclemap import MODEL_VERSION

        version = seg.musclemap.model_version
        if version is not None and version != MODEL_VERSION:
            issues.append(
                Issue(
                    "warning",
                    "segmentation",
                    f"segmentation.musclemap.model_version {version} is untested; msk-pipe "
                    f"checked the label names of {MODEL_VERSION}",
                    "segmentation.musclemap.model_version",
                )
            )
    for kind in ("bones", "muscles"):
        params = getattr(config.labelmap, kind)
        if params.opening_radius_mm > params.closing_radius_mm + 2:
            issues.append(
                Issue(
                    "warning",
                    "labelmap",
                    f"labelmap.{kind}.opening_radius_mm ({params.opening_radius_mm:g}) exceeds "
                    f"closing_radius_mm ({params.closing_radius_mm:g}) by more than 2 mm: the "
                    "opening removes more than the closing filled",
                    f"labelmap.{kind}.opening_radius_mm",
                )
            )
    if not config.mesh.repair.enabled:
        issues.append(
            Issue(
                "warning",
                "mesh",
                "mesh.repair.enabled=false: open or non-manifold surfaces reach pystaple and "
                "Muscle Wrapping",
                "mesh.repair.enabled",
            )
        )
    if config.mesh.on_invalid == "warn":
        issues.append(
            Issue(
                "warning",
                "mesh",
                "mesh.on_invalid=warn continues with invalid bone surfaces; pystaple then fails",
                "mesh.on_invalid",
            )
        )
    if config.export.bone_weights == "PositionOnFibreQuadraticSpline":
        issues.append(
            Issue(
                "warning",
                "export",
                "export.bone_weights=PositionOnFibreQuadraticSpline is linear in Muscle "
                "Wrapping 2.x (hamstrings without excursion, E15); use InverseDistance",
                "export.bone_weights",
            )
        )
    if config.export.fibre_export:
        issues.append(
            Issue(
                "info",
                "export",
                "export.fibre_export=true: Muscle Wrapping writes fibres and meshes of every "
                "frame (large output)",
                "export.fibre_export",
            )
        )
    return issues


def _motion(config: PipelineConfig) -> list[Issue]:
    motion, side = config.export.motion, config.skeleton.side
    coordinate = motion.coordinate or f"hip_flexion_{side}"
    base, _, suffix = coordinate.rpartition("_")
    issues = []
    if suffix in ("l", "r") and suffix != side:
        issues.append(
            Issue(
                "warning",
                "export",
                f"export.motion.coordinate '{coordinate}' belongs to the other leg "
                f"(skeleton.side={side})",
                "export.motion.coordinate",
            )
        )
    limits = ANATOMICAL_RANGES_DEG.get(base if suffix in ("l", "r") else coordinate)
    if limits and not limits[0] <= motion.end_deg <= limits[1]:
        issues.append(
            Issue(
                "warning",
                "export",
                f"export.motion.end_deg = {motion.end_deg:g} is outside the anatomical range "
                f"of {coordinate} ({limits[0]:g} - {limits[1]:g} deg)",
                "export.motion.end_deg",
            )
        )
    frames = math.ceil(abs(motion.end_deg) / motion.step_deg)
    if frames > MAX_FRAMES:
        issues.append(
            Issue(
                "warning",
                "export",
                f"about {frames} motion frames (end_deg / step_deg); Muscle Wrapping time "
                f"grows with the frames, keep them below {MAX_FRAMES}",
                "export.motion.step_deg",
            )
        )
    return issues


# ---------------------------------------------------------------------------- environment


def check_environment(config: PipelineConfig) -> list[Issue]:
    """Threads, segmentation workers, run folder and free disk space; a requested GPU."""
    import psutil

    issues = []
    cores = os.cpu_count() or 1
    threads = config.runtime.threads
    if threads is not None and threads > cores:
        issues.append(
            Issue(
                "warning",
                "runtime",
                f"runtime.threads = {threads} exceeds the {cores} CPU cores (only overhead)",
                "runtime.threads",
            )
        )
    saving = config.segmentation.totalsegmentator.nr_thr_saving
    if isinstance(saving, int) and Segmenter.TOTALSEGMENTATOR in (
        config.segmentation.bones,
        config.segmentation.muscles,
    ):
        ram = psutil.virtual_memory().total
        if saving * WORKER_MEMORY > ram:
            issues.append(
                Issue(
                    "warning",
                    "segmentation",
                    f"segmentation.totalsegmentator.nr_thr_saving = {saving} workers need about "
                    f"{saving * WORKER_MEMORY / GIB:.0f} GiB, the computer has "
                    f"{ram / GIB:.0f} GiB RAM (use 'auto')",
                    "segmentation.totalsegmentator.nr_thr_saving",
                )
            )
    issues += _runs_dir(Path(config.runtime.runs_dir))
    if config.runtime.device == "gpu":
        from mskpipe.core.device import gpu_unavailable_reason

        reason = gpu_unavailable_reason()
        if reason:
            issues.append(
                Issue(
                    "error", "runtime", f"GPU requested but not usable: {reason}", "runtime.device"
                )
            )
    return issues


def _runs_dir(folder: Path) -> list[Issue]:
    folder = folder.expanduser()
    existing = folder
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.is_dir() or not os.access(existing, os.W_OK):
        return [
            Issue(
                "error",
                "runtime",
                f"runtime.runs_dir '{folder}' cannot be created or written ({existing} is "
                "not a writable folder)",
                "runtime.runs_dir",
            )
        ]
    free = shutil.disk_usage(existing).free
    if free < MIN_FREE_DISK:
        return [
            Issue(
                "warning",
                "runtime",
                f"only {free / GIB:.1f} GiB free for runs in '{folder}'; one run with "
                f"intermediate outputs needs about 2-3 GB (keep at least "
                f"{MIN_FREE_DISK / GIB:.0f} GiB)",
                "runtime.runs_dir",
            )
        ]
    return []


# ---------------------------------------------------------------------------- input


def check_input(
    config: PipelineConfig, image: str | Path, modality: str | None = None
) -> list[Issue]:
    """Voxel spacing of the input volume, morphology radii below one voxel and (if
    ``modality`` is given) the modality detected from the intensities. The header checks
    of :func:`mskpipe.api.check_image` (3D, orientation) come first and are not repeated."""
    import nibabel as nib

    try:
        zooms = sorted(float(z) for z in nib.load(str(image)).header.get_zooms()[:3])
    except Exception:  # unreadable header: reported by api.check_image
        return []
    if len(zooms) < 3:
        return []
    issues = []
    in_plane, slice_mm = zooms[:2], zooms[2]
    lo, hi = IN_PLANE_MM
    if any(not lo <= z <= hi for z in in_plane):
        issues.append(
            Issue(
                "warning",
                "input",
                f"in-plane spacing {in_plane[0]:.3g} x {in_plane[1]:.3g} mm is outside "
                f"{lo:g} - {hi:g} mm (segmentation models are trained on such images)",
            )
        )
    if slice_mm > MAX_SLICE_MM:
        issues.append(
            Issue(
                "warning",
                "input",
                f"slices are {slice_mm:.3g} mm apart (> {MAX_SLICE_MM:g} mm): thin bones and "
                "small muscles may have holes or gaps (E17)",
            )
        )
    smallest = zooms[0]
    for kind in ("bones", "muscles"):
        params = getattr(config.labelmap, kind)
        for name in ("closing_radius_mm", "opening_radius_mm"):
            radius = getattr(params, name)
            if 0 < radius < smallest:
                issues.append(
                    Issue(
                        "warning",
                        "labelmap",
                        f"labelmap.{kind}.{name} = {radius:g} mm is smaller than the voxel "
                        f"({smallest:.3g} mm): no effect",
                        f"labelmap.{kind}.{name}",
                    )
                )
    if modality is not None:
        from mskpipe.io.modality import detect_modality

        try:
            guess = detect_modality(image)
        except Exception:  # unreadable data: reported elsewhere
            guess = None
        if guess is not None and guess.modality is not None and guess.modality != modality:
            issues.append(
                Issue(
                    "warning",
                    "input",
                    f"modality given as {modality.upper()}, but the volume looks like {guess}",
                )
            )
    return issues


# ---------------------------------------------------------------------------- all


def check(
    config: PipelineConfig,
    image: str | Path | None = None,
    modality: str | None = None,
    *,
    registry: Registry | None = None,
    environment: bool = True,
    detect_modality: bool = True,
) -> list[Issue]:
    """All checks; ``image`` adds the input checks, ``environment=False`` skips the checks
    of this computer (threads, RAM, disk, GPU), ``detect_modality=False`` skips reading the
    voxels (the modality was detected already)."""
    issues = check_config(config, registry)
    if environment:
        issues += check_environment(config)
    if image is not None:
        issues += check_input(config, image, modality if detect_modality else None)
    return issues
