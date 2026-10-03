# SPDX-License-Identifier: Apache-2.0
"""Step ``segment``: input volume -> raw multi-label segmentations of the chosen tools.

Input: the input volume (``00_input/input.nii[.gz]``) and its modality.

Output (``01_segment/``)::

    <tool>_<task>.nii.gz   one multi-label image per tool run, in the tool's label scheme,
                           on the input grid (e.g. totalsegmentator_total.nii.gz,
                           musclemap_wholebody.nii.gz)
    segmentation.json      index with the tool's label name -> ID maps (mskpipe.io.segmentation)

Which tools run and which labels they must deliver follows ``segmentation.*`` through the
unified label scheme (``config/unified_labels.yaml``). Both legs are segmented regardless
of ``skeleton.side``, so a run for the other leg reuses this step from the cache.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from mskpipe.config import PipelineConfig
from mskpipe.core.manifest import package_info
from mskpipe.core.registry import PluginError, Registry, default_registry
from mskpipe.core.step import Step, StepContext, StepError

if TYPE_CHECKING:
    from mskpipe.plugins.base import SegmentationOutput, SegmentationRequest

# Distributions whose version changes a segmenter's outputs (part of the step fingerprint).
TOOL_PACKAGES: dict[str, tuple[str, ...]] = {
    "totalsegmentator": ("TotalSegmentator", "nnunetv2", "torch"),
    "musclemap": ("scripts", "monai", "torch"),
}
_AFFINE_ATOL = 1e-3  # mm


class SegmentStep(Step):
    name: ClassVar[str] = "segment"
    version: ClassVar[str] = "1"
    config_sections: ClassVar[tuple[str, ...]] = ("segmentation",)

    def __init__(self, registry: Registry | None = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> Registry:
        return self._registry or default_registry()

    def fingerprint_extra(self, config: PipelineConfig) -> dict[str, Any]:
        from mskpipe.labelmap.scheme import load_scheme, scheme_sha256

        tools: dict[str, Any] = {}
        for tool in load_scheme().requests(config, "ct"):
            try:
                plugin = self.registry.get("segmenter", tool)
                ident: dict[str, Any] = dict(self.registry.identity("segmenter", tool))
                ident.update(plugin.cache_identity(config))
            except PluginError as exc:  # reported by run(); keep the fingerprint computable
                ident = {"error": str(exc)}
            ident["packages"] = {d: _package_version(d) for d in TOOL_PACKAGES.get(tool, ())}
            tools[tool] = ident
        return {"scheme": scheme_sha256(), "tools": tools}

    def run(self, ctx: StepContext) -> None:
        from mskpipe.io.segmentation import SEGMENTATION_FILE, SegmentationIndex, ToolOutput
        from mskpipe.labelmap.scheme import SchemeError, load_scheme

        image = ctx.ws.input_image
        modality = ctx.input.modality.value
        try:
            requests = load_scheme().requests(ctx.config, modality)
        except SchemeError as exc:
            raise StepError(str(exc)) from exc
        if not requests:
            raise StepError("No segmentation source configured")

        grid = _grid(image)
        ctx.logger.info(
            "[segment] input %s, %s mm, %s, %s",
            "x".join(map(str, grid["shape"])),
            "x".join(f"{s:.3g}" for s in grid["spacing_mm"]),
            grid["orientation"],
            modality,
        )

        plugins = {}
        for tool in requests:
            try:
                plugins[tool] = self.registry.create("segmenter", tool)
            except PluginError as exc:
                raise StepError(str(exc)) from exc
            if modality not in plugins[tool].modalities:
                raise StepError(f"Segmenter '{tool}' does not support modality '{modality}'")

        entries: list[ToolOutput] = []
        tool_metrics: dict[str, Any] = {}
        warnings: list[str] = []
        for tool, request in requests.items():
            ctx.check_cancel()
            plugin = plugins[tool]
            start = time.perf_counter()
            try:
                outputs = plugin.segment(ctx, image, ctx.out_dir, plugin.Params(), request)
            except StepError:
                raise
            except (OSError, ValueError, RuntimeError, NotImplementedError) as exc:
                raise StepError(f"{tool}: {exc}") from exc
            elapsed = time.perf_counter() - start
            if not outputs:
                raise StepError(f"Segmenter '{tool}' returned no output")

            stats = [_check_output(tool, out, ctx.out_dir, grid) for out in outputs]
            report = _coverage(request, outputs, stats)
            for msg in report["warnings"]:
                warnings.append(f"{tool}: {msg}")
                ctx.logger.warning("[segment] %s: %s", tool, msg)
            for out in outputs:
                rel = out.image.resolve().relative_to(ctx.out_dir.resolve()).as_posix()
                entries.append(ToolOutput(tool=tool, task=out.task, file=rel, labels=out.labels))
                ctx.record.add_output(out.image)
            tool_metrics[tool] = {
                "tasks": [o.task for o in outputs],
                "time_s": round(elapsed, 3),
                "device": ctx.device.device_for(plugin.supports_gpu),
                "n_requested": len(request.labels),
                "not_provided": report["not_provided"],
                "empty": report["empty"],
                "voxels": report["voxels"],
            }
            ctx.logger.info(
                "[segment] %s: %s in %.1f s", tool, ", ".join(o.task for o in outputs), elapsed
            )

        index_path = ctx.out_dir / SEGMENTATION_FILE
        SegmentationIndex(outputs=tuple(entries)).save(index_path)
        ctx.record.add_output(index_path)
        ctx.record.metrics.update(
            {
                "modality": modality,
                "input": grid,
                "tools": tool_metrics,
                "segment_time_s": round(sum(m["time_s"] for m in tool_metrics.values()), 3),
                "warnings": warnings,
            }
        )


# ---------------------------------------------------------------------- helpers


def _package_version(dist: str) -> str | None:
    info = package_info(dist)
    if info.commit:
        return info.commit
    # local version labels (+cpu, +cu128) differ between the CPU and GPU environments;
    # the device is deliberately not part of the cache key
    return info.version.split("+", 1)[0] if info.version else None


def _grid(image: Path) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np

    try:
        img = nib.load(image)
    except (OSError, nib.filebasedimages.ImageFileError) as exc:
        raise StepError(f"Cannot read the input image {image.name}: {exc}") from exc
    if len(img.shape) != 3 and not (len(img.shape) == 4 and img.shape[3] == 1):
        raise StepError(f"Expected a 3D input volume, got shape {img.shape}")
    affine = np.asarray(img.affine, dtype=float)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    return {
        "shape": [int(n) for n in img.shape[:3]],
        "spacing_mm": [round(float(s), 4) for s in spacing],
        "orientation": "".join(nib.aff2axcodes(affine)),
        "dtype": str(img.get_data_dtype()),
        "affine": [[round(float(v), 6) for v in row] for row in affine],
    }


def _check_output(
    tool: str, out: SegmentationOutput, out_dir: Path, grid: dict[str, Any]
) -> dict[int, int]:
    """Check that an output lies on the input grid; return voxel counts per label value."""
    import numpy as np

    from mskpipe.io.nifti import NiftiError, load_labelmap

    path = Path(out.image)
    if not path.is_file():
        raise StepError(f"{tool}: output not found: {path}")
    try:
        path.resolve().relative_to(out_dir.resolve())
    except ValueError:
        raise StepError(f"{tool}: output {path} is outside {out_dir}") from None
    try:
        vol = load_labelmap(path)
    except NiftiError as exc:
        raise StepError(f"{tool}: {exc}") from exc
    if list(vol.data.shape) != grid["shape"] or not np.allclose(
        vol.affine, np.asarray(grid["affine"]), atol=_AFFINE_ATOL
    ):
        raise StepError(
            f"{tool}: {path.name} is not on the input grid "
            f"(shape {list(vol.data.shape)} vs {grid['shape']})"
        )
    counts = np.bincount(vol.data.ravel())
    return {int(v): int(counts[v]) for v in np.flatnonzero(counts) if v > 0}


def _coverage(
    request: SegmentationRequest,
    outputs: list[SegmentationOutput],
    stats: list[dict[int, int]],
) -> dict[str, Any]:
    """Requested labels the tool does not know or did not find, and voxels per label."""
    voxels: dict[str, int] = {}
    known: set[str] = set()
    for out, counts in zip(outputs, stats, strict=True):
        for name, value in out.labels.items():
            if name in request.labels:
                known.add(name)
                voxels[name] = voxels.get(name, 0) + counts.get(value, 0)
    not_provided = sorted(request.labels - known)
    empty = sorted(n for n in known if voxels[n] == 0)
    warnings = []
    if not_provided:
        warnings.append(f"labels not provided by the tool: {', '.join(not_provided)}")
    if empty:
        warnings.append(f"no voxels for {', '.join(empty)}")
    return {
        "not_provided": not_provided,
        "empty": empty,
        "voxels": dict(sorted(voxels.items())),
        "warnings": warnings,
    }
