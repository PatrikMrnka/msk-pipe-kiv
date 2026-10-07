# SPDX-License-Identifier: Apache-2.0
"""TotalSegmentator (tasks ``total``/``total_mr``, optionally ``appendicular_bones[_mr]``).

One multi-label image per task (``--ml``), written by the tool on the input grid. The
label name -> ID map is read from the NIfTI extension TotalSegmentator writes into that
image (IDs differ between ``total`` and ``total_mr``, so they are never hard-coded).
The tool runs as ``python -m totalsegmentator.bin.TotalSegmentator`` in a subprocess.
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from mskpipe.plugins.base import (
    PluginParams,
    SegmentationOutput,
    SegmentationRequest,
    SegmenterPlugin,
)

if TYPE_CHECKING:
    from mskpipe.config import PipelineConfig
    from mskpipe.config.schema import TotalSegmentatorConfig
    from mskpipe.core.step import StepContext

MODULE = "totalsegmentator.bin.TotalSegmentator"
ROI_TASKS = frozenset({"total", "total_mr"})  # tasks supporting --roi_subset and --fast
LICENSED_TASKS = frozenset({"appendicular_bones", "appendicular_bones_mr"})
LICENCE_HINT = (
    "TotalSegmentator task '{task}' needs a licence key (free for non-commercial use: "
    "https://backend.totalsegmentator.com/license-academic/). Set it once with "
    "`pixi run -e cpu totalseg_set_license -l <key>`, or take tibia/fibula from MuscleMap: "
    "--set segmentation.tibia_fibula=musclemap."
)


LICENCE_LENGTH = 18  # TotalSegmentator's own offline check (config.has_valid_license_offline)


class TotalSegmentatorError(RuntimeError):
    """TotalSegmentator output cannot be used."""


def licence_number() -> str | None:
    """Licence key stored by ``totalseg_set_license`` (``$TOTALSEG_HOME_DIR`` or
    ``~/.totalsegmentator``, ``config.json``); read directly, because
    ``totalsegmentator.config`` imports torch."""
    home = os.environ.get("TOTALSEG_HOME_DIR")
    folder = Path(home) if home else Path.home() / ".totalsegmentator"
    try:
        data = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    key = data.get("license_number") if isinstance(data, dict) else None
    return key if isinstance(key, str) and key else None


def choose_task(tasks: Iterable[str], modality: str) -> str:
    """The task for ``modality`` among those allowed by the label scheme (``*_mr`` = MRI)."""
    tasks = tuple(tasks)
    if not tasks:
        return "total_mr" if modality == "mri" else "total"
    suited = [t for t in tasks if t.endswith("_mr") == (modality == "mri")]
    if not suited:
        raise TotalSegmentatorError(
            f"No TotalSegmentator task for modality '{modality}' among {list(tasks)}"
        )
    return suited[0]


def plan_runs(request: SegmentationRequest) -> dict[str, frozenset[str]]:
    """Task -> labels needed from it (jobs sharing a task run once)."""
    runs: dict[str, frozenset[str]] = {}
    for job in request.jobs:
        task = choose_task(job.tasks, request.modality)
        runs[task] = runs.get(task, frozenset()) | job.labels
    return runs


def known_labels(task: str) -> frozenset[str] | None:
    """Label names TotalSegmentator defines for ``task`` (its class map, a plain dict
    module), or ``None`` when TotalSegmentator is not installed / the task is unknown."""
    try:
        from totalsegmentator.map_to_binary import class_map
    except ImportError:
        return None
    names = class_map.get(task)
    return frozenset(names.values()) if names else None


def split_labels(task: str, labels: Iterable[str]) -> tuple[frozenset[str], frozenset[str]]:
    """(labels the task has, labels it lacks): ``--roi_subset`` fails on unknown names,
    e.g. ``vertebrae_S1`` exists in ``total`` but not in ``total_mr``."""
    labels = frozenset(labels)
    known = known_labels(task)
    if known is None:
        return labels, frozenset()
    return labels & known, labels - known


def build_command(
    image: Path,
    out_file: Path,
    task: str,
    labels: Iterable[str],
    cfg: TotalSegmentatorConfig,
    *,
    device: str,
) -> list[str]:
    """Command line of one TotalSegmentator run (multi-label output file)."""
    argv = [sys.executable, "-m", MODULE, "-i", str(image), "-o", str(out_file), "--ml"]
    argv += ["-ta", task, "-d", "gpu" if device == "cuda" else "cpu"]
    if task in ROI_TASKS:
        if cfg.roi_subset:
            argv += ["-rs", *sorted(labels)]
        if cfg.fast:
            argv.append("-f")
    if cfg.higher_order_resampling:
        argv.append("--higher_order_resampling")
    argv += list(cfg.extra_args)
    return argv


def parse_label_xml(content: bytes | str) -> dict[str, int]:
    """Label name -> ID from TotalSegmentator's CaretExtension XML."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    try:
        root = ET.fromstring(content.strip())
    except ET.ParseError as exc:
        raise TotalSegmentatorError(f"Invalid label table in the NIfTI extension: {exc}") from exc
    labels: dict[str, int] = {}
    for node in root.iter("Label"):
        name = (node.text or "").strip()
        key = node.get("Key")
        if not name or key is None:
            continue
        value = int(key)
        if value > 0:
            labels[name] = value
    if not labels:
        raise TotalSegmentatorError("The NIfTI extension contains no labels")
    return labels


def read_label_map(path: Path) -> dict[str, int]:
    """Label map of a TotalSegmentator ``--ml`` output (from its NIfTI extension)."""
    import nibabel as nib

    img = nib.load(path)
    for ext in img.header.extensions:
        content = ext.get_content()
        raw = content if isinstance(content, bytes) else str(content).encode("utf-8")
        if b"CaretExtension" in raw:
            return parse_label_xml(raw)
    raise TotalSegmentatorError(f"{path.name}: no label table in the NIfTI extension")


class TotalSegmentator(SegmenterPlugin):
    name = "totalsegmentator"
    description = "TotalSegmentator: pelvis, femur, gluteal muscles; tibia/fibula with licence."
    requires_modules = ("totalsegmentator",)
    supports_gpu = True

    @classmethod
    def preflight(cls, config: PipelineConfig, modality: str) -> list[str]:
        from mskpipe.labelmap.scheme import load_scheme

        request = load_scheme().requests(config, modality).get(cls.name)
        if request is None:
            return []
        try:
            licensed = sorted(set(plan_runs(request)) & LICENSED_TASKS)
        except TotalSegmentatorError as exc:
            return [str(exc)]
        if not licensed:
            return []
        key = licence_number()
        if key is None:
            return [LICENCE_HINT.format(task=licensed[0])]
        if len(key) != LICENCE_LENGTH:
            return [
                f"Invalid TotalSegmentator licence key ({len(key)} characters). "
                + LICENCE_HINT.format(task=licensed[0])
            ]
        return []

    def segment(
        self,
        ctx: StepContext,
        image: Path,
        out_dir: Path,
        params: PluginParams,
        request: SegmentationRequest,
    ) -> list[SegmentationOutput]:
        from mskpipe.core.step import StepCancelled, StepError

        cfg = ctx.config.segmentation.totalsegmentator
        device = ctx.device.device_for(self.supports_gpu)
        try:
            runs = plan_runs(request)
        except TotalSegmentatorError as exc:
            raise StepError(str(exc)) from exc

        outputs: list[SegmentationOutput] = []
        for task, wanted in runs.items():
            ctx.check_cancel()
            labels, lacking = split_labels(task, wanted)
            if lacking:
                ctx.logger.info(
                    "[segment] TotalSegmentator %s has no %s", task, ", ".join(sorted(lacking))
                )
            out_file = out_dir / f"totalsegmentator_{task}.nii.gz"
            argv = build_command(
                image,
                out_file,
                task,
                labels,
                cfg,
                device=device,
            )
            ctx.logger.info("[segment] TotalSegmentator %s on %s", task, device)
            try:
                ctx.run_command(argv)
            except StepCancelled:
                raise
            except StepError as exc:
                if task in LICENSED_TASKS:
                    raise StepError(f"{exc}. {LICENCE_HINT.format(task=task)}") from exc
                raise
            if not out_file.is_file():
                raise StepError(f"TotalSegmentator wrote no output for task '{task}'")
            try:
                label_map = read_label_map(out_file)
            except (TotalSegmentatorError, OSError, ValueError) as exc:
                raise StepError(str(exc)) from exc
            outputs.append(SegmentationOutput(out_file, label_map, task))
        return outputs
