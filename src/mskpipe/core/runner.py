# SPDX-License-Identifier: Apache-2.0
"""Pipeline runner: ordering, caching, partial runs, logging, manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mskpipe.config import InputSpec, PipelineConfig
from mskpipe.core.device import DeviceReport, resolve_device
from mskpipe.core.manifest import Manifest
from mskpipe.core.step import Step, StepCancelled, StepContext
from mskpipe.core.workspace import STEPS, Workspace

logger = logging.getLogger("mskpipe")

ProgressCallback = Callable[[str, str], None]  # (step, status)
_DONE = ("completed", "cached")
_INPUT_GLOB = "input.nii*"


class PipelineError(RuntimeError):
    """A step failed; ``__cause__`` holds the original exception."""

    def __init__(self, message: str, ws: Workspace, step: str | None = None) -> None:
        super().__init__(message)
        self.ws = ws
        self.step = step


class PipelineCancelled(PipelineError):
    """The run was cancelled by the user."""


@dataclass(frozen=True)
class RunResult:
    ws: Workspace
    manifest: Manifest


def run_pipeline(
    steps: Sequence[Step],
    spec: InputSpec | None = None,
    config: PipelineConfig | None = None,
    *,
    resume: str | Path | None = None,
    from_step: str | None = None,
    until_step: str | None = None,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    device: DeviceReport | None = None,
) -> RunResult:
    """Run ``steps`` for a new input, or continue an existing run (``resume``).

    ``device`` is normally resolved here from ``runtime.device``; pass a report to reuse one
    detection for a whole batch. An unusable ``runtime.device=gpu`` fails before any run
    folder is created.
    """
    _check_order(steps)
    names = [s.name for s in steps]
    for label, value in (("from_step", from_step), ("until_step", until_step)):
        if value is not None and value not in names:
            raise ValueError(f"{label}='{value}' is not one of {names}")

    if resume is not None:
        ws = Workspace.open(resume)
        spec, config = ws.load_input(), ws.load_config()
        device = device or resolve_device(config.runtime.device)
        manifest = (
            Manifest.load(ws.manifest_path) if ws.manifest_path.exists() else Manifest.create(ws)
        )
        manifest.status = "running"
        manifest.finished_at = None
        manifest.save()
    else:
        if spec is None or config is None:
            raise ValueError("spec and config are required unless resume is given")
        device = device or resolve_device(config.runtime.device)
        ws = Workspace.create(spec, config)
        manifest = Manifest.create(ws)
    manifest.device = device.model_dump(mode="json")
    manifest.save()

    handler = _attach_log(ws, config)
    notify = progress or (lambda _step, _status: None)
    start = names.index(from_step) if from_step else 0
    stop = names.index(until_step) if until_step else len(steps) - 1
    upstream = input_fingerprint(ws.load_input_record())
    current: str | None = None
    try:
        logger.info("Run %s (%s, %s)", ws.root.name, spec.subject_id, spec.modality.value)
        logger.info("Device: %s", device.summary())
        for i, step in enumerate(steps[: stop + 1]):
            current = step.name
            if cancel is not None and cancel.is_set():
                raise PipelineCancelled("Run cancelled", ws, step.name)
            fp = step_fingerprint(step, config, upstream)
            upstream = fp
            force = i >= start and from_step is not None

            if not force and _reuse(step, fp, ws, manifest, config, notify):
                continue
            if i < start:
                raise PipelineError(
                    f"Step '{step.name}' has no reusable outputs; run it before --from "
                    f"'{from_step}'",
                    ws,
                    step.name,
                )

            _invalidate_from(step.name, ws, manifest)
            logger.info("[%s] running", step.name)
            notify(step.name, "running")
            with manifest.step(step.name, fp, gpu_index=device.gpu_index) as rec:
                step.run(StepContext(ws, config, spec, rec, step.name, logger, device, cancel))
            notify(step.name, "completed")
            logger.info("[%s] completed in %.1f s", step.name, rec.resources.wall_s)
        manifest.finalize("completed")
        logger.info("Run completed: %s", ws.root)
        return RunResult(ws, manifest)
    except PipelineError as exc:
        manifest.finalize("interrupted" if isinstance(exc, PipelineCancelled) else "failed")
        logger.error("%s", exc)
        raise
    except StepCancelled as exc:
        manifest.finalize("interrupted")
        if current:
            notify(current, "interrupted")
        logger.warning("[%s] %s", current, exc)
        raise PipelineCancelled("Run cancelled", ws, current) from exc
    except KeyboardInterrupt as exc:
        manifest.finalize("interrupted")
        if current:
            notify(current, "interrupted")
        raise PipelineCancelled("Run interrupted", ws, current) from exc
    except Exception as exc:
        manifest.finalize("failed")
        if current:
            notify(current, "failed")
        logger.exception("[%s] failed", current)
        raise PipelineError(f"Step '{current}' failed: {exc}", ws, current) from exc
    finally:
        logger.removeHandler(handler)
        handler.close()


def input_fingerprint(record: dict[str, object]) -> str:
    """Identity of the run input: image content and modality (segmentation depends on both)."""
    blob = json.dumps(
        {"sha256": record["sha256"], "modality": record["modality"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def step_fingerprint(step: Step, config: PipelineConfig, upstream: str) -> str:
    """Hash of the step identity, its config sections and the upstream fingerprint."""
    payload = {
        "step": step.name,
        "version": step.version,
        "config": config.fingerprint(*step.config_sections) if step.config_sections else None,
        "upstream": upstream,
    }
    extra = dict(step.fingerprint_extra(config))
    if extra:  # only when present, so fingerprints of plain steps stay unchanged
        payload["extra"] = extra
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------- caching


def _reuse(
    step: Step,
    fp: str,
    ws: Workspace,
    manifest: Manifest,
    config: PipelineConfig,
    notify: ProgressCallback,
) -> bool:
    own = manifest.steps.get(step.name)
    if own is not None and own.status in _DONE and own.fingerprint == fp:
        logger.info("[%s] up to date", step.name)
        notify(step.name, "cached")
        return True
    if not config.runtime.cache:
        return False
    source = _find_cached(step.name, fp, ws)
    if source is None:
        return False
    src_ws, src_manifest = source
    _invalidate_from(step.name, ws, manifest)
    shutil.copytree(
        src_ws.step_dir(step.name),
        ws.step_dir(step.name),
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(_INPUT_GLOB),
    )
    manifest.mark_step(step.name, "cached", fp)
    rec = manifest.steps[step.name]
    rec.outputs = list(src_manifest.steps[step.name].outputs)
    rec.metrics = {**src_manifest.steps[step.name].metrics, "cached_from": src_ws.root.name}
    manifest.save()
    logger.info("[%s] reused from %s", step.name, src_ws.root.name)
    notify(step.name, "cached")
    return True


def _find_cached(step: str, fp: str, ws: Workspace) -> tuple[Workspace, Manifest] | None:
    """Most recent other run with a finished ``step`` of the same fingerprint."""
    for path in sorted(ws.root.parent.glob("*/manifest.json"), reverse=True):
        if path.parent == ws.root:
            continue
        try:
            other = Manifest.load(path)
            rec = other.steps.get(step)
            if rec is None or rec.status not in _DONE or rec.fingerprint != fp:
                continue
            return Workspace.open(path.parent), other
        except (OSError, ValueError, RuntimeError):  # unreadable or foreign runs
            continue
    return None


def _invalidate_from(step: str, ws: Workspace, manifest: Manifest) -> None:
    """Remove outputs and records of ``step`` and every later step."""
    for name in STEPS[STEPS.index(step) :]:
        manifest.steps.pop(name, None)
        folder = ws.step_dir(name)
        for child in folder.iterdir():
            if child.match(_INPUT_GLOB) and child.parent == ws.step_dir("prepare"):
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    manifest.save()


# ---------------------------------------------------------------------- helpers


def _check_order(steps: Sequence[Step]) -> None:
    names = [s.name for s in steps]
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        raise ValueError(f"Unknown steps {unknown}; expected names from {STEPS}")
    if names != sorted(names, key=STEPS.index) or len(set(names)) != len(names):
        raise ValueError(f"Steps must be unique and ordered as {STEPS}, got {names}")


def _attach_log(ws: Workspace, config: PipelineConfig) -> logging.Handler:
    handler = logging.FileHandler(ws.logs_dir / "run.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    handler.setLevel(config.runtime.log_level)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return handler
