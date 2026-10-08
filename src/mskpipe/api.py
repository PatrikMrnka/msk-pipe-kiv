# SPDX-License-Identifier: Apache-2.0
"""Programmatic interface shared by the CLI, batch runs and the GUI.

Typical use::

    from mskpipe import api

    request = api.RunRequest(image=Path("ct.nii.gz"), modality="ct", device="gpu")
    prepared = api.prepare_run(request)          # config + input checked, nothing written
    issues = api.preflight(prepared)             # licence key, atlas, plugins, NIfTI header
    summary = api.run(prepared, on_event=print)  # never raises for a failing step
    summary.status, summary.exit_code, summary.mw2_dir

Errors before a run folder exists (bad request or config, unusable GPU) raise
:class:`SetupError`; a failing or cancelled step is reported in :class:`RunSummary`.

Light to import: steps and plugins import their heavy libraries only when they run.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from mskpipe.config import (
    ConfigError,
    Device,
    InputSpec,
    Modality,
    PipelineConfig,
    dump_config,
    json_schema,
    load_config,
    render_template,
)
from mskpipe.core.device import DeviceError, DeviceReport, resolve_device
from mskpipe.core.manifest import Manifest
from mskpipe.core.registry import PluginError, Registry, default_registry, resolve_plugins
from mskpipe.core.runner import PipelineCancelled, PipelineError, run_pipeline
from mskpipe.core.step import Step
from mskpipe.core.workspace import STEPS, Workspace, WorkspaceError

if TYPE_CHECKING:
    from mskpipe.io.modality import ModalityGuess

__all__ = [
    "PIPELINE_STEPS",
    "Issue",
    "PreparedRun",
    "RunEvent",
    "RunRequest",
    "RunSummary",
    "SetupError",
    "StepSummary",
    "check_image",
    "config_schema",
    "config_template",
    "config_yaml",
    "default_steps",
    "describe_image",
    "detect_modality",
    "finalize_stale",
    "preflight",
    "prepare_run",
    "resolved_config",
    "run",
    "summarize",
]

logger = logging.getLogger("mskpipe")

PIPELINE_STEPS: tuple[str, ...] = STEPS[1:]
AUTO = "auto"  # modality detected from the image  # "prepare" = run folder set up by the runner
MW2_SETUP_FILE = "setup_MuscleGeneratorTool.xml"

RunStatus = Literal["running", "completed", "failed", "interrupted"]
EXIT_CODES: dict[str, int] = {"completed": 0, "failed": 1, "running": 1, "interrupted": 130}
EXIT_SETUP = 2  # bad request, config or device: nothing was run


class SetupError(ValueError):
    """The run cannot start (request, config, input or device); nothing was written."""


# ---------------------------------------------------------------------------- request


@dataclass(frozen=True)
class RunRequest:
    """What to run. Either a new input (``image`` + ``modality``) or ``resume``.

    ``device``, ``runs_dir`` and ``cache`` are shortcuts for ``runtime.*`` overrides and
    take precedence over ``overrides``. A resumed run keeps its stored configuration;
    only the device may be changed.
    """

    image: Path | None = None
    modality: Modality | str | None = None
    subject: str | None = None
    config: Path | None = None
    overrides: tuple[str, ...] = ()
    device: Device | str | None = None
    runs_dir: Path | None = None
    cache: bool | None = None
    from_step: str | None = None
    until_step: str | None = None
    resume: Path | None = None

    def all_overrides(self) -> list[str]:
        """``overrides`` followed by the shortcut options (later wins)."""
        out = list(self.overrides)
        if self.device is not None:
            out.append(f"runtime.device={Device(self.device).value}")
        if self.runs_dir is not None:
            out.append(f"runtime.runs_dir={Path(self.runs_dir).as_posix()}")
        if self.cache is not None:
            out.append(f"runtime.cache={'true' if self.cache else 'false'}")
        return out

    def check(self) -> None:
        """Raise :class:`SetupError` for contradictory or incomplete requests."""
        for label, value in (("from_step", self.from_step), ("until_step", self.until_step)):
            if value is not None and value not in PIPELINE_STEPS:
                raise SetupError(f"{label} '{value}' is not one of {', '.join(PIPELINE_STEPS)}")
        order = PIPELINE_STEPS.index
        if self.from_step and self.until_step and order(self.from_step) > order(self.until_step):
            raise SetupError(f"from_step '{self.from_step}' is after '{self.until_step}'")
        if self.device is not None:
            try:
                Device(self.device)
            except ValueError:
                raise SetupError(f"device must be cpu, gpu or auto, got '{self.device}'") from None
        if self.resume is not None:
            given = [
                name
                for name, value in (
                    ("image", self.image),
                    ("modality", self.modality),
                    ("subject", self.subject),
                    ("config", self.config),
                    ("overrides", self.overrides),
                    ("runs_dir", self.runs_dir),
                    ("cache", self.cache),
                )
                if value not in (None, ())
            ]
            if given:
                raise SetupError(
                    f"A resumed run keeps its input and configuration; remove {given} "
                    "(start a new run to change them)"
                )
            return
        if self.image is None or self.modality is None:
            raise SetupError("An input image and its modality (ct, mri or auto) are required")
        if str(self.modality) not in (*(m.value for m in Modality), AUTO):
            raise SetupError(f"modality must be ct, mri or auto, got '{self.modality}'")


@dataclass(frozen=True)
class PreparedRun:
    """A checked request: the input, the resolved config and the steps that will run."""

    request: RunRequest
    spec: InputSpec
    config: PipelineConfig
    steps: tuple[str, ...]
    workspace: Workspace | None = None  # set when resuming
    notes: tuple[str, ...] = ()  # logged at the start of the run (e.g. detected modality)

    @property
    def modality(self) -> str:
        return self.spec.modality.value


def prepare_run(request: RunRequest, registry: Registry | None = None) -> PreparedRun:
    """Validate ``request`` and resolve its configuration (plugin defaults filled in).

    Nothing is written. Raises :class:`SetupError`.
    """
    request.check()
    first = PIPELINE_STEPS.index(request.from_step) if request.from_step else 0
    last = PIPELINE_STEPS.index(request.until_step) if request.until_step else None
    steps = PIPELINE_STEPS[first : None if last is None else last + 1]
    if request.resume is not None:
        try:
            ws = Workspace.open(request.resume)
            return PreparedRun(request, ws.load_input(), ws.load_config(), steps, ws)
        except (WorkspaceError, ConfigError, OSError, ValueError) as exc:
            raise SetupError(f"Cannot resume '{request.resume}': {exc}") from None

    assert request.image is not None
    image = Path(request.image).expanduser()
    if not image.is_file():
        raise SetupError(f"Input image not found: {image}")
    modality, notes = request.modality, ()
    if str(modality) == AUTO:
        guess = detect_modality(image)
        if guess.modality is None:
            raise SetupError(
                f"Cannot detect the modality of {image.name} ({guess.reason}); "
                "give it explicitly (ct or mri)"
            )
        modality, notes = guess.modality, (f"Modality: {guess}",)
    try:
        spec = InputSpec.model_validate(
            {"image": image, "modality": modality, "subject_id": request.subject}
        )
    except ValidationError as exc:
        msgs = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()
        )
        raise SetupError(f"Invalid input: {msgs}") from None
    config = resolved_config(request.config, request.all_overrides(), registry)
    return PreparedRun(request, spec, config, steps, notes=notes)


def detect_modality(path: str | Path) -> ModalityGuess:
    """CT or MRI from the JSON sidecar, the header or the intensities (reads the volume)."""
    from mskpipe.io.modality import detect_modality as detect

    return detect(path)


def resolved_config(
    path: str | Path | None = None,
    overrides: Sequence[str] = (),
    registry: Registry | None = None,
) -> PipelineConfig:
    """Defaults < YAML ``path`` < ``key.path=value`` overrides, plugin defaults filled in.

    Raises :class:`SetupError` with the user-facing message of the config error.
    """
    try:
        return resolve_plugins(load_config(path, overrides), registry)
    except ConfigError as exc:
        raise SetupError(str(exc)) from None


# ---------------------------------------------------------------------------- preflight


@dataclass(frozen=True)
class Issue:
    """A problem found before a run. ``error`` would make the run fail."""

    severity: Literal["error", "warning"]
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.where}: {self.message}"


def preflight(prepared: PreparedRun, registry: Registry | None = None) -> list[Issue]:
    """Cheap checks of what the planned steps need: input header, plugins and their data.

    Catches what would otherwise fail minutes into a run (e.g. a missing TotalSegmentator
    licence key or attachment atlas after the segmentation).
    """
    reg = registry or default_registry()
    issues: list[Issue] = []
    if "segment" in prepared.steps:
        image = prepared.workspace.input_image if prepared.workspace else prepared.spec.image
        issues += check_image(image)
    for where, kind, name in _plugins(prepared):
        try:
            plugin = reg.get(kind, name)
        except PluginError as exc:
            issues.append(Issue("error", where, str(exc)))
            continue
        missing = plugin.missing()
        if missing:
            issues.append(Issue("error", where, f"'{name}' needs {', '.join(missing)}"))
            continue
        try:
            problems = plugin.preflight(prepared.config, prepared.modality)
        except Exception as exc:  # a broken third-party check must not block the run
            issues.append(Issue("warning", where, f"'{name}' preflight failed: {exc}"))
            continue
        issues += [Issue("error", where, p) for p in problems]
    return issues


def _plugins(prepared: PreparedRun) -> list[tuple[str, str, str]]:
    config, out = prepared.config, []
    if "segment" in prepared.steps:
        from mskpipe.labelmap.scheme import SchemeError, load_scheme

        try:
            tools = sorted(load_scheme().requests(config, prepared.modality))
        except SchemeError:
            tools = []  # reported by the step with its own message
        out += [("segmentation", "segmenter", t) for t in tools]
    if "skeleton" in prepared.steps:
        out.append(("skeleton", "skeleton", config.skeleton.backend))
    if "attachments" in prepared.steps:
        out.append(("attachments", "attachments", config.attachments.method))
    return out


def check_image(path: Path) -> list[Issue]:
    """Header checks of an input volume (3D, qform/sform, suspicious 1 mm spacing)."""
    import nibabel as nib
    import numpy as np

    where = "input"
    try:
        img = nib.load(str(path))
    except Exception as exc:  # nibabel raises many types for broken files
        return [Issue("error", where, f"Cannot read NIfTI header of {path.name}: {exc}")]
    issues: list[Issue] = []
    header = img.header
    if len(img.shape) < 3 or (len(img.shape) > 3 and any(int(n) > 1 for n in img.shape[3:])):
        issues.append(Issue("error", where, f"Expected a 3D volume, got shape {img.shape}"))
    codes = (int(header["qform_code"]), int(header["sform_code"]))
    if codes == (0, 0):
        issues.append(
            Issue(
                "warning",
                where,
                "No qform/sform in the NIfTI header: position and orientation are unknown, "
                "meshes will not match the patient frame",
            )
        )
    zooms = np.asarray(header.get_zooms()[:3], dtype=float)
    if np.allclose(zooms, 1.0):
        issues.append(
            Issue(
                "warning",
                where,
                "Voxel spacing is exactly 1 x 1 x 1 mm; check that the header was not written "
                "with a default spacing (a wrong spacing scales every mesh)",
            )
        )
    return issues


_check_image = check_image  # name used before the GUI


def describe_image(path: Path) -> str:
    """One line about an input volume: shape, voxel spacing, orientation (header only)."""
    import nibabel as nib

    img = nib.load(str(path))
    shape = " x ".join(str(int(n)) for n in img.shape[:3])
    zooms = " x ".join(f"{float(z):.4g}" for z in img.header.get_zooms()[:3])
    axes = "".join(nib.aff2axcodes(img.affine))
    return f"{shape} voxels, {zooms} mm, {axes}"


# ---------------------------------------------------------------------------- events


def _now() -> datetime:
    return datetime.now(UTC)


class RunEvent(BaseModel):
    """Progress of a run, for GUIs and machine-readable output (one JSON object per line)."""

    kind: Literal["run_started", "step", "log", "run_finished"]
    time: datetime | None = None
    run_dir: str | None = None
    step: str | None = None
    status: str | None = None
    wall_s: float | None = None
    steps: list[str] | None = None
    device: str | None = None
    level: str | None = None
    message: str | None = None
    exit_code: int | None = None

    def to_line(self) -> str:
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_line(cls, line: str) -> RunEvent:
        return cls.model_validate_json(line)


EventCallback = Callable[[RunEvent], None]


class _EventLogHandler(logging.Handler):
    def __init__(self, emit: EventCallback, level: int) -> None:
        super().__init__(level)
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit(
                RunEvent(
                    kind="log", time=_now(), level=record.levelname, message=record.getMessage()
                )
            )
        except Exception:  # a GUI callback must never break the run
            self.handleError(record)


# ---------------------------------------------------------------------------- summary


class StepSummary(BaseModel):
    name: str
    status: str
    wall_s: float | None = None
    cpu_s: float | None = None
    peak_rss_bytes: int | None = None
    cached_from: str | None = None
    error: str | None = None


class RunSummary(BaseModel):
    """State of one run folder, read from its ``manifest.json``."""

    run_dir: Path
    run_id: str
    status: RunStatus
    subject: str | None = None
    modality: str | None = None
    device: str | None = None
    device_reason: str | None = None
    steps: list[StepSummary]
    failed_step: str | None = None
    error: str | None = None
    mw2_dir: Path | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]

    @property
    def executed_wall_s(self) -> float:
        """Wall time of the steps executed in this run (cached steps excluded)."""
        return sum(s.wall_s or 0.0 for s in self.steps if s.status == "completed")


def summarize(run_dir: str | Path) -> RunSummary:
    """Summary of an existing run folder (any status)."""
    ws = Workspace.open(run_dir)
    if not ws.manifest_path.is_file():
        raise WorkspaceError(f"No manifest.json in {ws.root}")
    m = Manifest.load(ws.manifest_path)
    steps: list[StepSummary] = []
    failed: StepSummary | None = None
    for name, rec in m.steps.items():
        res = rec.resources
        s = StepSummary(
            name=name,
            status=rec.status,
            wall_s=round(res.wall_s, 3) if res else None,
            cpu_s=round(res.cpu_s, 3) if res else None,
            peak_rss_bytes=res.peak_rss_bytes if res else None,
            cached_from=rec.metrics.get("cached_from"),
            error=rec.error,
        )
        steps.append(s)
        if failed is None and rec.status in ("failed", "interrupted"):
            failed = s
    device = m.device or {}
    export = m.steps.get("export_mw2")
    mw2 = ws.step_dir("export_mw2")
    mw2_ok = (
        export is not None
        and export.status in ("completed", "cached")
        and (mw2 / MW2_SETUP_FILE).is_file()
    )
    return RunSummary(
        run_dir=ws.root,
        run_id=m.run_id,
        status=m.status,
        subject=m.input.get("subject_id"),
        modality=m.input.get("modality"),
        device=device.get("selected"),
        device_reason=device.get("reason"),
        steps=steps,
        failed_step=failed.name if failed else None,
        error=failed.error if failed else None,
        mw2_dir=mw2 if mw2_ok else None,
        created_at=m.created_at,
        finished_at=m.finished_at,
    )


def finalize_stale(run_dir: str | Path, reason: str = "process terminated") -> bool:
    """Mark a run left in ``running`` (its process was killed) as ``interrupted``.

    Returns True if the manifest was changed.
    """
    ws = Workspace.open(run_dir)
    if not ws.manifest_path.is_file():
        return False
    m = Manifest.load(ws.manifest_path)
    if m.status != "running":
        return False
    now = _now()
    for rec in m.steps.values():
        if rec.status == "running":
            rec.status = "interrupted"
            rec.finished_at = now
            rec.error = rec.error or reason
    m.finalize("interrupted")
    return True


# ---------------------------------------------------------------------------- run


def default_steps(registry: Registry | None = None) -> list[Step]:
    """All pipeline steps, in order (``segment`` ... ``export_mw2``)."""
    from mskpipe.steps.attachments import AttachmentsStep
    from mskpipe.steps.export_mw2 import ExportMw2Step
    from mskpipe.steps.labelmap import LabelmapStep
    from mskpipe.steps.mesh import MeshStep
    from mskpipe.steps.segment import SegmentStep
    from mskpipe.steps.skeleton import SkeletonStep

    return [
        SegmentStep(registry),
        LabelmapStep(),
        MeshStep(),
        SkeletonStep(registry),
        AttachmentsStep(registry),
        ExportMw2Step(),
    ]


def run(
    request: RunRequest | PreparedRun,
    *,
    on_event: EventCallback | None = None,
    log_level: int = logging.INFO,
    cancel: threading.Event | None = None,
    device: DeviceReport | None = None,
    steps: Sequence[Step] | None = None,
    registry: Registry | None = None,
) -> RunSummary:
    """Run the pipeline (or continue a run) and return its summary.

    ``on_event`` receives :class:`RunEvent` objects (run start, step status changes, log
    records of ``log_level`` and above, run end) from the calling thread. Setting
    ``cancel`` stops the run: a running tool and its child processes are terminated, the
    run is recorded as ``interrupted``. ``device`` reuses one GPU detection for several
    runs (batch). Raises :class:`SetupError` if the run cannot start.
    """
    prepared = request if isinstance(request, PreparedRun) else prepare_run(request, registry)
    req = prepared.request
    emit = on_event or (lambda _event: None)
    pipeline = list(steps) if steps is not None else default_steps(registry)
    state: dict[str, Any] = {}

    if device is None and req.resume is not None and req.device is not None:
        device = _resolve_device(Device(req.device).value)

    def started(ws: Workspace, manifest: Manifest) -> None:
        state["manifest"] = manifest
        dev = manifest.device or {}
        emit(
            RunEvent(
                kind="run_started",
                time=_now(),
                run_dir=str(ws.root),
                steps=list(prepared.steps),
                device=dev.get("selected"),
                message=dev.get("reason"),
            )
        )

    def progress(step: str, status: str) -> None:
        wall = None
        manifest: Manifest | None = state.get("manifest")
        rec = manifest.steps.get(step) if manifest else None
        if status == "completed" and rec is not None and rec.resources is not None:
            wall = round(rec.resources.wall_s, 3)
        emit(RunEvent(kind="step", time=_now(), step=step, status=status, wall_s=wall))

    handler = _EventLogHandler(emit, log_level) if on_event is not None else None
    if handler is not None:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    if prepared.workspace is None:
        logger.info(
            "Preparing the run (device '%s', copying and hashing the input)",
            prepared.config.runtime.device.value,
        )
    else:
        logger.info("Resuming %s", prepared.workspace.root.name)
    for note in prepared.notes:
        logger.info("%s", note)
    try:
        try:
            if prepared.workspace is not None:
                result = run_pipeline(
                    pipeline,
                    resume=prepared.workspace.root,
                    from_step=req.from_step,
                    until_step=req.until_step,
                    progress=progress,
                    cancel=cancel,
                    device=device,
                    on_start=started,
                )
            else:
                result = run_pipeline(
                    pipeline,
                    prepared.spec,
                    prepared.config,
                    from_step=req.from_step,
                    until_step=req.until_step,
                    progress=progress,
                    cancel=cancel,
                    device=device,
                    on_start=started,
                )
            root = result.ws.root
        except DeviceError as exc:
            raise SetupError(str(exc)) from None
        except WorkspaceError as exc:
            raise SetupError(str(exc)) from None
        except PipelineError as exc:  # incl. PipelineCancelled; recorded in the manifest
            root = exc.ws.root
            if not isinstance(exc, PipelineCancelled):
                state["error"] = str(exc)
        summary = summarize(root)
        if summary.status == "failed" and summary.error is None:
            summary = summary.model_copy(update={"error": state.get("error")})
    finally:
        if handler is not None:
            logger.removeHandler(handler)
    emit(
        RunEvent(
            kind="run_finished",
            time=_now(),
            run_dir=str(summary.run_dir),
            status=summary.status,
            wall_s=round(summary.executed_wall_s, 3),
            message=summary.error,
            exit_code=summary.exit_code,
        )
    )
    return summary


def _resolve_device(requested: str) -> DeviceReport:
    try:
        return resolve_device(requested)
    except DeviceError as exc:
        raise SetupError(str(exc)) from None


# ---------------------------------------------------------------------------- config


def config_template() -> str:
    """Commented YAML with every key and its default (``mskpipe config init``)."""
    return render_template()


def config_schema() -> dict[str, Any]:
    """JSON Schema of the config file (editors, GUI forms)."""
    return json_schema()


def config_yaml(config: PipelineConfig) -> str:
    """Resolved config as YAML (as written to ``config.resolved.yaml``)."""
    return dump_config(config)
