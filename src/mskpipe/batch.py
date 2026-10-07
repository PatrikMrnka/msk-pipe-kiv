# SPDX-License-Identifier: Apache-2.0
"""Batch runs: several inputs and/or configurations, run one after another.

Batch file (YAML; paths relative to the file)::

    version: 1
    experiment: E13                 # copied to batch.json, used by `mskpipe stats`
    repeat: 3                       # every run n times (timings: use runtime.cache=false)
    defaults:                       # for every job; a job overrides them
      config: lhdl.yaml
      device: gpu
      set: [runtime.cache=false]
    jobs:
      - name: lhdl
        image: data/lhdl/ct.nii.gz
        modality: ct
        subject: LHDL
        set: [skeleton.side=r]
        matrix:                     # one run per combination (cartesian product)
          attachments.params.nonrigid: [coherent_icp, cpd, none]

CSV alternative, one run per row: ``name,image,modality,subject,config,device,set,until,repeat``
(``set`` separated by ``;``).

Precedence of settings: ``defaults.set`` < job ``set`` < ``matrix`` < command-line ``--set``;
device: command line > job > defaults. Repeats are the outer loop, so drift of the
machine (temperature, other load) is spread over all variants.

Every batch writes ``<runs_dir>/batches/<YYYYMMDD-HHMMSS>_<experiment>/``: ``batch.json``
(entries with run folder, status and time; rewritten after every run) and ``summary.csv``.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mskpipe import api
from mskpipe.config import Device, Modality
from mskpipe.core.device import DeviceError, DeviceReport, resolve_device

__all__ = [
    "BATCH_FILE",
    "BatchEntry",
    "BatchFile",
    "BatchResult",
    "PlannedRun",
    "expand",
    "load_batch",
    "load_batch_index",
    "run_batch",
]

BATCH_FILE = "batch.json"
SUMMARY_FILE = "summary.csv"
FORMAT = "mskpipe.batch"
VERSION = 1
_NAME = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
SUMMARY_COLUMNS = (
    "index",
    "job",
    "variant",
    "repeat",
    "status",
    "exit_code",
    "wall_s",
    "run_id",
    "run_dir",
    "error",
)

EntryStatus = Literal[
    "pending", "running", "completed", "failed", "interrupted", "setup_error", "skipped"
]


# ---------------------------------------------------------------------------- file model


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JobDefaults(_Model):
    config: Path | None = Field(None, description="Config YAML (relative to the batch file).")
    device: Device | None = None
    set: list[str] = Field(default_factory=list, description="key.path=value overrides.")
    until: str | None = Field(None, description="Last step (default: export_mw2).")


class JobSpec(JobDefaults):
    name: str | None = Field(None, pattern=_NAME, description="Job name (default: subject).")
    image: Path
    modality: Modality
    subject: str | None = None
    matrix: dict[str, list[Any]] = Field(
        default_factory=dict, description="key.path -> values; one run per combination."
    )
    repeat: int | None = Field(None, ge=1, le=100)

    @field_validator("matrix")
    @classmethod
    def _non_empty(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for key, values in value.items():
            if not values:
                raise ValueError(f"matrix '{key}' has no values")
            if "=" in key or not key.strip():
                raise ValueError(f"matrix key '{key}' must be a config path like a.b.c")
        return value


class BatchFile(_Model):
    version: Literal[1] = 1
    experiment: str | None = Field(None, pattern=_NAME)
    repeat: int = Field(1, ge=1, le=100)
    defaults: JobDefaults = Field(default_factory=JobDefaults)
    jobs: list[JobSpec] = Field(min_length=1)


# ---------------------------------------------------------------------------- loading


def load_batch(path: str | Path) -> tuple[BatchFile, Path]:
    """Read a YAML or CSV batch file; returns it with its folder (base of relative paths).

    Raises :class:`api.SetupError`.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise api.SetupError(f"Cannot read batch file '{path}': {exc.strerror}") from None
    try:
        data = _read_csv(text) if path.suffix.lower() == ".csv" else yaml.safe_load(text)
    except (yaml.YAMLError, csv.Error, ValueError) as exc:
        raise api.SetupError(f"Invalid batch file '{path}': {exc}") from None
    if not isinstance(data, dict):
        raise api.SetupError(f"Batch file '{path}' must contain a mapping with 'jobs'")
    try:
        batch = BatchFile.model_validate(data)
    except ValidationError as exc:
        lines = [f"Invalid batch file ({path}):"]
        lines += [f"  {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        raise api.SetupError("\n".join(lines)) from None
    return batch, path.resolve().parent


def _read_csv(text: str) -> dict[str, Any]:
    jobs = []
    for i, row in enumerate(csv.DictReader(text.splitlines()), start=2):
        row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        if not any(row.values()):
            continue
        unknown = set(row) - {
            "name", "image", "modality", "subject", "config", "device", "set", "until", "repeat",
        }  # fmt: skip
        if unknown:
            raise ValueError(f"unknown CSV columns {sorted(unknown)}")
        job: dict[str, Any] = {k: v for k, v in row.items() if v and k != "set"}
        if row.get("set"):
            job["set"] = [s.strip() for s in row["set"].split(";") if s.strip()]
        if "repeat" in job:
            try:
                job["repeat"] = int(job["repeat"])
            except ValueError:
                raise ValueError(f"row {i}: repeat must be an integer") from None
        jobs.append(job)
    return {"jobs": jobs}


# ---------------------------------------------------------------------------- expansion


@dataclass(frozen=True)
class PlannedRun:
    index: int
    job: str
    variant: str  # "" or "nonrigid=cpd,side=l"
    repeat: int  # 1-based
    request: api.RunRequest
    matrix: dict[str, Any] = field(default_factory=dict)


def expand(
    batch: BatchFile,
    base: Path,
    *,
    overrides: Sequence[str] = (),
    device: Device | str | None = None,
    runs_dir: Path | None = None,
    cache: bool | None = None,
) -> list[PlannedRun]:
    """All runs of the batch, in execution order (repeat > job > matrix combination)."""
    items: list[tuple[str, dict[str, Any], JobSpec]] = []
    names: dict[str, int] = {}
    for job in batch.jobs:
        name = job.name or job.subject or _stem(job.image)
        names[name] = names.get(name, 0) + 1
        if names[name] > 1:
            name = f"{name}-{names[name]}"
        keys = list(job.matrix)
        for values in itertools.product(*(job.matrix[k] for k in keys)) if keys else [()]:
            items.append((name, dict(zip(keys, values, strict=True)), job))
    repeats = max([batch.repeat, *(j.repeat or 0 for j in batch.jobs)])

    planned: list[PlannedRun] = []
    for r in range(1, repeats + 1):
        for name, combo, job in items:
            if r > (job.repeat or batch.repeat):
                continue
            config = job.config or batch.defaults.config
            sets = [*batch.defaults.set, *job.set]
            sets += [f"{k}={_yaml_value(v)}" for k, v in combo.items()]
            sets += list(overrides)
            request = api.RunRequest(
                image=_abs(job.image, base),
                modality=job.modality,
                subject=job.subject,
                config=_abs(config, base) if config else None,
                overrides=tuple(sets),
                device=device or job.device or batch.defaults.device,
                runs_dir=runs_dir,
                cache=cache,
                until_step=job.until or batch.defaults.until,
            )
            variant = ",".join(f"{k.rsplit('.', 1)[-1]}={_yaml_value(v)}" for k, v in combo.items())
            planned.append(PlannedRun(len(planned) + 1, name, variant, r, request, combo))
    return planned


def _stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _abs(path: Path, base: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else base / path


def _yaml_value(value: Any) -> str:
    return yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("...").strip()


# ---------------------------------------------------------------------------- running


class BatchEntry(BaseModel):
    index: int
    job: str
    variant: str = ""
    repeat: int = 1
    matrix: dict[str, Any] = Field(default_factory=dict)
    overrides: list[str] = Field(default_factory=list)
    image: str | None = None
    status: EntryStatus = "pending"
    exit_code: int | None = None
    wall_s: float | None = None
    run_id: str | None = None
    run_dir: str | None = None
    error: str | None = None


class BatchIndex(BaseModel):
    format: str = FORMAT
    version: int = VERSION
    experiment: str | None = None
    batch_file: str | None = None
    batch_sha256: str | None = None
    status: Literal["running", "completed", "failed", "interrupted"] = "running"
    started_at: datetime
    finished_at: datetime | None = None
    entries: list[BatchEntry]


@dataclass
class BatchResult:
    folder: Path
    index: BatchIndex

    @property
    def entries(self) -> list[BatchEntry]:
        return self.index.entries

    @property
    def exit_code(self) -> int:
        statuses = {e.status for e in self.entries}
        if "interrupted" in statuses:
            return 130
        if statuses <= {"completed"}:
            return 0
        return 1


BatchCallback = Callable[[BatchEntry, api.RunEvent | None], None]


def check_batch(planned: Sequence[PlannedRun]) -> list[tuple[PlannedRun, api.Issue]]:
    """Prepare and preflight every run; issues of identical setups are reported once."""
    found: list[tuple[PlannedRun, api.Issue]] = []
    seen: set[tuple[str, str, str]] = set()
    for p in planned:
        try:
            prepared = api.prepare_run(p.request)
        except api.SetupError as exc:
            found.append((p, api.Issue("error", "setup", str(exc))))
            continue
        for issue in api.preflight(prepared):
            key = (issue.severity, issue.where, issue.message)
            if key not in seen:
                seen.add(key)
                found.append((p, issue))
    return found


def run_batch(
    planned: Sequence[PlannedRun],
    folder: Path,
    *,
    experiment: str | None = None,
    batch_file: Path | None = None,
    stop_on_error: bool = False,
    cancel: threading.Event | None = None,
    on_event: BatchCallback | None = None,
    run: Callable[..., api.RunSummary] | None = None,
) -> BatchResult:
    """Run ``planned`` one after another; ``folder`` receives batch.json and summary.csv.

    A failing run does not stop the batch unless ``stop_on_error``; a cancelled run
    (``cancel`` or Ctrl+C) stops it, the remaining runs are recorded as ``skipped``.
    GPU detection is done once per requested device.
    """
    run_one = run or api.run
    folder.mkdir(parents=True, exist_ok=True)
    index = BatchIndex(
        experiment=experiment,
        batch_file=str(batch_file) if batch_file else None,
        batch_sha256=_sha256(batch_file) if batch_file else None,
        started_at=datetime.now(UTC),
        entries=[
            BatchEntry(
                index=p.index,
                job=p.job,
                variant=p.variant,
                repeat=p.repeat,
                matrix=p.matrix,
                overrides=list(p.request.overrides),
                image=str(p.request.image) if p.request.image else None,
            )
            for p in planned
        ],
    )
    result = BatchResult(folder, index)
    _save(result)
    devices: dict[str, DeviceReport] = {}
    stop = False
    for p, entry in zip(planned, index.entries, strict=True):
        if stop or (cancel is not None and cancel.is_set()):
            entry.status = "skipped"
            continue
        callback = (lambda ev, _e=entry: on_event(_e, ev)) if on_event else None
        entry.status = "running"
        _save(result)
        if on_event:
            on_event(entry, None)
        try:
            prepared = api.prepare_run(p.request)
            requested = prepared.config.runtime.device.value
            if requested not in devices:
                devices[requested] = resolve_device(requested)
            summary = run_one(prepared, on_event=callback, cancel=cancel, device=devices[requested])
        except KeyboardInterrupt:  # between runs (the runner handles it inside a run)
            entry.status, entry.exit_code, entry.error = "interrupted", 130, "interrupted"
        except (api.SetupError, DeviceError) as exc:
            entry.status, entry.exit_code, entry.error = "setup_error", api.EXIT_SETUP, str(exc)
        except Exception as exc:  # a bug must not lose a long batch: record it and go on
            entry.status, entry.exit_code = "failed", 1
            entry.error = f"{type(exc).__name__}: {exc}"
        else:
            entry.status = summary.status if summary.status != "running" else "failed"
            entry.exit_code = summary.exit_code
            entry.wall_s = round(summary.executed_wall_s, 3)
            entry.run_id, entry.run_dir = summary.run_id, str(summary.run_dir)
            entry.error = summary.error
        if on_event:
            on_event(entry, None)
        _save(result)
        stop = entry.status == "interrupted" or (stop_on_error and entry.status != "completed")
    statuses = {e.status for e in index.entries}
    if "interrupted" in statuses or (cancel is not None and cancel.is_set()):
        index.status = "interrupted"
    elif statuses <= {"completed"}:
        index.status = "completed"
    else:
        index.status = "failed"
    index.finished_at = datetime.now(UTC)
    _save(result)
    return result


def batch_folder(runs_dir: Path, experiment: str | None, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = Path(runs_dir).expanduser() / "batches" / f"{stamp}_{experiment or 'batch'}"
    folder, i = base, 1
    while folder.exists():
        folder = base.with_name(f"{base.name}-{i}")
        i += 1
    return folder


def load_batch_index(folder: str | Path) -> BatchIndex:
    path = Path(folder)
    path = path / BATCH_FILE if path.is_dir() else path
    index = BatchIndex.model_validate_json(path.read_text(encoding="utf-8"))
    if index.format != FORMAT:
        raise ValueError(f"{path} is not an mskpipe batch index")
    return index


def _save(result: BatchResult) -> None:
    path = result.folder / BATCH_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(result.index.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    with (result.folder / SUMMARY_FILE).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for e in result.index.entries:
            writer.writerow(e.model_dump(mode="json"))


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def entries_json(entries: Sequence[BatchEntry]) -> str:
    return json.dumps([e.model_dump(mode="json") for e in entries], indent=2, ensure_ascii=False)
