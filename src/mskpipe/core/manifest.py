# SPDX-License-Identifier: Apache-2.0
"""Run manifest: machine-readable record of one pipeline run (``manifest.json``).

It is rewritten atomically after every state change, so an interrupted or
crashed run still leaves a valid file. ``mskpipe stats`` aggregates these.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from mskpipe.core.metrics import ResourceSampler, ResourceUsage
from mskpipe.core.workspace import Workspace

MANIFEST_VERSION = 1

# display name -> distribution name
TRACKED_PACKAGES: dict[str, str] = {
    "mskpipe": "mskpipe",
    "totalsegmentator": "TotalSegmentator",
    "nnunetv2": "nnunetv2",
    "musclemap": "scripts",  # MuscleMap is distributed under the name "scripts"
    "pystaple": "pystaple",
    "fast-simplification": "fast-simplification",
    "torch": "torch",
    "numpy": "numpy",
    "simpleitk": "SimpleITK",
    "vtk": "vtk",
    "pydantic": "pydantic",
}

RunStatus = Literal["running", "completed", "failed", "interrupted"]
StepStatus = Literal["running", "completed", "failed", "interrupted", "cached", "skipped"]


def _now() -> datetime:
    return datetime.now(UTC)


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")  # newer manifests stay readable


class PackageInfo(_Model):
    version: str | None = None
    commit: str | None = None  # for packages installed from git


class GitInfo(_Model):
    commit: str | None = None
    dirty: bool | None = None


class Environment(_Model):
    platform: str
    python: str
    cpu: str
    cpu_logical: int | None
    cpu_physical: int | None
    ram_total_bytes: int
    packages: dict[str, PackageInfo]
    git: GitInfo


class StepRecord(_Model):
    status: StepStatus = "running"
    fingerprint: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    resources: ResourceUsage | None = None
    outputs: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    _root: Path | None = PrivateAttr(default=None)

    def add_output(self, path: str | Path) -> None:
        """Record an output file, relative to the run folder when possible."""
        path = Path(path)
        if self._root is not None:
            with suppress(ValueError):
                path = path.resolve().relative_to(self._root)
        self.outputs.append(path.as_posix())


class Manifest(_Model):
    manifest_version: int = MANIFEST_VERSION
    run_id: str
    run_key: str
    status: RunStatus = "running"
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    config_fingerprint: str
    input: dict[str, Any]
    environment: Environment
    device: dict[str, Any] | None = None
    steps: dict[str, StepRecord] = Field(default_factory=dict)
    _path: Path | None = PrivateAttr(default=None)

    # ------------------------------------------------------------------ io

    @classmethod
    def create(cls, ws: Workspace) -> Manifest:
        config = ws.load_config()
        sections = [n for n in type(config).model_fields if n != "config_version"]
        manifest = cls(
            run_id=ws.root.name,
            run_key=ws.run_key,
            config_fingerprint=config.fingerprint(*sections),
            input=ws.load_input_record(),
            environment=capture_environment(),
        )
        manifest._path = ws.manifest_path
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        path = Path(path)
        manifest = cls.model_validate_json(path.read_text(encoding="utf-8"))
        manifest._path = path
        for rec in manifest.steps.values():
            rec._root = path.parent.resolve()
        return manifest

    def save(self) -> None:
        if self._path is None:
            raise RuntimeError("Manifest has no path; use Manifest.create() or Manifest.load()")
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp.replace(self._path)

    # ------------------------------------------------------------------ steps

    @contextmanager
    def step(
        self, name: str, fingerprint: str | None = None, *, gpu_index: int | None = None
    ) -> Iterator[StepRecord]:
        """Record a step: timing, resources, outputs, metrics and failure."""
        rec = StepRecord(fingerprint=fingerprint, started_at=_now())
        rec._root = self._path.parent.resolve() if self._path else None
        self.steps[name] = rec
        self.save()
        sampler = ResourceSampler(gpu_index=gpu_index)
        try:
            with sampler:
                yield rec
        except KeyboardInterrupt:
            rec.status = "interrupted"
            raise
        except BaseException as exc:
            # StepCancelled (core.step) carries cancelled=True; not imported here (cycle)
            rec.status = "interrupted" if getattr(exc, "cancelled", False) else "failed"
            rec.error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            rec.status = "completed"
        finally:
            rec.finished_at = _now()
            rec.resources = sampler.usage
            self.save()

    def mark_step(self, name: str, status: StepStatus, fingerprint: str | None = None) -> None:
        """Record a step that did not execute (``cached`` or ``skipped``)."""
        now = _now()
        self.steps[name] = StepRecord(
            status=status, fingerprint=fingerprint, started_at=now, finished_at=now
        )
        self.save()

    def finalize(self, status: RunStatus) -> None:
        self.status = status
        self.finished_at = _now()
        self.save()


# ---------------------------------------------------------------------- environment


def capture_environment() -> Environment:
    return Environment(
        platform=platform.platform(),
        python=sys.version.split()[0],
        cpu=platform.processor() or platform.machine(),
        cpu_logical=psutil.cpu_count(logical=True),
        cpu_physical=psutil.cpu_count(logical=False),
        ram_total_bytes=psutil.virtual_memory().total,
        packages={name: package_info(dist) for name, dist in TRACKED_PACKAGES.items()},
        git=git_info(Path(__file__).resolve().parent),
    )


def package_info(dist_name: str) -> PackageInfo:
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return PackageInfo()
    commit = None
    with suppress(ValueError, OSError):
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
    return PackageInfo(version=dist.version, commit=commit)


def git_info(path: Path) -> GitInfo:
    """Commit of the source tree containing ``path``; empty for installed releases."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if not commit:
        return GitInfo()
    status = run("status", "--porcelain", "--untracked-files=no")
    return GitInfo(commit=commit, dirty=bool(status) if status is not None else None)
