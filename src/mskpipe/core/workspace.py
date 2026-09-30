# SPDX-License-Identifier: Apache-2.0
"""Run workspace: one self-contained folder per pipeline run.

Layout::

    runs/<YYYYMMDD-HHMMSS>_<subject>_<key8>/
        input.json              original path, modality, subject_id, sha256
        config.resolved.yaml    fully resolved PipelineConfig
        00_input/ ... 06_mw2_input/
        logs/

``run_key`` hashes the input content and all result-affecting config
sections (everything except ``runtime``); identical input + settings give
the same key, which the runner uses to find reusable results.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mskpipe.config import InputSpec, PipelineConfig, dump_config, load_config

STEP_DIRS: dict[str, str] = {
    "prepare": "00_input",
    "segment": "01_segment",
    "labelmap": "02_labelmap",
    "mesh": "03_mesh",
    "skeleton": "04_skeleton",
    "attachments": "05_attachments",
    "export_mw2": "06_mw2_input",
}
STEPS: tuple[str, ...] = tuple(STEP_DIRS)

INPUT_FILE = "input.json"
CONFIG_FILE = "config.resolved.yaml"
MANIFEST_FILE = "manifest.json"
LOGS_DIR = "logs"

_RESULT_SECTIONS = tuple(
    name for name in PipelineConfig.model_fields if name not in {"config_version", "runtime"}
)
_CHUNK = 1 << 20


class WorkspaceError(RuntimeError):
    """Workspace cannot be created or opened."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def compute_run_key(input_sha256: str, config: PipelineConfig) -> str:
    """Identifier of (input content, result-affecting settings)."""
    blob = f"{input_sha256}:{config.fingerprint(*_RESULT_SECTIONS)}".encode()
    return hashlib.sha256(blob).hexdigest()


def _nifti_suffix(path: Path) -> str:
    return ".nii.gz" if path.name.lower().endswith(".nii.gz") else ".nii"


@dataclass(frozen=True)
class Workspace:
    root: Path
    run_key: str

    # ------------------------------------------------------------------ creation

    @classmethod
    def create(
        cls, spec: InputSpec, config: PipelineConfig, *, now: datetime | None = None
    ) -> Workspace:
        """Create a new run folder, copy the input and write the resolved config."""
        image = spec.image.expanduser().resolve()
        if not image.is_file():
            raise WorkspaceError(f"Input image not found: {image}")

        sha = file_sha256(image)
        run_key = compute_run_key(sha, config)
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}_{spec.subject_id[:32]}_{run_key[:8]}"
        runs_dir = config.runtime.runs_dir.expanduser().resolve()
        runs_dir.mkdir(parents=True, exist_ok=True)

        root = _unique_dir(runs_dir, base)
        ws = cls(root=root, run_key=run_key)
        try:
            for name in (*STEP_DIRS.values(), LOGS_DIR):
                (root / name).mkdir()
            shutil.copy2(image, ws.input_image_path(image))
            record = {
                "image": str(image),
                "modality": spec.modality.value,
                "subject_id": spec.subject_id,
                "sha256": sha,
                "size_bytes": image.stat().st_size,
                "run_key": run_key,
            }
            _write_json(root / INPUT_FILE, record)
            dump_config(config, root / CONFIG_FILE)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return ws

    @classmethod
    def open(cls, root: str | Path) -> Workspace:
        """Open an existing run folder."""
        root = Path(root).expanduser().resolve()
        missing = [
            n
            for n in (INPUT_FILE, CONFIG_FILE, *STEP_DIRS.values(), LOGS_DIR)
            if not (root / n).exists()
        ]
        if missing:
            raise WorkspaceError(f"Not an msk-pipe run folder: {root} (missing {missing})")
        record = _read_json(root / INPUT_FILE)
        return cls(root=root, run_key=record["run_key"])

    # ------------------------------------------------------------------ contents

    def load_input(self) -> InputSpec:
        record = _read_json(self.root / INPUT_FILE)
        return InputSpec(
            image=Path(record["image"]),
            modality=record["modality"],
            subject_id=record["subject_id"],
        )

    def load_input_record(self) -> dict[str, Any]:
        return _read_json(self.root / INPUT_FILE)

    def load_config(self) -> PipelineConfig:
        return load_config(self.config_path)

    # ------------------------------------------------------------------ paths

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILE

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE

    @property
    def logs_dir(self) -> Path:
        return self.root / LOGS_DIR

    @property
    def input_image(self) -> Path:
        """Copy of the input volume inside the workspace."""
        matches = sorted((self.root / STEP_DIRS["prepare"]).glob("input.nii*"))
        if not matches:
            raise WorkspaceError(f"Input image missing in {self.root}")
        return matches[0]

    def input_image_path(self, original: Path) -> Path:
        return self.root / STEP_DIRS["prepare"] / f"input{_nifti_suffix(original)}"

    def step_dir(self, step: str) -> Path:
        try:
            return self.root / STEP_DIRS[step]
        except KeyError:
            raise WorkspaceError(f"Unknown step '{step}'; expected one of {STEPS}") from None


def _unique_dir(parent: Path, base: str) -> Path:
    for i in range(1000):
        candidate = parent / (base if i == 0 else f"{base}-{i}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise WorkspaceError(f"Cannot create a unique run folder for {base} in {parent}")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
