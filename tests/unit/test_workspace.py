# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from pathlib import Path

import pytest

from mskpipe.config import InputSpec, PipelineConfig, load_config
from mskpipe.core.workspace import STEP_DIRS, STEPS, Workspace, WorkspaceError, file_sha256

NOW = datetime(2026, 9, 30, 12, 0, 0)


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "LHDL CT.nii.gz"
    path.parent.mkdir()
    path.write_bytes(b"not a real nifti, content only matters for hashing")
    return path


@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    return load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}"])


def _spec(image: Path) -> InputSpec:
    return InputSpec(image=image, modality="ct")


def test_create_builds_layout(image, config):
    ws = Workspace.create(_spec(image), config, now=NOW)
    assert ws.root.parent == config.runtime.runs_dir.resolve()
    assert ws.root.name == f"20260930-120000_LHDL_CT_{ws.run_key[:8]}"
    for name in STEP_DIRS.values():
        assert (ws.root / name).is_dir()
    assert ws.logs_dir.is_dir()
    assert ws.input_image.name == "input.nii.gz"
    assert file_sha256(ws.input_image) == file_sha256(image)


def test_input_and_config_roundtrip(image, config):
    ws = Workspace.create(_spec(image), config, now=NOW)
    assert ws.load_config() == config
    spec = ws.load_input()
    assert spec.modality == "ct" and spec.subject_id == "LHDL_CT"
    record = ws.load_input_record()
    assert record["sha256"] == file_sha256(image)
    assert record["size_bytes"] == image.stat().st_size


def test_open_existing(image, config):
    ws = Workspace.create(_spec(image), config, now=NOW)
    again = Workspace.open(ws.root)
    assert again == ws


def test_open_rejects_foreign_folder(tmp_path):
    with pytest.raises(WorkspaceError, match="Not an msk-pipe run folder"):
        Workspace.open(tmp_path)


def test_missing_input(tmp_path, config):
    with pytest.raises(WorkspaceError, match="not found"):
        Workspace.create(_spec(tmp_path / "missing.nii.gz"), config)


def test_same_second_gets_unique_folder(image, config):
    a = Workspace.create(_spec(image), config, now=NOW)
    b = Workspace.create(_spec(image), config, now=NOW)
    assert a.root != b.root
    assert b.root.name == a.root.name + "-1"
    assert a.run_key == b.run_key


def test_run_key_ignores_runtime_but_not_results(image, config, tmp_path):
    runs = f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}"
    base = Workspace.create(_spec(image), config, now=NOW)
    threads = Workspace.create(
        _spec(image), load_config(overrides=[runs, "runtime.threads=2"]), now=NOW
    )
    mesh = Workspace.create(
        _spec(image), load_config(overrides=[runs, "mesh.bones.smooth_iterations=5"]), now=NOW
    )
    assert base.run_key == threads.run_key
    assert base.run_key != mesh.run_key


def test_run_key_depends_on_input_content(image, config):
    a = Workspace.create(_spec(image), config, now=NOW)
    image.write_bytes(b"different content")
    b = Workspace.create(_spec(image), config, now=NOW)
    assert a.run_key != b.run_key


def test_step_dir(image, config):
    ws = Workspace.create(_spec(image), config, now=NOW)
    assert [ws.step_dir(s).name for s in STEPS] == list(STEP_DIRS.values())
    with pytest.raises(WorkspaceError, match="Unknown step"):
        ws.step_dir("nonexistent")
