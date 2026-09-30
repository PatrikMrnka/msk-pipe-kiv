# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from mskpipe.config import InputSpec, load_config
from mskpipe.core.manifest import Manifest, capture_environment, git_info, package_info
from mskpipe.core.metrics import ResourceSampler
from mskpipe.core.workspace import Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    image = tmp_path / "subject.nii.gz"
    image.write_bytes(b"fake volume")
    config = load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}"])
    return Workspace.create(
        InputSpec(image=image, modality="mri"), config, now=datetime(2026, 9, 30)
    )


def test_create_writes_valid_manifest(ws):
    manifest = Manifest.create(ws)
    data = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert data["status"] == "running"
    assert data["run_key"] == ws.run_key
    assert data["input"]["modality"] == "mri"
    assert Manifest.load(ws.manifest_path).run_id == manifest.run_id


def test_successful_step(ws):
    manifest = Manifest.create(ws)
    out = ws.step_dir("mesh") / "femur_r.obj"
    with manifest.step("mesh", "abc") as rec:
        out.write_text("v 0 0 0\n", encoding="utf-8")
        rec.add_output(out)
        rec.metrics["n_faces"] = 1234
        time.sleep(0.05)
    manifest.finalize("completed")

    loaded = Manifest.load(ws.manifest_path)
    step = loaded.steps["mesh"]
    assert loaded.status == "completed" and loaded.finished_at is not None
    assert step.status == "completed" and step.fingerprint == "abc"
    assert step.outputs == ["03_mesh/femur_r.obj"]
    assert step.metrics == {"n_faces": 1234}
    assert step.resources.wall_s >= 0.05
    assert step.resources.peak_rss_bytes > 0


def test_failed_step_is_recorded_and_reraised(ws):
    manifest = Manifest.create(ws)
    with pytest.raises(ValueError, match="boom"), manifest.step("skeleton"):
        raise ValueError("boom")
    step = Manifest.load(ws.manifest_path).steps["skeleton"]
    assert step.status == "failed"
    assert step.error == "ValueError: boom"
    assert step.resources is not None


def test_interrupted_step(ws):
    manifest = Manifest.create(ws)
    with pytest.raises(KeyboardInterrupt), manifest.step("segment"):
        raise KeyboardInterrupt
    assert Manifest.load(ws.manifest_path).steps["segment"].status == "interrupted"


def test_mark_cached(ws):
    manifest = Manifest.create(ws)
    manifest.mark_step("segment", "cached", "fp")
    step = Manifest.load(ws.manifest_path).steps["segment"]
    assert step.status == "cached" and step.resources is None


def test_no_temp_file_left(ws):
    Manifest.create(ws)
    assert not list(ws.root.glob("*.tmp"))


def test_environment_and_packages():
    env = capture_environment()
    assert env.ram_total_bytes > 0
    assert env.packages["pydantic"].version is not None
    assert package_info("definitely-not-installed-package").version is None


def test_git_info_outside_repo(tmp_path):
    assert git_info(tmp_path).commit is None


def test_sampler_counts_child_processes():
    code = "import time; x = b'\\x01' * (200 * 2**20); time.sleep(1.0)"
    with ResourceSampler(interval_s=0.1) as sampler:
        subprocess.run([sys.executable, "-c", code], check=True)
    usage = sampler.usage
    assert usage.peak_rss_bytes >= 200 * 2**20
    assert usage.wall_s >= 1.0
    assert usage.samples >= 5
    assert usage.gpu_mem_peak_bytes is None
