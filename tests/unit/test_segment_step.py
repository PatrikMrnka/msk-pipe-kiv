# SPDX-License-Identifier: Apache-2.0
"""Step ``segment`` with fake segmenters, and cancellation of external tools."""

import sys
import threading
import time
from pathlib import Path
from typing import ClassVar

import nibabel as nib
import numpy as np
import psutil
import pytest

from mskpipe.config import InputSpec, load_config
from mskpipe.core.manifest import Manifest
from mskpipe.core.registry import Registry
from mskpipe.core.runner import PipelineCancelled, PipelineError, run_pipeline
from mskpipe.core.step import Step, StepContext
from mskpipe.io.segmentation import SEGMENTATION_FILE, SegmentationIndex
from mskpipe.plugins.base import SegmentationOutput, SegmenterPlugin
from mskpipe.steps.labelmap import LabelmapStep
from mskpipe.steps.segment import SegmentStep

AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])  # LPS-like axes; x = -i


def _write(path: Path, data: np.ndarray, affine: np.ndarray = AFFINE) -> Path:
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


class FakeTS(SegmenterPlugin):
    name = "totalsegmentator"
    supports_gpu = True
    mode = "ok"  # ok | wrong_grid | missing_label | none
    requests: ClassVar[list] = []

    def segment(self, ctx, image, out_dir, params, request):
        FakeTS.requests.append(request)
        if FakeTS.mode == "none":
            return []
        img = nib.load(image)
        data = np.zeros(img.shape, np.uint8)
        data[2:6, 2:10, 2:6] = 77  # hip_left (x > 0 => i < 0 side... irrelevant here)
        data[10:14, 2:10, 2:6] = 78
        data[2:6, 2:10, 8:14] = 75
        data[10:14, 2:10, 8:14] = 76
        data[7:9, 2:10, 2:4] = 25  # sacrum, between the hip bones
        data[7:9, 2:10, 4:6] = 26  # S1 body
        affine = img.affine.copy()
        if FakeTS.mode == "wrong_grid":
            affine[0, 3] += 5
        labels = {
            "femur_left": 75,
            "femur_right": 76,
            "hip_left": 77,
            "hip_right": 78,
            "sacrum": 25,
            "vertebrae_S1": 26,
        }
        if FakeTS.mode == "missing_label":
            labels.pop("femur_left")
        out = _write(out_dir / "totalsegmentator_total.nii.gz", data, affine)
        outputs = [SegmentationOutput(out, labels, "total")]
        tasks = {t for job in request.jobs for t in job.tasks}
        if tasks & {"appendicular_bones", "appendicular_bones_mr"}:
            task = "appendicular_bones_mr" if request.modality == "mri" else "appendicular_bones"
            app = np.zeros(img.shape, np.uint8)
            app[2:6, 2:10, 15:19] = 1  # tibia, both legs
            app[10:14, 2:10, 15:19] = 1
            app[2:6, 11:14, 15:19] = 2  # fibula
            app[10:14, 11:14, 15:19] = 2
            path = _write(out_dir / f"totalsegmentator_{task}.nii.gz", app, affine)
            outputs.append(SegmentationOutput(path, {"tibia": 1, "fibula": 2}, task))
        return outputs


class FakeMM(SegmenterPlugin):
    name = "musclemap"
    supports_gpu = True
    modalities = frozenset({"ct"})

    def segment(self, ctx, image, out_dir, params, request):
        img = nib.load(image)
        data = np.zeros(img.shape, np.uint16)
        data[10:14, 2:10, 15:19] = 8162  # tibia_r
        data[10:14, 11:15, 8:14] = 7182  # biceps long head r
        out = _write(out_dir / "musclemap_wholebody.nii.gz", data, img.affine)
        labels = {"tibia_r": 8162, "biceps_femoris_long_head_r": 7182, "sacrum": 6160}
        return [SegmentationOutput(out, labels, "wholebody")]


@pytest.fixture(autouse=True)
def _reset():
    FakeTS.mode, FakeTS.requests = "ok", []


@pytest.fixture
def registry() -> Registry:
    reg = Registry()
    reg.register(FakeTS)
    reg.register(FakeMM)
    return reg


@pytest.fixture
def spec(tmp_path: Path) -> InputSpec:
    image = _write(tmp_path / "subject.nii.gz", np.zeros((16, 16, 20), np.int16))
    return InputSpec(image=image, modality="ct")


def config(tmp_path: Path, *overrides: str):
    return load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}", *overrides])


def test_segment_writes_index_and_metrics(tmp_path, spec, registry):
    result = run_pipeline([SegmentStep(registry)], spec, config(tmp_path))
    out = result.ws.step_dir("segment")
    index = SegmentationIndex.load(out / SEGMENTATION_FILE)
    assert [(o.tool, o.task, o.file) for o in index] == [
        ("musclemap", "wholebody", "musclemap_wholebody.nii.gz"),
        ("totalsegmentator", "total", "totalsegmentator_total.nii.gz"),
        ("totalsegmentator", "appendicular_bones", "totalsegmentator_appendicular_bones.nii.gz"),
    ]
    rec = result.manifest.steps["segment"]
    assert rec.status == "completed"
    m = rec.metrics
    assert m["modality"] == "ct"
    assert m["input"]["shape"] == [16, 16, 20] and m["input"]["orientation"] == "LPS"
    ts = m["tools"]["totalsegmentator"]
    assert ts["voxels"]["hip_left"] == 4 * 8 * 4 and ts["not_provided"] == []
    assert ts["voxels"]["tibia"] == 2 * 4 * 8 * 4  # default: tibia/fibula from TS (as BP)
    assert ts["tasks"] == ["total", "appendicular_bones"]
    assert ts["device"] == "cpu"
    mmm = m["tools"]["musclemap"]
    assert mmm["voxels"]["biceps_femoris_long_head_r"] > 0
    assert "tibia_r" not in mmm["voxels"]
    assert "gracilis_l" in mmm["not_provided"]
    assert any("musclemap" in w for w in m["warnings"])
    assert "segmentation.json" in " ".join(rec.outputs)
    # both legs are requested from the tools, whatever skeleton.side is
    assert {"hip_left", "femur_left", "femur_right"} <= FakeTS.requests[0].labels


@pytest.mark.parametrize("tibia_fibula", ["ts_appendicular", "musclemap"])
def test_segment_then_labelmap(tmp_path, spec, registry, tibia_fibula):
    cfg = config(
        tmp_path,
        "labelmap.bones.min_voxels=1",
        "labelmap.muscles.min_voxels=1",
        f"segmentation.tibia_fibula={tibia_fibula}",
    )
    result = run_pipeline([SegmentStep(registry), LabelmapStep()], spec, cfg)
    assert result.manifest.steps["labelmap"].status == "completed"
    struct = result.manifest.steps["labelmap"].metrics["structures"]
    assert {"pelvis_no_sacrum", "femur_r", "tibia_r", "biceps_femoris_r"} <= set(struct)
    assert "tibia_r" not in result.manifest.steps["labelmap"].metrics["missing"]


def test_missing_label_is_a_warning(tmp_path, spec, registry):
    FakeTS.mode = "missing_label"
    result = run_pipeline([SegmentStep(registry)], spec, config(tmp_path))
    assert result.manifest.steps["segment"].metrics["tools"]["totalsegmentator"][
        "not_provided"
    ] == ["femur_left"]


@pytest.mark.parametrize(("mode", "message"), [("wrong_grid", "input grid"), ("none", "no output")])
def test_bad_output_fails(tmp_path, spec, registry, mode, message):
    FakeTS.mode = mode
    with pytest.raises(PipelineError) as err:
        run_pipeline([SegmentStep(registry)], spec, config(tmp_path))
    assert message in str(err.value.__cause__)


def test_unsupported_modality(tmp_path, registry):
    image = _write(tmp_path / "mri.nii.gz", np.zeros((8, 8, 8), np.int16))
    spec = InputSpec(image=image, modality="mri")
    with pytest.raises(PipelineError) as err:
        run_pipeline([SegmentStep(registry)], spec, config(tmp_path))
    assert "modality 'mri'" in str(err.value.__cause__)


def test_modality_is_part_of_the_cache_key(tmp_path, registry):
    reg = Registry()
    reg.register(FakeTS)

    class AnyMM(FakeMM):
        modalities = frozenset({"ct", "mri"})

    reg.register(AnyMM)
    image = _write(tmp_path / "s.nii.gz", np.zeros((16, 16, 20), np.int16))
    cfg = config(tmp_path)
    a = run_pipeline([SegmentStep(reg)], InputSpec(image=image, modality="ct"), cfg)
    b = run_pipeline([SegmentStep(reg)], InputSpec(image=image, modality="mri"), cfg)
    c = run_pipeline([SegmentStep(reg)], InputSpec(image=image, modality="ct"), cfg)
    fa, fb, fc = (r.manifest.steps["segment"] for r in (a, b, c))
    assert fa.fingerprint != fb.fingerprint
    assert fc.status == "cached" and fc.fingerprint == fa.fingerprint
    assert FakeTS.requests[1].modality == "mri"


def test_fingerprint_depends_on_musclemap_model_version():
    step = SegmentStep(Registry.default(external=False))  # real plugins, not executed
    a = step.fingerprint_extra(load_config())
    b = step.fingerprint_extra(
        load_config(overrides=["segmentation.musclemap.model_version='1.3'"])
    )
    assert a["tools"]["totalsegmentator"] == b["tools"]["totalsegmentator"]
    assert a["tools"]["musclemap"]["model_version"] == "1.4"
    assert b["tools"]["musclemap"]["model_version"] == "1.3"


def test_side_does_not_change_the_segment_fingerprint(registry):
    step = SegmentStep(registry)
    assert step.fingerprint_extra(load_config()) == step.fingerprint_extra(
        load_config(overrides=["skeleton.side=l"])
    )


# ---------------------------------------------------------------------- cancellation

SLEEPER = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "print(child.pid, flush=True)\n"
    "time.sleep(60)\n"
)


class Sleeper(Step):
    name = "segment"

    def run(self, ctx: StepContext) -> None:
        ctx.run_command([sys.executable, "-c", SLEEPER])


def test_cancel_terminates_the_tool_and_its_children(tmp_path, spec):
    cancel = threading.Event()
    seen: dict = {}

    def progress(step: str, status: str) -> None:
        if status == "running":
            threading.Timer(2.0, cancel.set).start()

    start = time.monotonic()
    with pytest.raises(PipelineCancelled) as err:
        run_pipeline([Sleeper()], spec, config(tmp_path), progress=progress, cancel=cancel)
    assert time.monotonic() - start < 30
    ws = err.value.ws
    manifest = Manifest.load(ws.manifest_path)
    assert manifest.status == "interrupted"
    assert manifest.steps["segment"].status == "interrupted"
    log = (ws.logs_dir / "segment.log").read_text(encoding="utf-8").split()
    seen["child"] = int(log[0])
    assert not psutil.pid_exists(seen["child"]) or (
        psutil.Process(seen["child"]).status() == psutil.STATUS_ZOMBIE
    )


def test_run_command_passes_utf8_env(tmp_path, spec):
    class Env(Step):
        name = "segment"

        def run(self, ctx: StepContext) -> None:
            ctx.run_command(
                [sys.executable, "-c", "import os; print(os.environ['PYTHONIOENCODING'], 'Ž')"]
            )

    result = run_pipeline([Env()], spec, config(tmp_path))
    text = (result.ws.logs_dir / "segment.log").read_text(encoding="utf-8")
    assert "utf-8 Ž" in text


def test_tool_error_message_contains_last_output_line(tmp_path, spec):
    class Failing(Step):
        name = "segment"

        def run(self, ctx: StepContext) -> None:
            ctx.run_command(
                [sys.executable, "-c", "raise RuntimeError('operator torchvision::nms missing')"]
            )

    with pytest.raises(PipelineError) as err:
        run_pipeline([Failing()], spec, config(tmp_path))
    message = str(err.value.__cause__)
    assert "exited with code 1: RuntimeError: operator torchvision::nms missing" in message
    assert "logs/segment.log" in message
