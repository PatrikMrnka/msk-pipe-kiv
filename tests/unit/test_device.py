# SPDX-License-Identifier: Apache-2.0
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from mskpipe.config import InputSpec, load_config
from mskpipe.core import device as dev
from mskpipe.core.device import (
    DeviceError,
    DeviceReport,
    GpuInfo,
    TorchInfo,
    probe_torch_cuda,
    resolve_device,
)
from mskpipe.core.manifest import Manifest
from mskpipe.core.runner import run_pipeline
from mskpipe.core.step import Step, StepContext

RTX = GpuInfo(
    index=0,
    name="NVIDIA GeForce RTX 5070 Ti",
    memory_total_bytes=16 << 30,
    compute_capability="12.0",
)
NO_GPU = {"detected_by": None, "driver_version": None, "driver_cuda": None, "gpus": []}
HAS_GPU = {"detected_by": "nvml", "driver_version": "576.52", "driver_cuda": "12.9", "gpus": [RTX]}
TORCH_OK = TorchInfo(
    version="2.7.1+cu128",
    cuda="12.8",
    cuda_available=True,
    arch_list=["sm_90", "sm_100", "sm_120"],
    kernel_ok=True,
)


@pytest.fixture
def fake(monkeypatch):
    """Control detection results; records whether the torch probe was used."""
    state = SimpleNamespace(gpus=NO_GPU, torch=TORCH_OK, probed=0)

    def probe(python=None):
        state.probed += 1
        return state.torch

    monkeypatch.setattr(dev, "detect_gpus", lambda: state.gpus)
    monkeypatch.setattr(dev, "probe_torch_cuda", probe)
    return state


# ---------------------------------------------------------------------- selection


def test_cpu_never_probes_torch(fake):
    fake.gpus = HAS_GPU
    report = resolve_device("cpu")
    assert (report.selected, report.gpu_index, fake.probed) == ("cpu", None, 0)
    assert report.gpus == [RTX]  # still recorded for the manifest


def test_auto_uses_usable_gpu(fake):
    fake.gpus = HAS_GPU
    report = resolve_device("auto")
    assert (report.selected, report.gpu_index) == ("cuda", 0)
    assert report.gpu == RTX and report.torch == TORCH_OK
    assert "RTX 5070 Ti" in report.summary()


def test_gpu_request_succeeds(fake):
    fake.gpus = HAS_GPU
    assert resolve_device("gpu").reason == "requested"


@pytest.mark.parametrize(
    ("gpus", "torch", "reason"),
    [
        (NO_GPU, TORCH_OK, "no NVIDIA GPU found"),
        ({**HAS_GPU, "driver_version": "551.86"}, TORCH_OK, "older than 570"),
        (HAS_GPU, TorchInfo(), "PyTorch is not installed"),
        (HAS_GPU, TorchInfo(version="2.7.1+cpu"), "CPU-only build 2.7.1+cpu"),
        (
            HAS_GPU,
            TorchInfo(
                version="2.4.1+cu121",
                cuda="12.1",
                cuda_available=True,
                arch_list=["sm_80", "sm_90"],
                error="RuntimeError: no kernel image",
            ),
            "has no kernels for sm_120",
        ),
    ],
)
def test_auto_falls_back_with_reason_and_gpu_fails(fake, gpus, torch, reason):
    fake.gpus, fake.torch = gpus, torch
    report = resolve_device("auto")
    assert report.selected == "cpu" and report.gpu_index is None
    assert reason in report.reason and report.reason.startswith("auto: ")
    with pytest.raises(DeviceError, match=re.escape(reason)):
        resolve_device("gpu")


def test_unknown_device_value():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


# ---------------------------------------------------------------------- report helpers


def test_env_and_device_for():
    cpu = DeviceReport(requested="auto", selected="cpu", reason="x")
    gpu = DeviceReport(requested="gpu", selected="cuda", gpu_index=1, reason="x", gpus=[RTX])
    assert cpu.env()["CUDA_VISIBLE_DEVICES"] == ""
    assert gpu.env()["CUDA_VISIBLE_DEVICES"] == "1"
    assert gpu.env()["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert cpu.env(threads=4)["OMP_NUM_THREADS"] == "4" and "OMP_NUM_THREADS" not in cpu.env()
    assert (gpu.device_for(True), gpu.device_for(False), cpu.device_for(True)) == (
        "cuda",
        "cpu",
        "cpu",
    )


# ---------------------------------------------------------------------- detectors


class _FakeNvml:
    class NVMLError(Exception):
        pass

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlSystemGetDriverVersion(self):
        return b"576.52"

    def nvmlSystemGetCudaDriverVersion(self):
        return 12090

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetName(self, h):
        return "NVIDIA GeForce RTX 5070 Ti"

    def nvmlDeviceGetMemoryInfo(self, h):
        return SimpleNamespace(total=16 << 30)

    def nvmlDeviceGetCudaComputeCapability(self, h):
        return (12, 0)


def test_nvml_detection(monkeypatch):
    monkeypatch.setattr(dev, "pynvml", _FakeNvml())
    found = dev._detect_nvml()
    assert found == {**HAS_GPU, "driver_version": "576.52"}


def test_nvml_without_driver(monkeypatch):
    nvml = _FakeNvml()

    def no_driver():
        raise nvml.NVMLError("Driver Not Loaded")

    nvml.nvmlInit = no_driver
    monkeypatch.setattr(dev, "pynvml", nvml)
    assert dev._detect_nvml() is None


def test_parse_nvidia_smi():
    out = "0, NVIDIA GeForce RTX 5070 Ti, 16303, 576.52, 12.0\n"
    found = dev._parse_nvidia_smi(out)
    assert found["driver_version"] == "576.52" and found["detected_by"] == "nvidia-smi"
    (gpu,) = found["gpus"]
    assert (gpu.name, gpu.compute_capability) == ("NVIDIA GeForce RTX 5070 Ti", "12.0")
    assert gpu.memory_total_bytes == 16303 * 1024 * 1024
    assert dev._parse_nvidia_smi("0, GPU, [N/A], 470.1\n")["gpus"][0].compute_capability is None
    assert dev._parse_nvidia_smi("No devices were found\n") is None


def test_real_probe_runs_in_subprocess():
    info = probe_torch_cuda()  # torch absent (CI) or CPU/GPU build (local); must not fail
    assert info.error is None or not info.error.startswith("probe failed")
    assert "torch" not in sys.modules  # the pipeline process itself never imports torch


# ---------------------------------------------------------------------- runner integration


class Probe(Step):
    name = "prepare"
    seen: ClassVar[dict[str, str]] = {}

    def run(self, ctx: StepContext) -> None:
        Probe.seen["device"] = ctx.device.selected
        ctx.run_command(
            [
                sys.executable,
                "-c",
                "import os; print('CVD=' + repr(os.environ['CUDA_VISIBLE_DEVICES']))",
            ]
        )


@pytest.fixture
def spec(tmp_path: Path) -> InputSpec:
    image = tmp_path / "s01.nii.gz"
    image.write_bytes(b"volume")
    return InputSpec(image=image, modality="ct")


def test_runner_records_device_and_hides_gpu_on_cpu(fake, spec, tmp_path):
    fake.gpus = HAS_GPU
    cfg = load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}"])
    result = run_pipeline([Probe()], spec, cfg)
    manifest = Manifest.load(result.ws.manifest_path)
    assert manifest.device["selected"] == "cpu" and manifest.device["gpus"][0]["name"] == RTX.name
    assert Probe.seen["device"] == "cpu"
    assert "CVD=''" in (result.ws.logs_dir / "prepare.log").read_text(encoding="utf-8")
    assert "Device: cpu" in (result.ws.logs_dir / "run.log").read_text(encoding="utf-8")


def test_runner_uses_given_gpu_report(spec, tmp_path):
    cfg = load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}"])
    report = DeviceReport(
        requested="gpu", selected="cuda", gpu_index=0, reason="requested", gpus=[RTX]
    )
    result = run_pipeline([Probe()], spec, cfg, device=report)
    assert Probe.seen["device"] == "cuda"
    assert "CVD='0'" in (result.ws.logs_dir / "prepare.log").read_text(encoding="utf-8")
    assert Manifest.load(result.ws.manifest_path).device["gpu_index"] == 0


def test_unusable_gpu_fails_before_run_folder(fake, spec, tmp_path):
    runs = tmp_path / "runs"
    cfg = load_config(overrides=[f"runtime.runs_dir={runs.as_posix()}", "runtime.device=gpu"])
    with pytest.raises(DeviceError):
        run_pipeline([Probe()], spec, cfg)
    assert not runs.exists() or not any(runs.iterdir())
