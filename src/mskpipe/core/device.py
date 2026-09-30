# SPDX-License-Identifier: Apache-2.0
"""Compute device selection: which NVIDIA GPU (if any) a run uses, and why.

Detection order:

1. NVML (``nvidia-ml-py``, always installed): GPUs, driver, CUDA version of the driver.
2. ``nvidia-smi``: fallback when NVML cannot be loaded.
3. PyTorch probe in a subprocess (only for ``gpu``/``auto``): CUDA available, and a tiny
   kernel runs, which also catches a torch build without kernels for the GPU architecture.

The probe runs in a subprocess so that the pipeline process never imports torch or creates a
CUDA context, which would inflate the memory measured for every step. Only an NVIDIA driver
is required from the user; CUDA itself comes with the torch wheel.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from contextlib import suppress
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

try:
    import pynvml
except ImportError:  # pragma: no cover - declared dependency, kept optional for safety
    pynvml = None

MIN_DRIVER = 570  # needed by torch wheels built with CUDA 12.8 (Blackwell, RTX 50xx)
PROBE_TIMEOUT_S = 120

_GPU_HINT = (
    "Run msk-pipe in its GPU environment (e.g. `pixi run -e gpu mskpipe ...`) with an NVIDIA "
    f"driver {MIN_DRIVER} or newer, or use runtime.device=cpu (or auto)."
)


class DeviceError(RuntimeError):
    """``runtime.device=gpu`` was requested but no usable GPU was found."""


class GpuInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    name: str
    memory_total_bytes: int | None = None
    compute_capability: str | None = None  # e.g. "12.0"


class TorchInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str | None = None
    cuda: str | None = None  # CUDA version torch was built with
    cuda_available: bool = False
    arch_list: list[str] = Field(default_factory=list)
    kernel_ok: bool = False
    error: str | None = None


class DeviceReport(BaseModel):
    """Result of device selection; stored as ``manifest.device``."""

    model_config = ConfigDict(frozen=True)

    requested: Literal["cpu", "gpu", "auto"]
    selected: Literal["cpu", "cuda"]
    gpu_index: int | None = None
    reason: str
    detected_by: Literal["nvml", "nvidia-smi"] | None = None
    driver_version: str | None = None
    driver_cuda: str | None = None  # highest CUDA version the driver supports
    gpus: list[GpuInfo] = Field(default_factory=list)
    torch: TorchInfo | None = None

    @property
    def gpu(self) -> GpuInfo | None:
        if self.gpu_index is None:
            return None
        return next((g for g in self.gpus if g.index == self.gpu_index), None)

    def device_for(self, supports_gpu: bool) -> Literal["cpu", "cuda"]:
        """Device for one tool/plugin: GPU only if selected and supported."""
        return "cuda" if supports_gpu and self.selected == "cuda" else "cpu"

    def env(self, threads: int | None = None) -> dict[str, str]:
        """Environment variables for external tools run by the steps."""
        env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}  # same numbering as NVML/nvidia-smi
        env["CUDA_VISIBLE_DEVICES"] = "" if self.selected == "cpu" else str(self.gpu_index)
        if threads is not None:
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[var] = str(threads)
        return env

    def summary(self) -> str:
        if self.selected == "cuda" and self.gpu is not None:
            head = f"cuda:{self.gpu_index} ({self.gpu.name})"
        else:
            head = "cpu"
        return f"{head} - {self.reason}"


# ---------------------------------------------------------------------- selection


def resolve_device(requested: str, *, probe_torch: bool | None = None) -> DeviceReport:
    """Translate ``runtime.device`` into the device actually used.

    ``cpu`` never selects a GPU (GPUs are still listed for the manifest). ``auto`` falls back
    to CPU with a reason; ``gpu`` raises :class:`DeviceError` instead, so CPU timings are
    never recorded as GPU ones.
    """
    requested = str(requested)
    if requested not in ("cpu", "gpu", "auto"):
        raise ValueError(f"Unknown device '{requested}'")
    found = detect_gpus()
    base: dict[str, Any] = {"requested": requested, **found}

    if requested == "cpu":
        return DeviceReport(selected="cpu", reason="requested", **base)

    def fallback(reason: str, torch: TorchInfo | None = None) -> DeviceReport:
        if requested == "gpu":
            raise DeviceError(f"GPU requested but not usable: {reason}. {_GPU_HINT}")
        return DeviceReport(selected="cpu", reason=f"auto: {reason}", torch=torch, **base)

    gpus: list[GpuInfo] = found["gpus"]
    if not gpus:
        return fallback("no NVIDIA GPU found (NVML and nvidia-smi)")
    major = _driver_major(found["driver_version"])
    if major is not None and major < MIN_DRIVER:
        return fallback(f"NVIDIA driver {found['driver_version']} is older than {MIN_DRIVER}")
    if probe_torch is False:
        return fallback("torch probe disabled")

    torch = probe_torch_cuda()
    if torch.version is None:
        return fallback("PyTorch is not installed in this environment", torch)
    if not torch.cuda_available:
        build = f"CPU-only build {torch.version}" if torch.cuda is None else torch.version
        return fallback(f"torch.cuda is not available ({build})", torch)
    if not torch.kernel_ok:
        cc = gpus[0].compute_capability
        arch = f"sm_{cc.replace('.', '')}" if cc else "?"
        detail = torch.error or "CUDA test kernel failed"
        if cc and arch not in torch.arch_list:
            detail = f"torch {torch.version} has no kernels for {arch} ({gpus[0].name})"
        return fallback(detail, torch)

    return DeviceReport(
        selected="cuda",
        gpu_index=gpus[0].index,
        reason="requested" if requested == "gpu" else "auto: usable NVIDIA GPU",
        torch=torch,
        **base,
    )


# ---------------------------------------------------------------------- detection


def detect_gpus() -> dict[str, Any]:
    """GPUs and driver info; empty result (not an error) when there is no NVIDIA driver."""
    for detector in (_detect_nvml, _detect_nvidia_smi):
        found = detector()
        if found is not None:
            return found
    return {"detected_by": None, "driver_version": None, "driver_cuda": None, "gpus": []}


def _detect_nvml() -> dict[str, Any] | None:
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None
    try:
        driver = _text(pynvml.nvmlSystemGetDriverVersion())
        cuda = pynvml.nvmlSystemGetCudaDriverVersion()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                cc: str | None = "{}.{}".format(*pynvml.nvmlDeviceGetCudaComputeCapability(handle))
            except pynvml.NVMLError:
                cc = None
            gpus.append(
                GpuInfo(
                    index=i,
                    name=_text(pynvml.nvmlDeviceGetName(handle)),
                    memory_total_bytes=int(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                    compute_capability=cc,
                )
            )
    except pynvml.NVMLError:
        return None
    finally:
        with suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()
    return {
        "detected_by": "nvml",
        "driver_version": driver,
        "driver_cuda": f"{cuda // 1000}.{cuda % 1000 // 10}" if cuda else None,
        "gpus": gpus,
    }


def _detect_nvidia_smi() -> dict[str, Any] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    # compute_cap is not supported by old drivers -> retry without it
    for fields in (
        "index,name,memory.total,driver_version,compute_cap",
        "index,name,memory.total,driver_version",
    ):
        try:
            out = subprocess.run(
                [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        return _parse_nvidia_smi(out)
    return None


def _parse_nvidia_smi(out: str) -> dict[str, Any] | None:
    gpus, driver = [], None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        mem = int(parts[2]) * 1024 * 1024 if parts[2].isdigit() else None  # MiB
        cc = parts[4] if len(parts) > 4 and parts[4] not in ("", "[N/A]") else None
        gpus.append(
            GpuInfo(
                index=int(parts[0]), name=parts[1], memory_total_bytes=mem, compute_capability=cc
            )
        )
        driver = parts[3]
    if not gpus:
        return None
    return {
        "detected_by": "nvidia-smi",
        "driver_version": driver,
        "driver_cuda": None,
        "gpus": gpus,
    }


_PROBE = r"""
import json
info = {}
try:
    import torch
except ImportError:
    print(json.dumps(info)); raise SystemExit
info.update(version=torch.__version__, cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(), arch_list=[])
if info["cuda_available"]:
    try:
        info["arch_list"] = torch.cuda.get_arch_list()
        x = torch.ones(8, device="cuda")
        info["kernel_ok"] = float((x * 2).sum().item()) == 16.0
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
print(json.dumps(info))
"""


def probe_torch_cuda(python: str | None = None) -> TorchInfo:
    """Check PyTorch CUDA support in a separate interpreter (default: the current one)."""
    try:
        out = subprocess.run(
            [python or sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=True,
        ).stdout
        data = json.loads(out.strip().splitlines()[-1])
    except subprocess.CalledProcessError as exc:
        return TorchInfo(error=f"probe failed: {(exc.stderr or '').strip()[-300:]}")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return TorchInfo(error=f"probe failed: {exc}")
    if not data:
        return TorchInfo()
    return TorchInfo.model_validate(data)


# ---------------------------------------------------------------------- helpers


def _text(value: str | bytes) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def _driver_major(version: str | None) -> int | None:
    if not version:
        return None
    head = version.split(".")[0]
    return int(head) if head.isdigit() else None
