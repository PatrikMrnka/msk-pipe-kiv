# SPDX-License-Identifier: Apache-2.0
"""Resource usage of a pipeline step, including child processes.

A background thread samples the whole process tree (RSS, CPU times) and,
optionally, the GPU memory in use. Sampling is needed because external
tools (TotalSegmentator, MuscleMap) run as subprocesses and, on Windows,
psutil does not report CPU time of terminated children.

All values are sampled and therefore approximate:
* ``peak_rss_bytes`` - peak sum of RSS over the process tree,
* ``cpu_s`` - user + system time of the tree during the step,
* ``gpu_mem_peak_bytes`` - peak increase of device memory in use over the
  value at step start (device-wide; per-process data is not available
  under Windows WDDM).
"""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from types import TracebackType

import psutil
from pydantic import BaseModel

try:
    import pynvml
except ImportError:  # optional: GPU memory is then not measured
    pynvml = None


class ResourceUsage(BaseModel):
    wall_s: float
    cpu_s: float
    peak_rss_bytes: int
    gpu_mem_peak_bytes: int | None = None
    samples: int


class ResourceSampler:
    """Context manager measuring the current process and all its descendants."""

    def __init__(self, interval_s: float = 0.5, gpu_index: int | None = None) -> None:
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.usage: ResourceUsage | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._root = psutil.Process()
        self._cpu: dict[tuple[int, float], float] = {}
        self._cpu_baseline = 0.0
        self._peak_rss = 0
        self._samples = 0
        self._gpu = _GpuMemory(gpu_index) if gpu_index is not None else None
        self._t0 = 0.0

    def __enter__(self) -> ResourceSampler:
        times = self._root.cpu_times()
        self._cpu_baseline = times.user + times.system
        if self._gpu:
            self._gpu.start()
        self._t0 = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="mskpipe-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._sample()
        wall = time.perf_counter() - self._t0
        gpu_peak = self._gpu.stop() if self._gpu else None
        self.usage = ResourceUsage(
            wall_s=round(wall, 3),
            cpu_s=round(max(sum(self._cpu.values()) - self._cpu_baseline, 0.0), 3),
            peak_rss_bytes=self._peak_rss,
            gpu_mem_peak_bytes=gpu_peak,
            samples=self._samples,
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        try:
            procs = [self._root, *self._root.children(recursive=True)]
        except psutil.Error:
            procs = [self._root]
        rss = 0
        for proc in procs:
            try:
                with proc.oneshot():
                    rss += proc.memory_info().rss
                    times = proc.cpu_times()
                    key = (proc.pid, proc.create_time())
            except psutil.Error:
                continue
            self._cpu[key] = max(self._cpu.get(key, 0.0), times.user + times.system)
        self._peak_rss = max(self._peak_rss, rss)
        if self._gpu:
            self._gpu.sample()
        self._samples += 1


class _GpuMemory:
    """Device-wide GPU memory in use, via NVML; silently disabled if unavailable."""

    def __init__(self, index: int) -> None:
        self.index = index
        self._handle: object | None = None
        self._baseline = 0
        self._peak = 0

    def start(self) -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self._baseline = self._peak = pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
        except pynvml.NVMLError:  # no driver / no such GPU
            self._handle = None

    def sample(self) -> None:
        if self._handle is None:
            return
        with suppress(pynvml.NVMLError):
            self._peak = max(self._peak, pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def stop(self) -> int | None:
        if self._handle is None:
            return None
        with suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()
        return max(self._peak - self._baseline, 0)
