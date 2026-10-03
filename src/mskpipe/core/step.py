# SPDX-License-Identifier: Apache-2.0
"""Step interface shared by all pipeline steps."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, ClassVar

import psutil

from mskpipe.config import InputSpec, PipelineConfig
from mskpipe.core.device import DeviceReport
from mskpipe.core.manifest import StepRecord
from mskpipe.core.workspace import Workspace

POLL_S = 0.5  # how often a running tool checks for cancellation
KILL_GRACE_S = 5.0  # time to exit after terminate() before kill()
TAIL_LINES = 20  # last tool output lines kept for the error message
# child Pythons: UTF-8 output (Windows consoles default to cp1250/cp852), no buffering
_CHILD_ENV = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


class StepError(RuntimeError):
    """A step failed in an expected way (bad input, tool error)."""


class StepCancelled(StepError):
    """The run was cancelled while the step was running."""

    cancelled = True


@dataclass
class StepContext:
    ws: Workspace
    config: PipelineConfig
    input: InputSpec
    record: StepRecord
    step_name: str
    logger: logging.Logger
    device: DeviceReport = field(
        default_factory=lambda: DeviceReport(requested="cpu", selected="cpu", reason="default")
    )
    cancel: threading.Event | None = None

    @property
    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def check_cancel(self) -> None:
        """Raise :class:`StepCancelled` if the run was cancelled (call between long parts)."""
        if self.cancelled:
            raise StepCancelled(f"Step '{self.step_name}' cancelled")

    @property
    def out_dir(self) -> Path:
        """Output folder of the current step."""
        return self.ws.step_dir(self.step_name)

    def step_dir(self, step: str) -> Path:
        """Output folder of another (typically upstream) step."""
        return self.ws.step_dir(step)

    def run_command(
        self,
        args: Sequence[str | Path],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        """Run an external tool; output goes to ``logs/<step>.log``.

        The tool gets the run's device (``CUDA_VISIBLE_DEVICES``; empty on CPU runs) and
        thread limits; ``env`` adds to or overrides the current environment. On cancel
        (``ctx.cancel``) or Ctrl+C the tool and all its child processes are terminated.
        """
        argv = [str(a) for a in args]
        full_env = {
            **os.environ,
            **_CHILD_ENV,
            **self.device.env(self.config.runtime.threads),
            **(env or {}),
        }
        log_path = self.ws.logs_dir / f"{self.step_name}.log"
        tool = Path(argv[0]).name if len(argv) < 3 or argv[1] != "-m" else argv[2]
        self.logger.info("$ %s", subprocess.list2cmdline(argv))
        with log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=full_env,
                cwd=cwd,
            )
            assert proc.stdout is not None
            tail: deque[str] = deque(maxlen=TAIL_LINES)
            reader = threading.Thread(
                target=_pump, args=(proc.stdout, log, self.logger, tail), daemon=True
            )
            reader.start()
            try:
                code = self._wait(proc, tool)
            except BaseException:
                terminate_tree(proc)
                raise
            finally:
                reader.join(timeout=KILL_GRACE_S)
                proc.stdout.close()
        if code != 0:
            last = next((line for line in reversed(tail) if line.strip()), "")
            detail = f": {last.strip()[:300]}" if last else ""
            raise StepError(
                f"'{tool}' exited with code {code}{detail} "
                f"(see {log_path.relative_to(self.ws.root).as_posix()})"
            )

    def _wait(self, proc: subprocess.Popen[str], tool: str) -> int:
        while True:
            try:
                return proc.wait(timeout=POLL_S)
            except subprocess.TimeoutExpired:
                if self.cancelled:
                    self.logger.warning("[%s] cancelling '%s'", self.step_name, tool)
                    terminate_tree(proc)
                    raise StepCancelled(f"'{tool}' cancelled") from None


def terminate_tree(proc: subprocess.Popen[Any], grace_s: float = KILL_GRACE_S) -> None:
    """Terminate a process and all its descendants (nnU-Net and torch spawn workers)."""
    if proc.poll() is not None:
        return
    try:
        root = psutil.Process(proc.pid)
        procs = [*root.children(recursive=True), root]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(procs, timeout=grace_s)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(alive, timeout=grace_s)


def _pump(stream: IO[str], log: IO[str], logger: logging.Logger, tail: deque[str]) -> None:
    for line in stream:
        log.write(line)
        log.flush()
        tail.append(line)
        logger.debug("%s", line.rstrip())


class Step(ABC):
    """One pipeline step. ``name`` must be a key of ``workspace.STEP_DIRS``."""

    name: ClassVar[str]
    version: ClassVar[str] = "1"  # bump when a code change alters outputs
    config_sections: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def run(self, ctx: StepContext) -> None: ...

    def fingerprint_extra(self, config: PipelineConfig) -> Mapping[str, Any]:
        """Extra identity mixed into the cache key, e.g. the selected plugin and its version."""
        return {}
