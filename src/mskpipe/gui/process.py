# SPDX-License-Identifier: Apache-2.0
"""The pipeline as a child process of the GUI.

The GUI never runs the pipeline in its own process: ``python -m mskpipe run ... --events
jsonl --cancel-on-stdin`` is started with :class:`QProcess`, so that

* the resources measured for the manifest belong to the pipeline only (the sampler
  measures the whole process tree of the run),
* a crash of a native library (VTK, torch) does not take the window down,
* cancelling is clean: ``cancel`` on stdin stops the run between steps or terminates the
  running tool with all its children, and the run is recorded as ``interrupted``.
  If the child does not exit within :data:`CANCEL_GRACE_MS`, its whole process tree is
  killed and the manifest is finalised by :func:`mskpipe.api.finalize_stale`.

stdout carries one :class:`~mskpipe.api.RunEvent` JSON object per line; any other line
(stdout or stderr) is passed on as text.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import psutil
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from mskpipe.api import RunEvent

__all__ = [
    "CANCEL_GRACE_MS",
    "ChildRequest",
    "PipelineProcess",
    "child_command",
    "kill_tree",
    "mskpipe_command",
    "python_executable",
]

CANCEL_GRACE_MS = 30_000  # time for a clean cancel before the process tree is killed
EXIT_CANCELLED = 130


@dataclass(frozen=True)
class ChildRequest:
    """What the GUI asks the child ``mskpipe run`` to do."""

    image: Path
    modality: str
    runs_dir: Path
    device: str = "cpu"
    subject: str | None = None
    until_step: str | None = None
    overrides: Sequence[str] = field(default_factory=tuple)
    verbose: bool = False


def python_executable() -> str:
    """Interpreter for the child: ``python.exe`` next to ``pythonw.exe`` (GUI launcher)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        console = exe.with_name("python.exe")
        if console.is_file():
            return str(console)
    return str(exe)


def mskpipe_command() -> list[str]:
    """How the GUI starts the CLI: ``python -m mskpipe``."""
    return [python_executable(), "-m", "mskpipe"]


def child_command(request: ChildRequest, prefix: Sequence[str] | None = None) -> list[str]:
    """Command line of the child run (pre-run checks are done by the GUI beforehand).

    ``prefix`` replaces ``python -m mskpipe`` (tests start a CLI with fake steps).
    """
    argv = [*(prefix or mskpipe_command()), "run", str(request.image)]
    argv += ["--modality", request.modality, "--device", request.device]
    argv += ["--runs-dir", str(request.runs_dir)]
    if request.subject:
        argv += ["--subject", request.subject]
    if request.until_step:
        argv += ["--until", request.until_step]
    for item in request.overrides:
        argv += ["--set", item]
    argv += ["--no-preflight", "--events", "jsonl", "--cancel-on-stdin"]
    if request.verbose:
        argv.append("--verbose")
    return argv


def kill_tree(pid: int, timeout_s: float = 5.0) -> None:
    """Kill a process and all its descendants (QProcess.kill() reaches the child only)."""
    try:
        root = psutil.Process(pid)
        procs = [*root.children(recursive=True), root]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(procs, timeout=timeout_s)


class PipelineProcess(QObject):
    """One child run. Signals are delivered in the GUI thread."""

    event = Signal(object)  # RunEvent
    text = Signal(str, str)  # (line, "stdout" | "stderr") that is not an event
    finished = Signal(int, bool)  # (exit code, killed after a cancel timeout)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc = QProcess(self)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._finished)
        self._proc.errorOccurred.connect(self._error)
        self._buffers = {"stdout": b"", "stderr": b""}
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self.kill)
        self._killed = False
        self._done = False
        self.run_dir: Path | None = None
        self.cancel_grace_ms = CANCEL_GRACE_MS

    # ------------------------------------------------------------------ control

    def start(self, argv: Sequence[str]) -> None:
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self._proc.setProcessEnvironment(env)
        self._killed = self._done = False
        self.run_dir = None
        self._proc.start(argv[0], list(argv[1:]))

    def is_running(self) -> bool:
        return self._proc.state() != QProcess.ProcessState.NotRunning

    def cancel(self) -> None:
        """Ask the run to stop; kill the process tree if it does not exit in time."""
        if not self.is_running():
            return
        self._proc.write(b"cancel\n")
        self._kill_timer.start(self.cancel_grace_ms)

    def kill(self) -> None:
        if not self.is_running():
            return
        self._killed = True
        pid = self._proc.processId()
        if pid:
            kill_tree(pid)
        self._proc.kill()

    def wait(self, msecs: int = 30_000) -> bool:
        return self._proc.waitForFinished(msecs)

    # ------------------------------------------------------------------ output

    def _read_stdout(self) -> None:
        self._consume("stdout", bytes(self._proc.readAllStandardOutput()))

    def _read_stderr(self) -> None:
        self._consume("stderr", bytes(self._proc.readAllStandardError()))

    def _consume(self, channel: str, data: bytes, final: bool = False) -> None:
        buf = self._buffers[channel] + data
        *lines, rest = buf.split(b"\n")
        if final and rest:
            lines, rest = [*lines, rest], b""
        self._buffers[channel] = rest
        for raw in lines:
            line = raw.decode("utf-8", errors="replace").rstrip("\r")
            if not line.strip():
                continue
            if channel == "stdout" and line.startswith("{"):
                try:
                    event = RunEvent.from_line(line)
                except ValueError:
                    pass
                else:
                    if event.kind == "run_started" and event.run_dir:
                        self.run_dir = Path(event.run_dir)
                    self.event.emit(event)
                    continue
            self.text.emit(line, channel)

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if self._done:
            return
        self._done = True
        self._kill_timer.stop()
        self._consume("stdout", bytes(self._proc.readAllStandardOutput()), final=True)
        self._consume("stderr", bytes(self._proc.readAllStandardError()), final=True)
        if status == QProcess.ExitStatus.CrashExit and not self._killed:
            self.text.emit(f"The pipeline process crashed (exit code {code})", "stderr")
        self.finished.emit(code, self._killed)

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._done = True
            self.text.emit(f"Cannot start the pipeline: {self._proc.errorString()}", "stderr")
            self.finished.emit(-1, False)
