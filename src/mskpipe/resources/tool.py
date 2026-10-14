# SPDX-License-Identifier: Apache-2.0
"""Run a helper command (weights download, licence) and stream its output."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import threading
from collections.abc import Callable, Sequence
from typing import IO

Log = Callable[[str], None]
Cancelled = Callable[[], bool]

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_ENV = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
POLL_S = 0.5


_BAR = re.compile(r"(?P<pct>\d{1,3})%\|")  # tqdm: "Downloading:  35%|███  | 81M/232M [...]"
PROGRESS_STEP = 10


class ProgressFilter:
    """Passes a tool's output on, but of a tqdm progress bar only one line per
    ``step`` percent (and 100 %). tqdm redraws its bar with '\\r', which reading the output
    as text turns into one line per redraw: hundreds of log lines per download."""

    def __init__(self, log: Log, step: int = PROGRESS_STEP) -> None:
        self.log = log
        self.step = step
        self._last: int | None = None

    def __call__(self, text: str) -> None:
        match = _BAR.search(text)
        if match is None:
            self.log(text)
            return
        pct = min(100, int(match["pct"]))
        bucket = pct // self.step
        if bucket != self._last:  # first line of a bar, the next step or a new bar
            self._last = bucket
            self.log(text)


class ToolCancelled(RuntimeError):
    """The helper command was cancelled (its process tree was terminated)."""


def run_tool(argv: Sequence[str], log: Log, *, cancelled: Cancelled | None = None) -> int:
    """Run ``argv`` with stdin closed and no console window; every output line goes to
    ``log`` (progress bars thinned out, see :class:`ProgressFilter`). Returns the exit
    code (``127`` if the program cannot be started). If
    ``cancelled()`` becomes true, the command and its children are terminated and
    :class:`ToolCancelled` is raised."""
    try:
        proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, **_ENV},
            creationflags=_NO_WINDOW,
        )
    except OSError as exc:
        log(f"Cannot start {argv[0]}: {exc}")
        return 127
    assert proc.stdout is not None
    reader = threading.Thread(target=_pump, args=(proc.stdout, ProgressFilter(log)), daemon=True)
    reader.start()
    try:
        while True:
            try:
                code = proc.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                if cancelled is not None and cancelled():
                    from mskpipe.core.step import terminate_tree

                    terminate_tree(proc)
                    raise ToolCancelled(f"{argv[0]} cancelled") from None
    finally:
        reader.join(timeout=5)
        proc.stdout.close()
    return code


def _pump(stream: IO[str], log: Log) -> None:
    """Read to the end whatever happens (a full pipe would block the command)."""
    try:
        for line in stream:
            text = line.rstrip("\r\n")
            if text.strip():
                with contextlib.suppress(OSError, ValueError):
                    log(text)
    except (OSError, ValueError):
        return
