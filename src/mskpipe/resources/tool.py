# SPDX-License-Identifier: Apache-2.0
"""Run a helper command (weights download, licence) and stream its output."""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
from collections.abc import Callable, Sequence
from typing import IO

Log = Callable[[str], None]
Cancelled = Callable[[], bool]

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_ENV = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
POLL_S = 0.5


class ToolCancelled(RuntimeError):
    """The helper command was cancelled (its process tree was terminated)."""


def run_tool(argv: Sequence[str], log: Log, *, cancelled: Cancelled | None = None) -> int:
    """Run ``argv`` with stdin closed and no console window; every output line goes to
    ``log``. Returns the exit code (``127`` if the program cannot be started). If
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
    reader = threading.Thread(target=_pump, args=(proc.stdout, log), daemon=True)
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
