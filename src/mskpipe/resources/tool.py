# SPDX-License-Identifier: Apache-2.0
"""Run a helper command (weights download, licence) and stream its output."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence

Log = Callable[[str], None]

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_ENV = {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def run_tool(argv: Sequence[str], log: Log) -> int:
    """Run ``argv`` with stdin closed and no console window; every output line goes to
    ``log``. Returns the exit code (``127`` if the program cannot be started)."""
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
    with proc.stdout:
        for line in proc.stdout:
            text = line.rstrip("\r\n")
            if text.strip():
                log(text)
    return proc.wait()
