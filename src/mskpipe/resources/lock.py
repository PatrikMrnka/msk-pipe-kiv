# SPDX-License-Identifier: Apache-2.0
"""Exclusive lock between msk-pipe processes (GUI windows, CLI, runs) for work that
changes shared per-user data: downloading model weights, installing or removing GPU support,
installing an atlas, storing the licence key.

Without it two GUI windows, or a GUI and a ``mskpipe setup`` in a console, download the
same weights into the same files, or replace GPU support while a run uses it.

The lock is an OS file lock on ``<MSKPIPE_HOME>/locks/<name>.lock`` (``msvcrt`` on Windows,
``fcntl`` elsewhere): the operating system releases it when the holder exits, also when it
crashes or is killed, so a lock can never be left behind. Who holds it is written to
``<name>.json`` next to it (the locked file itself cannot be read on Windows).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from mskpipe.resources.paths import home

SETUP_LOCK = "setup"


class BusyError(RuntimeError):
    """Another msk-pipe process holds the lock."""

    def __init__(self, message: str, holder: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.holder = holder or {}


class LockCancelled(RuntimeError):
    """Waiting for the lock was cancelled."""


def lock_dir() -> Path:
    return home() / "locks"


def _try_lock(f: IO[bytes]) -> bool:
    if sys.platform == "win32":
        import msvcrt

        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(f: IO[bytes]) -> None:
    if sys.platform == "win32":
        import msvcrt

        f.seek(0)
        with contextlib.suppress(OSError):
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class Lock:
    """``with Lock("setup", "downloading weights"):`` - raises :class:`BusyError` when
    another process holds it (``wait_s`` > 0: wait that long first)."""

    def __init__(self, name: str = SETUP_LOCK, purpose: str = "") -> None:
        self.name = name
        self.purpose = purpose
        self.path = lock_dir() / f"{name}.lock"
        self.info_path = lock_dir() / f"{name}.json"
        self._file: IO[bytes] | None = None

    @property
    def held(self) -> bool:
        return self._file is not None

    def holder(self) -> dict[str, Any]:
        """What the current holder wrote (pid, purpose, time); empty if unknown."""
        try:
            data = json.loads(self.info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def describe_holder(self) -> str:
        info = self.holder()
        what = info.get("purpose") or "another msk-pipe task"
        pid = f" (process {info['pid']})" if info.get("pid") else ""
        return f"{what}{pid}"

    def acquire(
        self,
        *,
        wait_s: float = 0.0,
        poll_s: float = 1.0,
        cancelled: Callable[[], bool] | None = None,
        on_wait: Callable[[str], None] | None = None,
    ) -> Lock:
        if self._file is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = self.path.open("a+b")
        deadline = time.monotonic() + wait_s
        announced = False
        while not _try_lock(f):
            if cancelled is not None and cancelled():
                f.close()
                raise LockCancelled(f"Waiting for {self.describe_holder()} was cancelled")
            if wait_s <= 0 or time.monotonic() >= deadline:
                holder = self.holder()
                f.close()
                raise BusyError(
                    f"Another msk-pipe task is running: {self.describe_holder()}. "
                    "Wait until it finishes (or cancel it there) and try again.",
                    holder,
                )
            if not announced and on_wait is not None:
                on_wait(f"Waiting for {self.describe_holder()} to finish …")
                announced = True
            time.sleep(poll_s)
        self._file = f
        info = {"pid": os.getpid(), "purpose": self.purpose, "since": time.time()}
        with contextlib.suppress(OSError):
            self.info_path.write_text(json.dumps(info), encoding="utf-8")
        return self

    def release(self) -> None:
        f, self._file = self._file, None
        if f is None:
            return
        with contextlib.suppress(OSError):
            self.info_path.unlink(missing_ok=True)
        _unlock(f)
        f.close()

    def __enter__(self) -> Lock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()


def setup_lock(purpose: str) -> Lock:
    return Lock(SETUP_LOCK, purpose)
