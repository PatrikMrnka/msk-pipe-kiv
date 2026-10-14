# SPDX-License-Identifier: Apache-2.0
"""Setup lock between msk-pipe processes (mskpipe.resources.lock) and its users: CLI
`mskpipe setup`, weight downloads in the segment step, GPU support in use."""

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mskpipe.cli import EXIT_BUSY, app
from mskpipe.core.step import StepCancelled
from mskpipe.resources import gpu, lock
from mskpipe.steps import segment

runner = CliRunner()

HOLDER = textwrap.dedent(
    """
    import sys, time
    from mskpipe.resources.lock import setup_lock
    held = setup_lock("downloading in another window").acquire()
    print("locked", flush=True)
    time.sleep(120)
    """
)


@pytest.fixture
def other_process():
    """A real second process holding the setup lock (same MSKPIPE_HOME)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER], stdout=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL
    )
    assert proc.stdout.readline().strip() == "locked"
    yield proc
    proc.kill()
    proc.wait(10)


def test_lock_is_exclusive_and_reports_holder():
    first = lock.setup_lock("installing GPU support").acquire()
    try:
        with pytest.raises(lock.BusyError, match="installing GPU support") as info:
            lock.setup_lock("downloading model weights").acquire()
        assert info.value.holder["pid"] > 0
    finally:
        first.release()
    assert not first.held and not first.info_path.exists()
    with lock.setup_lock("again"):  # free again
        pass


def test_lock_held_by_another_process_and_released_when_it_dies(other_process):
    with pytest.raises(lock.BusyError, match=r"another window \(process \d+\)"):
        lock.setup_lock("here").acquire()
    other_process.kill()  # crash/kill: the OS releases the lock, nothing stays behind
    other_process.wait(10)
    with lock.setup_lock("here"):
        pass


def test_wait_cancel_and_timeout():
    held = lock.setup_lock("holder").acquire()
    try:
        messages = []
        start = time.monotonic()
        with pytest.raises(lock.BusyError):
            lock.setup_lock("x").acquire(wait_s=0.3, poll_s=0.05, on_wait=messages.append)
        assert time.monotonic() - start >= 0.3
        pid = held.holder()["pid"]
        assert messages == [f"Waiting for holder (process {pid}) to finish …"]
        with pytest.raises(lock.LockCancelled):
            lock.setup_lock("x").acquire(wait_s=60, poll_s=0.05, cancelled=lambda: True)
    finally:
        held.release()


def test_waiting_succeeds_when_released():
    held = lock.setup_lock("holder").acquire()
    threading.Timer(0.3, held.release).start()
    waiter = lock.setup_lock("waiter").acquire(wait_s=10, poll_s=0.05)
    assert waiter.held
    waiter.release()


def test_cli_setup_refuses_while_locked(tmp_path, other_process):
    (tmp_path / "atlas.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["setup", "atlas", str(tmp_path)])
    assert result.exit_code == EXIT_BUSY
    assert "another window" in result.output
    for args in (["setup", "gpu", "--remove"], ["setup", "licence", "aca_ABCDEFGHIJKLMN"]):
        assert runner.invoke(app, args).exit_code == EXIT_BUSY


# ---------------------------------------------------------------- segment step


def _ctx(tmp_path, cancelled=False):
    def check_cancel():
        if cancelled:
            raise StepCancelled("cancelled")

    logs = []
    logger = SimpleNamespace(
        info=lambda *a: logs.append(a[0] % a[1:]), warning=lambda *a: logs.append(a[0] % a[1:])
    )
    ctx = SimpleNamespace(
        config=None,
        ws=SimpleNamespace(root=tmp_path / "20261014_run"),
        cancelled=cancelled,
        check_cancel=check_cancel,
        logger=logger,
    )
    return ctx, logs


def test_segment_holds_lock_only_while_downloading(tmp_path, monkeypatch):
    from mskpipe.resources import weights

    monkeypatch.setattr(weights, "tool_weights_missing", lambda c, m, t: False)
    ctx, _ = _ctx(tmp_path)
    assert segment.download_lock(ctx, "musclemap", "ct") is None
    monkeypatch.setattr(weights, "tool_weights_missing", lambda c, m, t: True)
    held = segment.download_lock(ctx, "musclemap", "ct")
    try:
        assert held.held
        with pytest.raises(lock.BusyError, match="downloading musclemap weights"):
            lock.setup_lock("setup gpu").acquire()
    finally:
        held.release()


def test_segment_waits_for_another_download_and_can_be_cancelled(tmp_path, monkeypatch):
    from mskpipe.resources import weights

    monkeypatch.setattr(weights, "tool_weights_missing", lambda c, m, t: True)
    other = lock.setup_lock("mskpipe setup weights").acquire()
    try:
        ctx, _ = _ctx(tmp_path, cancelled=True)
        with pytest.raises(StepCancelled):
            segment.download_lock(ctx, "totalsegmentator", "ct")
    finally:
        other.release()


def test_segment_ignores_unknown_weight_status(tmp_path, monkeypatch):
    from mskpipe.resources import weights

    def broken(*args):
        raise OSError("cannot read")

    monkeypatch.setattr(weights, "tool_weights_missing", broken)
    ctx, _ = _ctx(tmp_path)
    assert segment.download_lock(ctx, "musclemap", "ct") is None


# ---------------------------------------------------------------- GPU support in use


def test_remove_gpu_in_use_leaves_overlay_intact(monkeypatch):
    folder = gpu.overlay_dir()
    (folder / gpu.SITE / "torch").mkdir(parents=True)
    (folder / gpu.MARKER).write_text("{}", encoding="utf-8")
    real_replace = Path.replace

    def locked(self, target):  # Windows: a loaded DLL blocks renaming the folder
        if self == folder:
            raise PermissionError(5, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", locked)
    with pytest.raises(gpu.GpuSetupError, match="in use"):
        gpu.remove_gpu(print)
    assert (folder / gpu.SITE / "torch").is_dir() and (folder / gpu.MARKER).is_file()
    monkeypatch.setattr(Path, "replace", real_replace)
    assert gpu.remove_gpu(print) and not folder.exists()
