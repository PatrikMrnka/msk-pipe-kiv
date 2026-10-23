# SPDX-License-Identifier: Apache-2.0
"""Setup window (licence key, atlas, weights, GPU support): Qt offscreen, actions run as
real child processes of the mskpipe CLI with an isolated MSKPIPE_HOME (conftest)."""

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEventLoop, QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

from mskpipe.config import load_config
from mskpipe.gui import app as gui_app
from mskpipe.gui import setup_dialog as sd
from mskpipe.resources import atlas, gpu
from mskpipe.resources.status import collect

CLI = [sys.executable, "-m", "mskpipe"]
TIMEOUT_S = 60


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name in ("critical", "warning", "information"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.delenv(atlas.ATLAS_ENV, raising=False)
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path / "ts"))  # no licence key


def wait_until(condition, timeout_s: float = TIMEOUT_S) -> None:
    end = time.monotonic() + timeout_s
    while not condition():
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if time.monotonic() > end:
            raise TimeoutError("condition not met")
        time.sleep(0.01)


def make_atlas(folder: Path) -> Path:
    (folder / "bones").mkdir(parents=True)
    (folder / "atlas.json").write_text('{"muscles": {}}', encoding="utf-8")
    return folder


def status_fn(**overrides):
    config = load_config(None, [f"{k}={v}" for k, v in overrides.items()])
    return lambda: collect(config, ["ct"])


def dialog(qapp, fn=None, **kwargs) -> sd.SetupDialog:
    return sd.SetupDialog(fn or status_fn(), command=CLI, modalities=("ct",), **kwargs)


# ---------------------------------------------------------------------------- helpers


def test_parse_progress():
    assert sd.parse_progress("  torch:  35 % of 3.05 GiB") == ("torch", 35)
    assert sd.parse_progress("  torchvision: 100 % of 0.01 GiB") == ("torchvision", 100)
    assert sd.parse_progress("Downloading torch-2.7.1") is None


def test_weights_args():
    assert sd.weights_args(["ct"], []) == ["setup", "weights", "--modality", "ct"]
    assert sd.weights_args(["ct", "mri"], ["a.b=1"]) == [
        "setup", "weights", "--modality", "all", "--set", "a.b=1",
    ]  # fmt: skip


def test_needs_attention():
    status = status_fn()()
    assert sd.needs_attention(status) == ["TotalSegmentator licence key", "attachment atlas"]
    status = status_fn(**{"segmentation.bones": "musclemap"})()
    assert sd.needs_attention(status) == ["attachment atlas"]


# ---------------------------------------------------------------------------- dialog


def test_dialog_shows_status(qapp):
    dlg = dialog(qapp)
    assert "not set" in dlg.licence_label.text()
    assert "missing" in dlg.atlas_label.text()
    assert dlg.musclemap_button.isEnabled()
    if not gpu.platform_supported():  # Linux CI: GPU support only in the Windows release
        assert not dlg.gpu_install.isEnabled()
        assert "Windows release" in dlg.gpu_label.text()
    assert not dlg.at_start_check.isVisible()
    dlg.deleteLater()


def test_invalid_key_is_rejected_locally(qapp):
    dlg = dialog(qapp)
    dlg.key_edit.setText("wrong")
    assert not dlg.save_licence()
    assert "must start with" in dlg.log_view.toPlainText()
    assert not dlg.is_running()
    dlg.deleteLater()


def test_install_atlas_in_child(qapp, tmp_path):
    dlg = dialog(qapp)
    source = make_atlas(tmp_path / "src")
    assert dlg.install_atlas(source)
    assert dlg.is_running() and not dlg.atlas_button.isEnabled()
    assert not dlg.install_atlas(source)  # one task at a time
    wait_until(lambda: not dlg.is_running())
    assert dlg.last_exit == 0, dlg.log_view.toPlainText()
    assert (atlas.default_atlas_dir() / "atlas.json").is_file()
    assert "ok" in dlg.atlas_label.text() and "installed" in dlg.atlas_label.text()
    # a second install replaces the first one (--force)
    assert dlg.install_atlas(source)
    wait_until(lambda: not dlg.is_running())
    assert dlg.last_exit == 0, dlg.log_view.toPlainText()
    dlg.deleteLater()


def test_failed_task_is_reported(qapp, tmp_path):
    dlg = dialog(qapp)
    dlg.install_atlas(tmp_path / "missing.zip")
    wait_until(lambda: not dlg.is_running())
    assert dlg.last_exit == 1
    text = dlg.log_view.toPlainText()
    assert "does not exist" in text and "failed (exit code 1)" in text
    dlg.deleteLater()


def test_gpu_signal_and_progress(qapp, tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(
        "print('Downloading torch')\nprint('  torch:  40 % of 3.05 GiB')\n", encoding="utf-8"
    )
    dlg = sd.SetupDialog(status_fn(), command=[sys.executable, str(script)])
    fired = []
    dlg.gpu_changed.connect(lambda: fired.append(1))
    dlg.run_task("Installing GPU support", [], after=dlg.gpu_changed.emit)
    wait_until(lambda: not dlg.is_running())
    assert fired == [1] and dlg.last_exit == 0
    assert "Downloading torch" in dlg.log_view.toPlainText()
    assert "40 %" not in dlg.log_view.toPlainText()  # progress lines go to the bar
    dlg.deleteLater()


def test_cancel_kills_child(qapp, tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\nprint('start', flush=True)\ntime.sleep(60)\n", "utf-8")
    dlg = sd.SetupDialog(status_fn(), command=[sys.executable, str(script)])
    dlg.run_task("Slow", [])
    wait_until(lambda: "start" in dlg.log_view.toPlainText())
    dlg.cancel()
    wait_until(lambda: not dlg.is_running(), 15)
    assert dlg.last_exit == sd.CANCELLED
    assert "Slow: cancelled" in dlg.log_view.toPlainText()
    dlg.deleteLater()


def test_missing_program(qapp, tmp_path):
    dlg = sd.SetupDialog(status_fn(), command=[str(tmp_path / "nope.exe")])
    dlg.run_task("Broken", ["x"])
    wait_until(lambda: not dlg.is_running(), 15)
    assert dlg.last_exit == 127
    dlg.deleteLater()


def test_broken_status_does_not_crash(qapp):
    def broken():
        raise ValueError("bad settings")

    dlg = sd.SetupDialog(broken, command=CLI)
    assert dlg.status is None
    assert "bad settings" in dlg.log_view.toPlainText()
    assert not dlg.weights_button.isEnabled()
    dlg.deleteLater()


# ---------------------------------------------------------------------------- main window


@pytest.fixture
def window(qapp, tmp_path):
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    win = gui_app.MainWindow(settings, command=CLI, probe_device=False)
    gui_app._select(win.modality_combo, "ct")
    yield win
    if win.setup_dialog is not None:
        win.setup_dialog.cancel()
    win.deleteLater()


def test_window_opens_setup_at_start(window, tmp_path):
    dlg = window.check_setup()
    assert dlg is not None and dlg.isVisible() and dlg.at_start_check.isVisible()
    assert window.open_setup() is dlg  # one instance
    assert dlg.modalities == ("ct",)
    dlg.at_start_check.setChecked(False)
    dlg.accept()
    assert window.settings.value("setup_at_start") == "false"
    assert window.check_setup() is None


def test_window_nothing_missing(window, tmp_path):
    atlas.install_atlas(str(make_atlas(tmp_path / "a")), print)
    window.apply_setting(("segmentation", "bones"), "musclemap")
    assert window.check_setup() is None


def test_use_musclemap_changes_form(window):
    dlg = window.open_setup()
    assert dlg.musclemap_button.isEnabled()
    dlg.musclemap_button.click()
    assert window.form.values()["segmentation"]["bones"] == "musclemap"
    assert "segmentation.bones=musclemap" in window.form.overrides()
    assert "not needed" in dlg.licence_label.text()
    assert not dlg.musclemap_button.isEnabled()
    dlg.accept()


def test_device_label_links_to_setup(window, monkeypatch):
    out = (
        '{"requested": "auto", "selected": "cpu", "reason": "auto: torch.cuda is not '
        'available (CPU-only build 2.7.1+cpu; GPU support not installed (mskpipe setup gpu))"}'
    )

    class FakeProc:
        def readAllStandardOutput(self):
            return out.encode()

        def readAllStandardError(self):
            return b""

    monkeypatch.setattr(window, "_device_proc", FakeProc())
    window._device_detected(0, None)
    assert 'href="setup"' in window.device_label.text()
    window.device_label.linkActivated.emit("setup")
    assert window.setup_dialog is not None and window.setup_dialog.isVisible()
    window.setup_dialog.accept()


# ---------------------------------------------------------------------------- one task at a time

SLOW = "import time\nprint('downloading', flush=True)\ntime.sleep(60)\n"


def _slow_dialog(window, tmp_path):
    """The window's setup dialog with a slow fake 'download' as its command."""
    script = tmp_path / "slow.py"
    script.write_text(SLOW, encoding="utf-8")
    dlg = window.open_setup()
    dlg.command = [sys.executable, str(script)]
    return dlg


def test_close_during_download_keeps_one_task(window, tmp_path):
    """Reported: Download now -> Close -> Setup… again -> Download now started a second
    download (a new window each time); the closed one kept running."""
    dlg = _slow_dialog(window, tmp_path)
    assert dlg.run_task("Downloading model weights", [])
    wait_until(lambda: "downloading" in dlg.log_view.toPlainText())
    pid = dlg._proc.processId()
    dlg.accept()  # Close button
    assert dlg.is_running()  # keeps running in the background
    assert "continues in the background" in dlg.log_view.toPlainText()
    again = window.open_setup()
    assert again is dlg and again.isVisible()  # same window, shows the progress
    assert not again.run_task("Downloading model weights", [])  # no second download
    assert not again.weights_button.isEnabled() and not again.gpu_install.isEnabled()
    assert "is running" in again.log_view.toPlainText()
    again.reject()  # Esc / X behave the same
    assert again.is_running() and again._proc.processId() == pid
    again.cancel()
    wait_until(lambda: not again.is_running(), 15)
    assert again.last_exit == sd.CANCELLED


def test_run_refused_while_setup_task_runs(window, tmp_path, monkeypatch):
    dlg = _slow_dialog(window, tmp_path)
    dlg.run_task("Installing GPU support", [])
    wait_until(lambda: "downloading" in dlg.log_view.toPlainText())
    assert not window.run_button.isEnabled()
    assert "Installing GPU support" in window.statusBar().currentMessage()
    started = []
    monkeypatch.setattr(window.process, "start", lambda *a: started.append(a))
    assert window.start_run() is False and started == []
    dlg.cancel()
    wait_until(lambda: not dlg.is_running(), 15)
    assert window.statusBar().currentMessage() == ""


def test_setup_actions_blocked_while_run_active(window):
    dlg = window.open_setup()
    window._set_running(True)  # a pipeline run started
    assert dlg.blocked_label.isVisible() or not dlg.isVisible()
    assert "pipeline run" in dlg.blocked_label.text()
    for button in (dlg.weights_button, dlg.atlas_button, dlg.key_button, dlg.gpu_install):
        assert not button.isEnabled()
    assert not dlg.install_atlas("C:/atlas.zip")
    assert "A pipeline run is in progress" in dlg.log_view.toPlainText()
    window._set_running(False)
    assert not dlg.blocked_label.isVisible() and dlg.atlas_button.isEnabled()


def test_quitting_cancels_the_setup_task(window, tmp_path, monkeypatch):
    dlg = _slow_dialog(window, tmp_path)
    dlg.run_task("Downloading model weights", [])
    wait_until(lambda: "downloading" in dlg.log_view.toPlainText())
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: answers.pop(0)))
    assert window.close() is False and dlg.is_running()  # "No": keep running, stay open
    assert window.close() is True  # "Yes": the download is cancelled
    wait_until(lambda: not dlg.is_running(), 15)
    assert dlg.last_exit == sd.CANCELLED


def test_busy_exit_code_is_explained(qapp, tmp_path):
    script = tmp_path / "busy.py"
    script.write_text("raise SystemExit(3)\n", encoding="utf-8")
    dlg = sd.SetupDialog(status_fn(), command=[sys.executable, str(script)])
    dlg.run_task("Downloading model weights", [])
    wait_until(lambda: not dlg.is_running(), 15)
    assert "another msk-pipe window or console" in dlg.log_view.toPlainText()
    dlg.deleteLater()


# ---------------------------------------------------------------------------- Fast-RNRR


def test_parse_progress_of_fast_rnrr_build():
    assert sd.parse_progress("  compiled 3/30: Registration.cpp") == ("compiling", 10)
    assert sd.parse_progress("  compiled 30/30: tools.cpp") == ("compiling", 100)


def test_needs_attention_fast_rnrr(monkeypatch):
    monkeypatch.delenv("MSKPIPE_FAST_RNRR", raising=False)
    monkeypatch.setenv("PATH", "")
    status = status_fn(
        **{
            "segmentation.bones": "musclemap",
            "attachments.params.nonrigid": "fast_rnrr",
        }
    )()
    assert sd.needs_attention(status) == ["attachment atlas", "Fast-RNRR"]


def test_fast_rnrr_build_asks_first(qapp, monkeypatch):
    monkeypatch.delenv("MSKPIPE_FAST_RNRR", raising=False)
    dlg = dialog(qapp)
    assert "not built" in dlg.rnrr_label.text() or "not supported" in dlg.rnrr_label.text()
    started = []
    monkeypatch.setattr(dlg, "run_task", lambda title, args, **kw: started.append(args) or True)
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
    )
    assert not dlg.build_fast_rnrr()
    assert not started and "not confirmed" in dlg.log_view.toPlainText()
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    assert dlg.build_fast_rnrr()
    assert started == [["setup", "fast-rnrr", "--accept-research-licence"]]
    dlg.deleteLater()
