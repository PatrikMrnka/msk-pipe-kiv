# SPDX-License-Identifier: Apache-2.0
"""GUI with a DICOM folder as input (Qt offscreen, real dcm2niix, fake pipeline steps)."""

import html
import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from dicom_data import ct_folder, write_series
from fake_pipeline import write_nifti
from PySide6.QtCore import QEventLoop, QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

from mskpipe import api
from mskpipe.gui import app as gui_app
from mskpipe.gui.process import ChildRequest, child_command

FAKE_CLI = [sys.executable, str(Path(__file__).with_name("fake_cli.py"))]
TIMEOUT_S = 60


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch):
    calls = []
    for name in ("critical", "warning", "information"):
        monkeypatch.setattr(
            QMessageBox,
            name,
            staticmethod(lambda *a, _n=name, **k: calls.append((_n, a)) or QMessageBox.Ok),
        )
    return calls


def wait_until(condition, timeout_s: float = TIMEOUT_S) -> None:
    end = time.monotonic() + timeout_s
    while not condition():
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if time.monotonic() > end:
            raise TimeoutError("condition not met")
        time.sleep(0.01)


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "preflight", lambda prepared, registry=None: [])
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    win = gui_app.MainWindow(settings, command=FAKE_CLI, probe_device=False)
    win.runs_edit.setText(str(tmp_path / "runs"))
    gui_app._select(win.device_combo, "cpu")
    gui_app._select(win.modality_combo, "auto")
    yield win
    if win.process.is_running():
        win.process.kill()
        win.process.wait(5000)
    win._pool.waitForDone(30_000)
    win.deleteLater()


def read(win) -> None:
    """Wait until the DICOM folder has been read in the background."""
    wait_until(lambda: win.dicom_scan() is not None)


def series_row_visible(win) -> bool:
    return win._input_grid.isRowVisible(win.series_row)


def test_format_switch_keeps_one_path_per_format(window, tmp_path):
    nifti = write_nifti(tmp_path / "s01.nii.gz")
    folder = write_series(tmp_path / "dicom", "CT")
    window.set_input(nifti, "nifti")
    assert window.image_edit.placeholderText() == gui_app.PLACEHOLDERS["nifti"]
    window.set_input(folder, "dicom")
    assert window.image_edit.placeholderText() == gui_app.PLACEHOLDERS["dicom"]
    window.set_format("nifti")
    assert window.image_edit.text() == str(nifti)
    window.set_format("dicom")
    assert window.image_edit.text() == str(folder)


def test_dicom_folder_is_checked(window, tmp_path):
    folder = ct_folder(tmp_path / "ct")
    window.set_input(folder, "dicom")
    read(window)
    info = window.image_info.text()
    assert "Series 3: 20 x 16 x 12 voxels" in info and "12 DICOM files" in info
    assert "Not usable" in info and "localizer" in info
    assert window.current_dicom().id == "3"
    assert series_row_visible(window) and not window.series_combo.isEnabled()  # one choice
    assert "detected CT (dicom" in window.modality_label.text()
    assert window.view_image_button.isEnabled()
    assert window.subject_edit.placeholderText() == "ct"


def test_several_series_are_offered(window, tmp_path):
    folder = ct_folder(tmp_path / "ct")
    write_series(folder, "MR", number=5, slices=8, description="t1")
    window.set_input(folder, "dicom")
    read(window)
    assert window.series_combo.isEnabled()
    assert [window.series_combo.itemData(i) for i in range(window.series_combo.count())] == [
        "3",
        "5",
    ]
    assert "2 usable series" in window.image_info.text()
    gui_app._select(window.series_combo, "5")
    assert window.current_dicom().id == "5"
    assert window.effective_modality() == "mri"
    assert "Series 5: 20 x 16 x 8 voxels" in window.image_info.text()


@pytest.mark.parametrize(
    ("make", "fmt", "message"),
    [
        (lambda p: p / "missing", "dicom", "Folder not found"),
        (lambda p: write_nifti(p / "s.nii.gz"), "dicom", "switch to 'NIfTI file'"),
        (lambda p: next(write_series(p / "d", "CT", slices=3).glob("*.dcm")), "dicom", "folder"),
        (lambda p: write_series(p / "d", "CT", slices=3), "nifti", "switch to 'DICOM folder'"),
        (lambda p: next(write_series(p / "d", "CT", slices=3).glob("*.dcm")), "nifti", "DICOM"),
        (lambda p: p / "missing.nii.gz", "nifti", "File not found"),
    ],
)
def test_wrong_input_for_the_format(window, tmp_path, make, fmt, message):
    window.set_input(make(tmp_path), fmt)
    assert message in html.unescape(window.image_info.text())
    assert not window.view_image_button.isEnabled()
    assert not series_row_visible(window)


def test_folder_without_usable_series(window, tmp_path, _no_dialogs):
    window.set_input(write_series(tmp_path / "pet", "PT", number=2, slices=4), "dicom")
    read(window)
    assert "No usable 3D CT or MR series" in window.image_info.text()
    assert "only CT and MR" in window.image_info.text()
    assert window.start_run() is False and _no_dialogs[-1][0] == "warning"
    (tmp_path / "text").mkdir()
    (tmp_path / "text" / "a.txt").write_text("x", encoding="utf-8")
    window.set_input(tmp_path / "text", "dicom")
    read(window)
    assert "No DICOM images" in window.image_info.text()
    assert window.start_run() is False


def test_run_from_dicom_folder(window, tmp_path):
    folder = ct_folder(tmp_path / "ct")
    write_series(folder, "CT", number=5, slices=8, description="Lung")
    window.set_input(folder, "dicom")
    read(window)
    gui_app._select(window.series_combo, "5")
    child = window.child_request()
    assert (child.image, child.series, child.modality) == (folder, "5", "ct")
    assert window.start_run()
    wait_until(lambda: not window.process.is_running() and window.summary is not None)
    assert window.summary.status == "completed", window.log_view.toPlainText()
    assert "DICOM: Series 5 'Lung'" in window.log_view.toPlainText()


def test_run_waits_for_the_folder(window, tmp_path):
    """Run right after choosing the folder: it is read synchronously."""
    window._pool.waitForDone()
    folder = write_series(tmp_path / "ct", "CT")
    window.set_format("dicom")
    window.image_edit.setText(str(folder))  # no editingFinished: not read yet
    assert window.child_request().series == "3"


def test_settings_remember_format_folder_and_series(qapp, tmp_path):
    folder = ct_folder(tmp_path / "ct")
    write_series(folder, "CT", number=5, slices=8)
    path = str(tmp_path / "gui.ini")
    first = gui_app.MainWindow(
        QSettings(path, QSettings.Format.IniFormat), command=FAKE_CLI, probe_device=False
    )
    first.set_input(folder, "dicom")
    read(first)
    gui_app._select(first.series_combo, "5")
    first._store()
    first._pool.waitForDone()
    second = gui_app.MainWindow(
        QSettings(path, QSettings.Format.IniFormat), command=FAKE_CLI, probe_device=False
    )
    read(second)
    assert second.input_format() == "dicom" and second.image_edit.text() == str(folder)
    assert second.current_dicom().id == "5"
    second._pool.waitForDone()


def test_child_command_passes_the_series():
    argv = child_command(
        ChildRequest(image=Path("d"), modality="ct", runs_dir=Path("r"), series="9_e2"),
        prefix=["mskpipe"],
    )
    assert argv[argv.index("--series") + 1] == "9_e2"
