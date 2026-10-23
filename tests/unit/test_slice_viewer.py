# SPDX-License-Identifier: Apache-2.0
"""Axial/coronal/sagittal viewer of the input volume (Qt offscreen)."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

import nibabel as nib
from PySide6.QtCore import QEventLoop, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from mskpipe.gui import slice_viewer as sv

TIMEOUT_S = 30
SHAPE = (20, 16, 12)  # i, j, k of the file


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(condition, timeout_s: float = TIMEOUT_S) -> None:
    end = time.monotonic() + timeout_s
    while not condition():
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if time.monotonic() > end:
            raise TimeoutError("condition not met")
        time.sleep(0.01)


def marked(path, affine, marker_ijk=(18, 14, 10), **header):
    """Volume of zeros with one bright voxel; returns (path, world position of the voxel)."""
    data = np.zeros(SHAPE, np.int16)
    data[marker_ijk] = 1000
    img = nib.Nifti1Image(data, affine)
    for key, value in header.items():
        img.header[key] = value
    nib.save(img, str(path))
    return path, (affine @ np.r_[marker_ijk, 1.0])[:3]


RAS = np.diag([0.8, 1.0, 2.0, 1.0])
LPS = np.diag([-0.8, -1.0, 2.0, 1.0])  # DICOM-like: i to the left, j to the posterior


def bright(img2d) -> tuple[int, int]:
    row, col = np.unravel_index(np.argmax(img2d), img2d.shape)
    return int(row), int(col)


# ---------------------------------------------------------------------------- volume


@pytest.mark.parametrize("affine", [RAS, LPS], ids=["RAS", "LPS"])
def test_screen_orientation(tmp_path, affine):
    """A voxel right, anterior and superior of the centre is drawn top left in the axial
    and coronal view (radiological convention) and top left in the sagittal view
    (anterior on the left), whatever the voxel order of the file."""
    marker = (18, 14, 10) if affine is RAS else (1, 1, 10)  # right-anterior-superior
    path, world = marked(tmp_path / "v.nii.gz", affine, marker)
    vol = sv.load_volume(path)
    assert vol.shape == SHAPE and vol.spacing == pytest.approx((0.8, 1.0, 2.0))
    ijk = np.unravel_index(np.argmax(vol.data), vol.shape)
    assert vol.world(ijk) == pytest.approx(world)
    assert vol.file_index(ijk) == marker
    assert vol.value(ijk) == 1000
    for view in sv.VIEWS:
        s_axis = sv.AXES[view][0]
        img = sv.plane(vol.data, view, ijk[s_axis])
        row, col = bright(img)
        assert row < img.shape[0] / 2 and col < img.shape[1] / 2, view


def test_scaled_and_4d_volume(tmp_path):
    data = np.zeros((*SHAPE, 3), np.int16)
    data[5, 5, 5, 0] = 100
    img = nib.Nifti1Image(data, RAS)
    img.header.set_slope_inter(0.5, -10)
    path = tmp_path / "s.nii"
    nib.save(img, str(path))
    vol = sv.load_volume(path)
    assert vol.data.dtype == np.int16  # stored values, no float copy of the volume
    assert vol.value((5, 5, 5)) == pytest.approx(40.0)
    assert vol.value((0, 0, 0)) == pytest.approx(-10.0)
    assert any("3 volumes" in n for n in vol.notes)


def test_to_gray():
    out = sv.to_gray(np.array([[-10.0, 0.0, 5.0, 10.0, np.nan]]), 0.0, 10.0)
    assert out.dtype == np.uint8 and out.flags.c_contiguous
    assert out.tolist() == [[0, 0, 127, 255, 0]]


def test_oblique_is_noted(tmp_path):
    a = np.deg2rad(20)
    rot = np.array([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0], [0, 0, 1, 0]])
    affine = np.vstack([rot, [0, 0, 0, 1]])
    path, _ = marked(tmp_path / "o.nii.gz", affine)
    assert sv.load_volume(path).oblique


# ---------------------------------------------------------------------------- window


@pytest.fixture
def viewer(qapp, tmp_path):
    win = sv.SliceViewer()
    win.resize(900, 400)
    win.show()
    yield win
    win.close()
    QApplication.processEvents()
    win.deleteLater()


def opened(viewer, path):
    viewer.open(path)
    wait_until(lambda: not viewer.is_loading())
    return viewer


def test_viewer_shows_center_and_moves(viewer, tmp_path):
    path, _ = marked(tmp_path / "v.nii.gz", RAS)
    opened(viewer, path)
    assert viewer.volume is not None and viewer.error is None
    assert viewer.cursor == [10, 8, 6]
    assert viewer.sliders["axial"].maximum() == SHAPE[2] - 1
    assert "20 x 16 x 12 voxels" in viewer.file_label.text()
    assert "Voxel [10, 8, 6]" in viewer.status_label.text()

    # a click in the axial view moves the sagittal and coronal planes
    viewer.pick("axial", 0.2, 0.7)  # top-left pixel = right, anterior
    assert viewer.cursor[:2] == [SHAPE[0] - 1, SHAPE[1] - 1]
    viewer.scroll("axial", 3)
    assert viewer.cursor[2] == 9
    viewer.scroll("axial", 100)
    assert viewer.cursor[2] == SHAPE[2] - 1  # clamped
    viewer.sliders["sagittal"].setValue(4)
    assert viewer.cursor[0] == 4
    assert "value 0" in viewer.status_label.text()
    viewer.cursor = [18, 14, 10]
    viewer.refresh()
    assert "value 1000" in viewer.status_label.text()


def test_viewer_mouse(viewer, tmp_path):
    path, _ = marked(tmp_path / "v.nii.gz", RAS)
    opened(viewer, path)
    pane = viewer.panes["coronal"]
    rect = pane.image_rect()
    assert not rect.isEmpty()
    # physical proportions: coronal = 20 x 0.8 mm wide, 12 x 2 mm high
    assert rect.width() / rect.height() == pytest.approx(16 / 24, rel=0.02)
    pane.picked.emit("coronal", 0.5, 0.5)  # what a click in the top-left corner sends
    assert viewer.cursor[0] == SHAPE[0] - 1 and viewer.cursor[2] == SHAPE[2] - 1
    event = QWheelEvent(
        QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(0, -240),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )  # fmt: skip
    before = viewer.cursor[1]
    pane.wheelEvent(event)
    assert viewer.cursor[1] == before - 2
    viewer.toggle("coronal")
    assert not viewer._columns["axial"].isVisible() and viewer._columns["coronal"].isVisible()
    viewer.toggle("coronal")
    assert all(c.isVisible() for c in viewer._columns.values())


def test_viewer_window_presets(viewer, tmp_path):
    path, _ = marked(tmp_path / "v.nii.gz", RAS)
    opened(viewer, path)
    sv_index = {value: i for i, (_, value) in enumerate(sv.PRESETS) if isinstance(value, str)}
    viewer.preset_combo.setCurrentIndex(sv_index["full"])
    assert viewer.window() == pytest.approx((0.0, 1000.0), abs=0.1)
    viewer.preset_combo.setCurrentIndex(2)  # CT soft tissue
    assert viewer.window() == pytest.approx((-160.0, 240.0))
    viewer.drag_window(50, 0)  # wider
    assert viewer.width_spin.value() > 400
    assert viewer.preset_combo.currentData() == "custom"
    viewer.level_spin.setValue(10.0)
    assert viewer.window()[0] + viewer.width_spin.value() / 2 == pytest.approx(10.0)


def test_viewer_unreadable_and_reopen(viewer, tmp_path):
    bad = tmp_path / "bad.nii.gz"
    bad.write_bytes(b"not a nifti")
    opened(viewer, bad)
    assert viewer.volume is None and viewer.error
    path, _ = marked(tmp_path / "v.nii.gz", RAS)
    opened(viewer, path)
    assert viewer.volume is not None and viewer.error is None
    vol = viewer.volume
    viewer.open(path)  # same unchanged file: nothing is read again
    assert viewer.volume is vol and not viewer.is_loading()
    viewer.close()
    assert viewer.volume is None  # memory released


def test_stale_result_is_dropped(viewer, tmp_path):
    first, _ = marked(tmp_path / "a.nii.gz", RAS)
    second, _ = marked(tmp_path / "b.nii.gz", LPS)
    viewer.open(first)
    viewer.open(second)
    wait_until(lambda: not viewer.is_loading())
    sv.QThreadPool.globalInstance().waitForDone(10_000)
    QApplication.processEvents()
    assert viewer.volume is not None and viewer.volume.path.name == "b.nii.gz"


# ---------------------------------------------------------------------------- main window


def test_main_window_view_button(qapp, tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox

    from mskpipe.gui import app as gui_app

    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a) or QMessageBox.Ok)
    )
    fake_cli = [sys.executable, str(Path(__file__).with_name("fake_cli.py"))]
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    win = gui_app.MainWindow(settings, command=fake_cli, probe_device=False)
    try:
        win.image_edit.setText("")
        assert not win.view_image_button.isEnabled()
        assert win.view_image() is None and warned

        path, _ = marked(tmp_path / "s01.nii.gz", RAS)
        win.image_edit.setText(str(path))
        assert win.view_image_button.isEnabled()
        viewer = win.view_image()
        wait_until(lambda: not viewer.is_loading())
        assert viewer.isVisible() and viewer.volume is not None
        assert win.view_image() is viewer  # one window, the file is not read again

        win.image_edit.setText(str(tmp_path / "missing.nii.gz"))
        assert not win.view_image_button.isEnabled()
        viewer.close()
        assert viewer.volume is None
    finally:
        win._pool.waitForDone(30_000)  # modality detection / volume read
        win.deleteLater()
