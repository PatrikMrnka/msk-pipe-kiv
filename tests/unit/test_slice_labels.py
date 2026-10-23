# SPDX-License-Identifier: Apache-2.0
"""Label maps over the slices of the image viewer: run discovery, names, colours, grid
mapping, compositing (numpy), and the viewer window with a run (Qt offscreen)."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

import nibabel as nib
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from mskpipe.gui import slice_labels as sl
from mskpipe.gui import slice_viewer as sv
from mskpipe.io.labels import Label, LabelTable
from mskpipe.io.segmentation import SegmentationIndex, ToolOutput
from mskpipe.qc.defects import Defect, DefectReport

TIMEOUT_S = 30
SHAPE = (20, 16, 12)
RAS = np.diag([0.8, 1.0, 2.0, 1.0])
FEMUR = (slice(2, 6), slice(3, 9), slice(1, 5))  # image voxels
GMED = (slice(12, 18), slice(4, 10), slice(6, 11))


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


def save(path: Path, data: np.ndarray, affine: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def lps_copy(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same image stored with i to the left and j to the posterior (DICOM-like)."""
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    flip[0, 3], flip[1, 3] = data.shape[0] - 1, data.shape[1] - 1
    return np.ascontiguousarray(data[::-1, ::-1, :]), affine @ flip


def make_run(root: Path) -> Path:
    """Run folder with input, unified label map, one tool output (other voxel order) and
    a defect repair."""
    image = np.full(SHAPE, 100, np.int16)
    save(root / "00_input" / "input.nii.gz", image, RAS)
    (root / "input.json").write_text(json.dumps({"image": "gone.nii.gz"}), encoding="utf-8")

    lm = np.zeros(SHAPE, np.uint8)
    lm[FEMUR] = 2
    lm[GMED] = 21
    save(root / "02_labelmap" / "labelmap.nii.gz", lm, RAS)
    LabelTable(
        labels=(
            Label(name="femur_r", value=2, kind="bone"),
            Label(name="gluteus_medius_r", value=21, kind="muscle"),
            Label(name="sacrum", value=8, kind="bone"),  # absent from the map
        )
    ).save(root / "02_labelmap" / "labels.json")

    ts = np.zeros(SHAPE, np.uint8)
    ts[FEMUR] = 76
    data, affine = lps_copy(ts, RAS)
    save(root / "01_segment" / "totalsegmentator_total.nii.gz", data, affine)
    SegmentationIndex(
        outputs=(
            ToolOutput(
                tool="totalsegmentator",
                task="total",
                file="totalsegmentator_total.nii.gz",
                labels={"femur_right": 76, "femur_left": 75},
            ),
        )
    ).save(root / "01_segment" / "segmentation.json")

    patches = np.zeros(SHAPE, np.int16)
    patches[3, 4, 2] = 1
    save(root / "02_labelmap" / "patches.nii.gz", patches, RAS)
    defect = Defect("femur_r", "hole", [3, 4, 2], [2.4, 4, 4], 3.0, True, "plate")
    DefectReport([defect]).write(root / "02_labelmap")
    return root


@pytest.fixture
def run(tmp_path):
    return make_run(tmp_path / "run")


# ---------------------------------------------------------------------------- numpy


def test_run_sources(run):
    image, sources = sl.run_sources(run)
    assert image == run / "00_input" / "input.nii.gz"
    assert [s.title for s in sources] == [
        "Label map (02_labelmap)",
        "TotalSegmentator total (01_segment)",
        "Defect repairs (02_labelmap)",
    ]
    assert sources[0].names[21] == "gluteus_medius_r" and sources[0].kinds[2] == "bone"
    assert sources[1].names == {76: "femur_right", 75: "femur_left"}
    assert sources[2].names == {1: "hole femur_r"}
    # a label file chosen by hand is named from the run files next to it
    assert sl.source_for_file(run / "02_labelmap" / "labelmap.nii.gz").names[2] == "femur_r"
    assert sl.source_for_file(sources[1].path).title == sources[1].title
    assert sl.source_for_file(run / "02_labelmap" / "patches.nii.gz").kinds == {1: "patch"}


def test_run_sources_without_input_copy(run):
    (run / "00_input" / "input.nii.gz").unlink()
    image, _ = sl.run_sources(run)
    assert image is None  # the original named in input.json is gone too


def test_load_layer(run):
    _, sources = sl.run_sources(run)
    layer = sl.load_layer(sources[0])
    assert list(layer.labels) == [2, 21]  # bones first; absent sacrum not listed
    femur = layer.labels[2]
    assert femur.name == "femur_r" and femur.kind == "bone"
    assert femur.voxels == 4 * 6 * 4
    assert femur.volume_ml == pytest.approx(96 * 1.6 / 1000)
    assert femur.centroid == pytest.approx((3.5, 5.5, 2.5))
    assert femur.color in sl.BONE_COLORS
    assert layer.labels[21].kind == "muscle" and layer.labels[21].color not in sl.BONE_COLORS
    i, j, k = layer.locate(21)
    assert layer.data[i, j, k] == 21
    assert layer.same_grid(SHAPE, RAS)

    tool = sl.load_layer(sources[1])  # names without kinds: guessed from the name
    assert tool.labels[76].kind == "bone" and tool.labels[76].name == "femur_right"
    assert not tool.same_grid(SHAPE, RAS)


def test_unnamed_and_invalid_layers(tmp_path):
    data = np.zeros((4, 4, 4), np.float32)
    data[1, 1, 1] = 3.0
    layer = sl.load_layer(sl.LayerSource(save(tmp_path / "f.nii.gz", data, RAS), "f"))
    assert layer.data.dtype == np.uint8
    assert layer.labels[3].name == "label 3" and layer.labels[3].kind == "other"
    data[0, 0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        sl.load_layer(sl.LayerSource(save(tmp_path / "n.nii.gz", data, RAS), "n"))


def test_label_plane_follows_the_world(run):
    """A tool output stored in another voxel order and a coarser label map fall on the
    same image voxels as the unified label map."""
    vol = sv.load_volume(run / "00_input" / "input.nii.gz")
    _, sources = sl.run_sources(run)
    unified = sl.load_layer(sources[0])
    tool = sl.load_layer(sources[1])
    for view in sv.VIEWS:
        index = 3 if view == "axial" else 4
        expected = sv.plane(unified.data, view, index)  # same grid: plain slice
        got = sl.label_plane(unified, vol.affine, vol.shape, sv.AXES[view], index)
        assert np.array_equal(got, expected), view
        femur = sl.label_plane(tool, vol.affine, vol.shape, sv.AXES[view], index) == 76
        assert np.array_equal(femur, expected == 2), view
    assert sl.label_at(tool, vol.affine, (3, 5, 2)) == 76
    assert sl.label_at(tool, vol.affine, (10, 0, 0)) == 0
    # coarse grid (2 mm voxels, other origin): nearest voxel, 0 outside
    coarse = np.zeros((4, 4, 4), np.uint8)
    coarse[1, 1, 1] = 9
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = (4.0, 4.0, 4.0)
    layer = sl.load_layer(sl.LayerSource(save(run / "c.nii.gz", coarse, affine), "c"))
    world = affine @ np.array([1.0, 1.0, 1.0, 1.0])  # centre of the labelled voxel
    ijk = np.rint(np.linalg.solve(vol.affine, world))[:3].astype(int)
    assert sl.label_at(layer, vol.affine, ijk) == 9
    assert sl.to_canonical(layer, vol.affine, vol.shape, (1, 1, 1)) == list(ijk)
    assert sl.label_at(layer, vol.affine, (19, 15, 11)) == 0


def test_compose():
    gray = np.full((3, 4), 100, np.uint8)
    labels = np.zeros((3, 4), np.uint8)
    labels[1, 1:3] = 1
    labels[2, 3] = 2
    lut = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255]], np.uint8)
    alpha = np.array([0, 1, 1], np.float32)
    rgb = sl.compose(gray, labels, lut, alpha, 0.5, outline=False)
    assert rgb.shape == (3, 4, 3) and rgb.flags.c_contiguous
    assert rgb[0, 0].tolist() == [100, 100, 100]
    assert rgb[1, 1].tolist() == [178, 50, 50]
    alpha[2] = 0  # hidden
    assert sl.compose(gray, labels, lut, alpha, 0.5, outline=False)[2, 3].tolist() == [100] * 3
    big = np.zeros((5, 5), np.uint8)
    big[1:4, 1:4] = 1
    out = sl.compose(np.zeros((5, 5), np.uint8), big, lut, alpha, 0.2, outline=True)
    assert out[1, 1].tolist() == [255, 0, 0]  # border: opaque
    assert out[2, 2].tolist() == [0, 0, 0]  # inside: not drawn


# ---------------------------------------------------------------------------- viewer


@pytest.fixture
def viewer(qapp):
    win = sv.SliceViewer()
    win.resize(1200, 500)
    win.show()
    yield win
    win.close()
    QApplication.processEvents()
    win.deleteLater()


def loaded(viewer):
    wait_until(lambda: not viewer.is_loading())
    return viewer


def test_viewer_run(viewer, run):
    assert viewer.open_run(run)
    loaded(viewer)
    assert viewer.volume is not None and viewer.layer is not None
    assert viewer.layer_combo.currentText() == "Label map (02_labelmap)"
    assert viewer.layer_combo.count() == 4  # None + 3 label maps
    names = [viewer.tree.topLevelItem(i).text(0) for i in range(viewer.tree.topLevelItemCount())]
    assert names == ["femur_r", "gluteus_medius_r"]

    viewer.jump_to(21)
    assert viewer.label_under_cursor() == 21
    assert "label gluteus_medius_r (21)" in viewer.status_label.text()
    pane = viewer.panes["axial"]
    assert pane._image.format() == sv.QImage.Format.Format_RGB888
    col, row = pane._cross
    color = pane._image.pixelColor(col, row)
    assert color.red() != color.green() or color.green() != color.blue()  # tinted

    viewer.show_only("bone")
    assert viewer.visible_values() == [2]
    assert viewer._item(21).checkState(0) == sv.Qt.CheckState.Unchecked
    viewer.refresh()
    color = viewer.panes["axial"]._image.pixelColor(col, row)
    assert color.red() == color.green() == color.blue()  # hidden: grey only
    viewer.jump_to(21)  # going to a hidden structure shows it
    assert 21 in viewer.visible_values()

    viewer.show_check.setChecked(False)
    assert viewer.panes["axial"]._image.format() == sv.QImage.Format.Format_Grayscale8
    viewer.show_check.setChecked(True)
    viewer.outline_check.setChecked(True)
    viewer.opacity_slider.setValue(0)  # outlines stay visible
    assert viewer.panes["axial"]._image.format() == sv.QImage.Format.Format_RGB888

    # the other label maps of the run
    viewer.layer_combo.setCurrentIndex(2)
    loaded(viewer)
    assert list(viewer.layer.labels) == [76]
    assert "nearest label voxel" in viewer.layer_info.text()
    viewer.jump_to(76)
    assert viewer.label_under_cursor() == 76
    viewer.layer_combo.setCurrentIndex(1)
    loaded(viewer)
    assert 21 in viewer.visible_values()  # visibility kept per label map
    viewer.layer_combo.setCurrentIndex(0)
    assert viewer.layer is None and viewer.tree.topLevelItemCount() == 0

    # click on a structure selects it in the list
    viewer.layer_combo.setCurrentIndex(1)
    loaded(viewer)
    viewer.cursor = [3, 5, 2]
    viewer.refresh()
    s_col, s_row = viewer.panes["axial"]._cross
    viewer.pick("axial", s_col + 0.5, s_row + 0.5)
    assert viewer.tree.currentItem().text(0) == "femur_r"

    viewer.close()
    assert viewer.layer is None and viewer.volume is None


def test_viewer_add_labels_and_errors(viewer, run, tmp_path):
    viewer.open(run / "00_input" / "input.nii.gz")
    loaded(viewer)
    assert viewer.layer is None and viewer.layer_combo.count() == 1
    viewer.add_labels(run / "02_labelmap" / "patches.nii.gz")
    loaded(viewer)
    assert viewer.layer.labels[1].name == "hole femur_r"
    viewer.add_labels(run / "02_labelmap" / "patches.nii.gz")  # already offered: selected
    assert viewer.layer_combo.count() == 2

    bad = tmp_path / "bad.nii.gz"
    bad.write_bytes(b"not a nifti")
    viewer.add_labels(bad)
    loaded(viewer)
    assert viewer.layer is None and viewer.layer_error
    assert "Cannot read the label map" in viewer.layer_info.text()

    other = save(tmp_path / "other.nii.gz", np.zeros((5, 5, 5), np.int16), RAS)
    viewer.open(other)  # another image without layers: the label maps are dropped
    loaded(viewer)
    assert viewer.layer_combo.count() == 1


def test_run_without_image(viewer, run, monkeypatch):
    shown = []
    monkeypatch.setattr(sv.QMessageBox, "warning", staticmethod(lambda *a: shown.append(a)))
    (run / "00_input" / "input.nii.gz").unlink()
    assert not viewer.open_run(run)
    assert shown


# ---------------------------------------------------------------------------- main window


def test_main_window_segmentation_button(qapp, run, tmp_path):
    import sys

    from PySide6.QtCore import QSettings

    from mskpipe.gui import app as gui_app

    fake_cli = [sys.executable, str(Path(__file__).with_name("fake_cli.py"))]
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    win = gui_app.MainWindow(settings, command=fake_cli, probe_device=False)
    try:
        assert not win.seg_button.isEnabled()
        assert win.view_segmentation() is None
        win.process.run_dir = run
        win._set_running(False)
        assert win.seg_button.isEnabled()
        viewer = win.view_segmentation()
        loaded(viewer)
        assert viewer.layer is not None and viewer.volume is not None
        assert viewer is win.slice_viewer
        viewer.close()
    finally:
        win._pool.waitForDone(30_000)
        win.deleteLater()


def test_view_command_checks_paths(tmp_path):
    from typer.testing import CliRunner

    from mskpipe.cli import app

    result = CliRunner().invoke(app, ["view", str(tmp_path / "missing.nii.gz")])
    assert result.exit_code == 1 and "not found" in result.output
    assert "label maps" in CliRunner().invoke(app, ["view", "--help"]).output
