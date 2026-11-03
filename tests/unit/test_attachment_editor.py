# SPDX-License-Identifier: Apache-2.0
"""Attachment editor: editing model (no display) and the dialogs (Qt offscreen)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_attachments import FakeMesh, atlas, config, spec  # noqa: F401  (fixtures)

from mskpipe.config import load_config
from mskpipe.core.runner import run_pipeline
from mskpipe.geometry.compare import closest_on_surface
from mskpipe.geometry.outline import signed_turn
from mskpipe.gui.attachment_model import AttachmentSession, SessionError
from mskpipe.io.attachment_vtk import read_points, write_points
from mskpipe.io.manual_areas import INDEX_FILE, match_folder, read_index
from mskpipe.steps.attachments import AttachmentsStep, read_attachments_index

ILIACUS_ORI = ("iliacus_r", "Ori")
GRACILIS_INS = ("gracilis_r", "Ins")


@pytest.fixture
def run_dir(tmp_path, spec, atlas):  # noqa: F811
    result = run_pipeline([FakeMesh(), AttachmentsStep()], spec, config(tmp_path, atlas))
    return result.ws.root


@pytest.fixture
def session(run_dir) -> AttachmentSession:
    return AttachmentSession.load(run_dir)


# --------------------------------------------------------------------------- model


def test_load_takes_the_run_outlines(session, run_dir):
    assert session.side == "r"
    assert set(session.muscles) == {"iliacus_r", "gracilis_r"}
    assert set(session.surfaces) == {"pelvis", "femur_r", "tibia_r"}
    assert session.method == "atlas_registration"
    assert session.state(ILIACUS_ORI) == "auto"
    assert session.state(GRACILIS_INS) == "missing"
    area = session.area(ILIACUS_ORI)
    assert area.stem == "Iliacus" and area.body == "pelvis"
    ref = read_points(run_dir / "05_attachments/iliacus_r/Iliacus_Ori.vtk")
    assert np.allclose(area.points, ref) and np.allclose(area.auto, ref)


def test_run_without_meshes(tmp_path):
    with pytest.raises(SessionError):
        AttachmentSession.load(tmp_path)


def test_edit_and_undo(session):
    before = session.points(ILIACUS_ORI).copy()
    n = len(before)
    p = session.snap(before[0] + (5.0, 0, 0), "pelvis")
    assert session.add_point(ILIACUS_ORI, p, 2) == 2
    assert len(session.points(ILIACUS_ORI)) == n + 1
    assert session.state(ILIACUS_ORI) == "edited" and session.dirty
    session.move_point(ILIACUS_ORI, 0, p)
    session.move_point(ILIACUS_ORI, 0, p + 1.0, record=False)  # same drag
    session.remove_point(ILIACUS_ORI, 1)
    assert session.undo() == ILIACUS_ORI  # remove
    assert session.undo() == ILIACUS_ORI  # drag (one step)
    assert session.undo() == ILIACUS_ORI  # add
    assert np.allclose(session.points(ILIACUS_ORI), before)
    assert session.state(ILIACUS_ORI) == "auto"
    assert session.undo() is None


def test_new_area_drawn_and_reset(session):
    femur = session.surfaces["femur_r"].mesh
    centre = femur.vertices.mean(0)
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring = centre + np.c_[8 * np.cos(ang), np.full(6, -40.0), 8 * np.sin(ang)]
    for q in closest_on_surface(ring, femur)[0]:
        session.add_point(GRACILIS_INS, q)
    assert session.state(GRACILIS_INS) == "new"
    assert session.area(GRACILIS_INS).body == "femur_r"
    assert not session.reset(GRACILIS_INS)  # no outline of the run to go back to
    session.clear(GRACILIS_INS)
    assert session.state(GRACILIS_INS) == "missing"
    assert session.undo() == GRACILIS_INS
    assert session.state(GRACILIS_INS) == "new"


def test_reorder_and_check(session):
    pts = session.points(ILIACUS_ORI).copy()
    shuffled = pts[np.r_[0, np.random.default_rng(1).permutation(np.arange(1, len(pts)))]]
    session.areas[ILIACUS_ORI].points = shuffled
    assert session.reorder(ILIACUS_ORI)
    ordered = session.points(ILIACUS_ORI)
    surface = session.surfaces["pelvis"]
    out = surface.outward(surface.closest(ordered)[2])
    assert signed_turn(ordered, out) < 0  # clockwise seen from outside
    rep = session.check(ILIACUS_ORI)
    assert rep["n_points"] == len(pts) and "problems" in rep
    assert session.check(GRACILIS_INS)["problems"][0].startswith("0 point(s)")


def test_save_is_read_by_method_manual(session, tmp_path, spec):  # noqa: F811
    folder = session.save(tmp_path / "manual")
    assert not session.dirty
    index = read_index(folder)
    assert index["side"] == "r" and index["areas"]["iliacus_r"]["Ori"]["source"] == "auto"
    assert "gracilis_r" not in index["areas"]
    match = match_folder(folder, "r", codes={})
    assert set(match.areas) == {("iliacus_r", "Ori"), ("iliacus_r", "Ins")}
    assert not match.unmatched and (folder / INDEX_FILE).is_file()

    cfg = load_config(
        overrides=[
            f"runtime.runs_dir={(tmp_path / 'runs2').as_posix()}",
            "attachments.method=manual",
            f"attachments.params.areas_dir={folder.as_posix()}",
        ]
    )
    result = run_pipeline([FakeMesh(), AttachmentsStep()], spec, cfg)
    out = result.ws.step_dir("attachments")
    assert read_attachments_index(out)["method"] == "manual"
    for kind in ("Ori", "Ins"):
        got = read_points(out / f"iliacus_r/Iliacus_{kind}.vtk")
        assert np.abs(got - session.points(("iliacus_r", kind))).max() < 1e-3


def test_save_replaces_cleared_area(session, tmp_path):
    folder = session.save(tmp_path / "manual")
    session.clear(ILIACUS_ORI)
    session.save(folder)
    assert not (folder / "iliacus_r/Iliacus_Ori.vtk").exists()
    assert (folder / "iliacus_r/Iliacus_Ins.vtk").exists()


def test_import_files(session, tmp_path):
    src = tmp_path / "nms"
    src.mkdir()
    pts = session.points(ILIACUS_ORI) + np.array([0.0, -2.0, 0.0])  # 2 mm off the bone
    np.savetxt(src / "Origin iliacus right.txt", pts)
    write_points(src / "gracilis_l_Ins.vtk", pts)  # other side: skipped
    match = match_folder(src, "r", codes={})
    keys = session.import_files(list(match.areas.values()))
    assert keys == [ILIACUS_ORI]
    assert session.state(ILIACUS_ORI) == "imported"
    on_bone = session.surfaces["pelvis"].closest(session.points(ILIACUS_ORI))[1]
    assert on_bone.max() < 1e-6  # snapped


# --------------------------------------------------------------------------- dialogs


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def no_boxes(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    for name in ("critical", "warning", "information"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )


def test_import_dialog(qapp, tmp_path, no_boxes):
    from mskpipe.gui.attachment_editor import ImportDialog

    pts = np.random.default_rng(0).normal(size=(5, 3))
    write_points(tmp_path / "iliacus_r_Ori.vtk", pts)
    write_points(tmp_path / "iliacus_origin.vtk", pts)  # duplicate
    np.savetxt(tmp_path / "notes.txt", pts)  # unknown
    dialog = ImportDialog("r", ["iliacus_r", "gracilis_r"], folder=str(tmp_path))
    ok = dialog.buttons.button(dialog.buttons.StandardButton.Ok)
    assert dialog.table.rowCount() == 3 and not ok.isEnabled()  # duplicate
    rows = {dialog.table.item(r, 0).text(): r for r in range(3)}
    dialog.table.cellWidget(rows["iliacus_origin.vtk"], 1).setCurrentIndex(0)  # skip it
    notes = rows["notes.txt"]
    box = dialog.table.cellWidget(notes, 1)
    box.setCurrentIndex(box.findData("gracilis_r"))
    dialog.table.cellWidget(notes, 2).setCurrentIndex(1)  # Ins
    assert ok.isEnabled()
    chosen = {(a.path.name, a.muscle, a.kind) for a in dialog.assignments()}
    assert chosen == {("iliacus_r_Ori.vtk", "iliacus_r", "Ori"), ("notes.txt", "gracilis_r", "Ins")}
    dialog.accept()
    again = match_folder(tmp_path, "r", codes={})  # areas.csv written
    assert set(again.areas) == {("iliacus_r", "Ori"), ("gracilis_r", "Ins")}


@pytest.fixture
def editor_cls(qapp):
    """Editor whose 3D view is a plain widget with a VTK window that never renders (no
    OpenGL in the test environment)."""
    import vtk
    from PySide6.QtWidgets import QWidget

    from mskpipe.gui.attachment_editor import AttachmentEditor

    class FakeView(QWidget):
        def __init__(self, parent):
            super().__init__(parent)
            self.interactor = vtk.vtkRenderWindowInteractor()

        def GetRenderWindow(self):
            return self

        def AddRenderer(self, renderer):
            self.renderer = renderer

        def GetInteractor(self):
            return self.interactor

        def Render(self):
            pass

        def Initialize(self):
            pass

        def Finalize(self):
            pass

    class Editor(AttachmentEditor):
        def _view_widget(self):
            return FakeView(self)

    return Editor


def test_editor_dialog(qapp, session, tmp_path, no_boxes, editor_cls):
    editor = editor_cls(session.run_dir, session=session)
    try:
        assert editor.key == ("gracilis_r", "Ori")  # first area in the list
        assert editor.items[ILIACUS_ORI].text(1) == "auto"
        editor.tree.setCurrentItem(editor.items[ILIACUS_ORI])
        assert editor.key == ILIACUS_ORI
        n = len(session.points(ILIACUS_ORI))
        editor.scene.selected = 0
        editor.remove_selected()
        assert len(session.points(ILIACUS_ORI)) == n - 1
        assert editor.items[ILIACUS_ORI].text(1) == "edited"
        editor.undo()
        assert len(session.points(ILIACUS_ORI)) == n
        editor.step_area(1)
        assert editor.key == ("iliacus_r", "Ins")
        got = []
        editor.applied.connect(lambda folder, run: got.append((folder, run)))
        editor.folder_edit.setText(str(tmp_path / "out"))
        assert editor.save(run=False) == tmp_path / "out"
        assert got == [(str(tmp_path / "out"), False)]
        assert (tmp_path / "out" / "iliacus_r" / "Iliacus_Ori.vtk").is_file()
    finally:
        editor.release()
        editor.deleteLater()


def test_window_switches_to_manual(qapp, tmp_path, no_boxes):
    from PySide6.QtCore import QSettings

    from mskpipe.gui import app as gui_app

    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    win = gui_app.MainWindow(settings, probe_device=False)
    try:
        assert not win.attachments_button.isEnabled()
        win.use_manual_attachments(tmp_path / "areas")
        assert win.form.field("attachments", "method").get() == "manual"
        assert win.form.field("attachments", "params", "areas_dir").get() == (
            (tmp_path / "areas").as_posix()
        )
        assert "attachments.method=manual" in win.form.overrides()
    finally:
        win._pool.waitForDone(30_000)
        win.deleteLater()


# --------------------------------------------------------------------------- scene


def test_scene_picking(session):
    import vtk

    from mskpipe.gui.attachment_scene import AttachmentScene

    scene = AttachmentScene(session.body_meshes())
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(True)
    window.SetSize(600, 600)
    window.AddRenderer(scene.renderer)  # never rendered: no OpenGL needed
    pts = session.points(ILIACUS_ORI)
    surface = session.surfaces["pelvis"]
    scene.show_area("iliacus_r", "pelvis", pts, auto=pts)
    scene.focus(pts, surface.outward(surface.closest(pts)[2]))
    disp = scene.to_display(pts)
    assert scene.handle_at(*disp[3]) == 3
    mid = (disp[3] + disp[4]) / 2
    assert scene.handle_at(*mid, tol_px=1.0) is None
    assert scene.segment_at(*mid) == 4  # inserted between points 3 and 4
    centre = scene.to_display(pts.mean(0, keepdims=True))[0]
    hit = scene.pick_bone(*centre)
    assert hit is not None and hit[0] == "pelvis"
    assert surface.closest(hit[1][None])[1][0] < 0.5  # on the surface


def test_import_in_another_frame_is_refused(session, tmp_path):
    from mskpipe.plugins.attachments.manual import FrameMismatchError

    src = tmp_path / "far"
    before = session.points(ILIACUS_ORI).copy()
    write_points(src / "iliacus_r_Ori.vtk", before + np.array([300.0, 100.0, 1000.0]))
    match = match_folder(src, "r", codes={})
    with pytest.raises(FrameMismatchError, match="another coordinate frame"):
        session.import_files(list(match.areas.values()))
    assert np.allclose(session.points(ILIACUS_ORI), before)  # nothing imported
    assert session.state(ILIACUS_ORI) == "auto" and not session.can_undo


def test_failed_run_starts_from_a_run_with_the_same_meshes(run_dir, tmp_path, spec):  # noqa: F811
    from mskpipe.core.runner import PipelineError

    bad = tmp_path / "bad"
    write_points(bad / "iliacus_r_Ori.vtk", np.random.default_rng(0).normal(size=(6, 3)) + 2000)
    cfg = load_config(
        overrides=[
            f"runtime.runs_dir={run_dir.parent.as_posix()}",
            "attachments.method=manual",
            f"attachments.params.areas_dir={bad.as_posix()}",
        ]
    )
    with pytest.raises(PipelineError) as info:
        run_pipeline([FakeMesh(), AttachmentsStep()], spec, cfg)
    assert "bones: x" in str(info.value.__cause__)
    failed = next(p for p in run_dir.parent.iterdir() if p.is_dir() and p != run_dir)
    session = AttachmentSession.load(failed)
    assert session.source_run == run_dir
    assert session.method == "atlas_registration"
    assert session.state(ILIACUS_ORI) == "auto"


# --------------------------------------------------------------------------- frame


def lps_copy(session, folder):
    """The run's iliacus origin written in LPS (x and y flipped)."""
    pts = session.points(ILIACUS_ORI) * np.array([-1.0, -1.0, 1.0])
    write_points(folder / "iliacus_r_Ori.vtk", pts)
    return folder


def test_session_guesses_frame_by_its_bones(session, tmp_path):
    folder = lps_copy(session, tmp_path / "lps")
    got = session.guess_frame([folder / "iliacus_r_Ori.vtk"])
    assert got.frame == "lps" and got.basis == "bones" and got.scores["lps"] < 5.0
    assert session.guess_frame([tmp_path / "missing.vtk"]) is None


def test_import_dialog_preselects_frame(qapp, session, tmp_path, no_boxes):
    from mskpipe.gui.attachment_editor import ImportDialog

    folder = lps_copy(session, tmp_path / "lps")
    dialog = ImportDialog("r", list(session.muscles), folder=str(folder), guess=session.guess_frame)
    assert dialog.frame == "lps"
    assert dialog.frame_note.text().startswith("Detected: LPS")
    keys = session.import_files(dialog.assignments(), dialog.frame)
    assert keys == [ILIACUS_ORI]


def test_run_with_bones(run_dir, spec, tmp_path):  # noqa: F811
    from mskpipe.gui.attachment_model import bone_vertices, run_with_bones

    assert run_with_bones(run_dir.parent, spec.image) == run_dir
    assert run_with_bones(run_dir.parent, tmp_path / "other.nii.gz") is None
    assert run_with_bones(tmp_path / "no_runs", spec.image) is None
    assert len(bone_vertices(run_dir)) >= 3


def test_window_fills_in_frame(qapp, session, run_dir, spec, tmp_path, no_boxes):  # noqa: F811
    from PySide6.QtCore import QSettings

    from mskpipe.gui import app as gui_app

    frame = ("attachments", "params", "frame")
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    win = gui_app.MainWindow(settings, probe_device=False)
    try:
        assert win.form.is_shown(*frame) is False  # method atlas_registration
        win.runs_edit.setText(str(run_dir.parent))
        win.set_input(spec.image)
        win.use_manual_attachments(lps_copy(session, tmp_path / "lps"))
        win._validate()
        assert win.form.field(*frame).get() == "lps"
        assert win.form.is_shown(*frame)  # basic setting
        note = win.form.note(*frame)
        assert "detected LPS" in note and run_dir.name in note
        assert win.frame_mismatch() is None
        # chosen by hand afterwards: kept while the inputs stay the same, flagged
        win.form.field(*frame).set("ras")
        win._validate()
        assert win.form.field(*frame).get() == "ras"
        assert "RAS does not fit: detected LPS" in win.form.note(*frame)
        assert "attachments.params.frame is ras" in win.frame_mismatch()
        win._store()
    finally:
        win._pool.waitForDone(30_000)
        win.deleteLater()
    # next start with the wrong frame in the saved settings: detected again
    win = gui_app.MainWindow(settings, probe_device=False)
    try:
        assert win.form.field(*frame).get() == "lps"
        assert "detected LPS" in win.form.note(*frame)
        # nothing to compare with (no run of this input, unreadable image): kept, no note
        win.runs_edit.setText(str(tmp_path / "empty"))
        win.set_input(tmp_path / "other.nii.gz")
        win.form.field(*frame).set("ras")
        win._validate()
        assert win.form.field(*frame).get() == "ras" and win.form.note(*frame) == ""
        # method changed and back: the note follows the rebuilt field
        win.form.field("attachments", "method").set("atlas_registration")
        win._validate()
        assert win.form.note(*frame) == ""
    finally:
        win._pool.waitForDone(30_000)
        win.deleteLater()


def test_window_fills_in_frame_from_the_image(qapp, tmp_path, no_boxes):
    import nibabel as nib
    from PySide6.QtCore import QSettings

    from mskpipe.gui import app as gui_app

    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [-400.0, -350.0, -1250.0]
    image = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((300, 300, 900), np.int16), affine), str(image))
    folder = tmp_path / "areas"
    write_points(folder / "iliacus_r_Ori.vtk", np.array([[200.0, 250.0, -500.0]]) + np.eye(3))
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    win = gui_app.MainWindow(settings, probe_device=False)
    try:
        win.runs_edit.setText(str(tmp_path / "runs"))
        win.set_input(image)
        win.use_manual_attachments(folder)
        win._validate()
        assert win.form.field("attachments", "params", "frame").get() == "lps"
        assert "inside the input image" in win.form.note("attachments", "params", "frame")
    finally:
        win._pool.waitForDone(30_000)
        win.deleteLater()
