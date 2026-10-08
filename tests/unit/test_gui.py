# SPDX-License-Identifier: Apache-2.0
"""GUI tests without a display (Qt offscreen) and without the real pipeline.

Runs are real child processes of ``tests/unit/fake_cli.py`` (the mskpipe CLI with fake
steps), so the event stream, cancellation and process-tree killing are exercised.
"""

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

import yaml
from fake_pipeline import write_nifti
from PySide6.QtCore import QEventLoop, QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

from mskpipe import api
from mskpipe.config import load_config
from mskpipe.core.registry import Registry, default_registry
from mskpipe.gui import app as gui_app
from mskpipe.gui.config_form import ConfigForm, humanize, yaml_value
from mskpipe.gui.process import (
    ChildRequest,
    PipelineProcess,
    child_command,
    python_executable,
)
from mskpipe.plugins.attachments.bone_registration import BoneRegistration
from mskpipe.plugins.base import AttachmentsPlugin, PluginParams

FAKE_CLI = [sys.executable, str(Path(__file__).with_name("fake_cli.py"))]
TIMEOUT_S = 60


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch):
    """Modal message boxes would block the offscreen test run."""
    calls = []
    for name in ("critical", "warning", "information"):
        monkeypatch.setattr(
            QMessageBox,
            name,
            staticmethod(lambda *a, _n=name, **k: calls.append((_n, a)) or QMessageBox.Ok),
        )
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    return calls


def wait_until(condition, timeout_s: float = TIMEOUT_S) -> None:
    end = time.monotonic() + timeout_s
    while not condition():
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if time.monotonic() > end:
            raise TimeoutError("condition not met")
        time.sleep(0.01)


# ---------------------------------------------------------------------------- form


def test_helpers():
    assert humanize("repair_max_opening_mm") == "Repair max opening [mm]"
    assert humanize("smooth_iterations") == "Smooth iterations"
    assert yaml_value(["a", "b"]) == "[a, b]"
    assert yaml_value("yes") == "'yes'"  # stays a string for --set
    assert yaml_value(None) == "null"


def test_form_defaults_and_overrides(qapp):
    form = ConfigForm()
    assert form.tab_names() == [
        "Runtime",
        "Segmentation",
        "Labelmap",
        "Mesh",
        "Skeleton",
        "Attachments",
        "Export",
    ]
    assert form.overrides() == []
    assert ("runtime", "device") not in form._fields  # set by the window
    form.field("mesh", "bones", "smooth_iterations").set(10)
    form.field("segmentation", "musclemap", "chunk_size").set(15)
    form.field("export", "exclude").set(["piriformis"])
    form.field("runtime", "threads").set(4)
    form.field("attachments", "params", "nonrigid").set("cpd")
    form.field("attachments", "params", "atlas_dir").set("C:/data/atlas")
    overrides = form.overrides()
    assert set(overrides) == {
        "runtime.threads=4",
        "segmentation.musclemap.chunk_size=15",
        "mesh.bones.smooth_iterations=10",
        "export.exclude=[piriformis]",
        "attachments.params.nonrigid=cpd",
        "attachments.params.atlas_dir=C:/data/atlas",
    }
    config = api.resolved_config(overrides=overrides)
    assert config.mesh.bones.smooth_iterations == 10
    assert config.segmentation.musclemap.chunk_size == 15
    assert config.attachments.params["atlas_dir"] == "C:/data/atlas"
    label = form.field("mesh", "bones", "smooth_iterations").label
    assert label is not None and label.font().bold()
    form.reset()
    assert form.overrides() == []
    assert not label.font().bold()


def test_form_nullable_and_validation(qapp):
    form = ConfigForm()
    threads = form.field("runtime", "threads")
    threads.set(None)
    assert threads.get() is None
    assert form.validate() is None
    chunk = form.field("segmentation", "musclemap", "chunk_size")
    chunk.editor.setText("big")  # neither an integer nor 'auto'
    message = form.validate()
    assert message and "segmentation.musclemap.chunk_size" in message
    assert "c0392b" in chunk.editor.styleSheet()  # union error path maps to the field
    assert not form.error_label.isHidden()
    chunk.editor.setText("auto")
    assert form.validate() is None
    assert form.error_label.isHidden()


class _OtherParams(PluginParams):
    radius_mm: float = 5.0


class _Other(AttachmentsPlugin):
    name = "other_method"
    Params = _OtherParams

    def compute(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


def test_form_plugin_params_follow_method(qapp):
    registry = Registry.default()
    registry.register(_Other)
    form = ConfigForm(registry)
    assert ("attachments", "params", "nonrigid") in form._fields
    form.field("attachments", "method").set("other_method")
    assert ("attachments", "params", "nonrigid") not in form._fields
    assert form.field("attachments", "params", "radius_mm").get() == 5.0
    # only the method differs: the new plugin's defaults are not overrides
    assert form.overrides() == ["attachments.method=other_method"]
    form.field("attachments", "params", "radius_mm").set(7.5)
    assert "attachments.params.radius_mm=7.5" in form.overrides()
    form.field("attachments", "method").set("bone_registration")
    assert form.field("attachments", "params", "nonrigid").get() == "coherent_icp"
    assert form.overrides() == []


def test_form_load_save_roundtrip(qapp, tmp_path):
    source = tmp_path / "in.yaml"
    source.write_text(
        "runtime: {device: gpu, cache: false}\n"
        "skeleton: {side: l}\n"
        "attachments: {params: {nonrigid: none, n_points: 4000}}\n",
        encoding="utf-8",
    )
    form = ConfigForm()
    data = form.load(source)
    assert data["runtime"]["device"] == "gpu"
    assert form.field("skeleton", "side").get() == "l"
    assert form.field("attachments", "params", "n_points").get() == 4000
    target = tmp_path / "out.yaml"
    form.save(target, {"runtime": {"device": "gpu"}})
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# msk-pipe configuration")
    saved = yaml.safe_load(text)
    assert saved["skeleton"] == {"side": "l"} and saved["runtime"]["cache"] is False
    config = load_config(target)
    assert config.runtime.device.value == "gpu"
    assert config.attachments.params == {"nonrigid": "none", "n_points": 4000}
    bad = tmp_path / "bad.yaml"
    bad.write_text("skeleton: {side: x}\n", encoding="utf-8")
    with pytest.raises(api.SetupError):
        form.load(bad)


def test_every_schema_field_roundtrips(qapp):
    """Each widget gives back the default it was set to (no lossy conversions)."""
    form = ConfigForm()
    ref = form.reference()
    lossy = {p: (f.get(), ref[p]) for p, f in form._fields.items() if f.get() != ref[p]}
    assert lossy == {}


# ---------------------------------------------------------------------------- process


def test_child_command():
    request = ChildRequest(
        image=Path("ct.nii.gz"),
        modality="ct",
        runs_dir=Path("runs"),
        device="gpu",
        subject="S1",
        until_step="mesh",
        overrides=("a.b=1", "c=[x, y]"),
        verbose=True,
    )
    argv = child_command(request, ["py", "-m", "mskpipe"])
    assert argv[:5] == ["py", "-m", "mskpipe", "run", "ct.nii.gz"]
    assert argv[argv.index("--set") + 1] == "a.b=1"
    assert argv.count("--set") == 2
    for flag in ("--no-preflight", "--cancel-on-stdin", "--verbose"):
        assert flag in argv
    assert argv[argv.index("--events") + 1] == "jsonl"
    assert argv[argv.index("--until") + 1] == "mesh"
    assert child_command(request)[0] == python_executable()


def test_python_executable_prefers_console(tmp_path, monkeypatch):
    (tmp_path / "pythonw.exe").write_bytes(b"")
    (tmp_path / "python.exe").write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "pythonw.exe"))
    assert Path(python_executable()).name == "python.exe"


def test_process_text_and_exit_code(qapp):
    proc = PipelineProcess()
    lines, done = [], []
    proc.text.connect(lambda line, ch: lines.append((ch, line)))
    proc.finished.connect(lambda code, killed: done.append((code, killed)))
    code = (
        "import sys; print('plain'); "
        'print(\'{"kind": "log", "message": "hi"}\'); '
        "sys.stderr.write('Error: bad\\n'); sys.exit(2)"
    )
    events = []
    proc.event.connect(events.append)
    proc.start([sys.executable, "-c", code])
    wait_until(lambda: done)
    assert done == [(2, False)]
    assert ("stdout", "plain") in lines and ("stderr", "Error: bad") in lines
    assert [e.message for e in events] == ["hi"]


def test_process_failed_to_start(qapp):
    proc = PipelineProcess()
    done, lines = [], []
    proc.finished.connect(lambda code, killed: done.append(code))
    proc.text.connect(lambda line, ch: lines.append(line))
    proc.start(["/nonexistent/python-xyz"])
    wait_until(lambda: done)
    assert done == [-1] and "Cannot start" in lines[0]


# ---------------------------------------------------------------------------- window


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "preflight", lambda prepared, registry=None: [])
    for var in ("FAKE_FAIL", "FAKE_WAIT_CANCEL", "FAKE_HANG"):
        monkeypatch.delenv(var, raising=False)
    settings = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    win = gui_app.MainWindow(settings, command=FAKE_CLI, probe_device=False)
    image = write_nifti(tmp_path / "data" / "s01.nii.gz")
    win.image_edit.setText(str(image))
    win._image_changed()
    win.runs_edit.setText(str(tmp_path / "runs"))
    gui_app._select(win.device_combo, "cpu")
    gui_app._select(win.modality_combo, "ct")
    yield win
    if win.process.is_running():
        win.process.kill()
        win.process.wait(5000)
    win._pool.waitForDone(30_000)  # modality detection thread
    win.deleteLater()


def statuses(win) -> dict[str, str]:
    tree = win.steps_tree
    return {
        tree.topLevelItem(i).data(0, 256): tree.topLevelItem(i).text(1)
        for i in range(tree.topLevelItemCount())
    }


def test_window_image_info(window):
    assert "8 x 8 x 6 voxels" in window.image_info.text()
    assert window.subject_edit.placeholderText() == "s01"
    window.image_edit.setText("/missing.nii.gz")
    window._image_changed()
    assert "not found" in window.image_info.text()


def test_window_run_completed(window):
    window.form.field("mesh", "bones", "smooth_iterations").set(12)
    assert window.start_run()
    assert not window.run_button.isEnabled() and window.cancel_button.isEnabled()
    wait_until(lambda: not window.process.is_running() and window.summary is not None)
    s = window.summary
    assert s.status == "completed", window.log_view.toPlainText()
    assert set(statuses(window).values()) == {"completed"}
    assert window.open_mw2_button.isEnabled() and window.open_run_button.isEnabled()
    assert "Completed" in window.status_label.text()
    assert "fake work" in window.log_view.toPlainText()
    assert window.progress.value() == window.progress.maximum() == 6
    resolved = yaml.safe_load((s.run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["mesh"]["bones"]["smooth_iterations"] == 12
    assert resolved["runtime"]["device"] == "cpu"


def test_window_until_and_failure(window, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "mesh")
    gui_app._select(window.until_combo, "skeleton")
    assert window.start_run()
    wait_until(lambda: not window.process.is_running() and window.summary is not None)
    assert window.summary.status == "failed" and window.summary.failed_step == "mesh"
    st = statuses(window)
    assert st["mesh"] == "failed" and st["labelmap"] == "completed"
    assert st["attachments"] == ""  # not planned (until skeleton)
    assert not window.open_mw2_button.isEnabled()
    assert "mesh broken" in window.status_label.text()


def test_window_cancel(window, monkeypatch):
    monkeypatch.setenv("FAKE_WAIT_CANCEL", "labelmap")
    assert window.start_run()
    wait_until(lambda: statuses(window)["labelmap"] == "running")
    window.cancel_run()
    wait_until(lambda: not window.process.is_running() and window.summary is not None)
    assert window.summary.status == "interrupted"
    assert window.summary.failed_step == "labelmap"
    assert statuses(window)["labelmap"] == "interrupted"
    assert "Cancelled" in window.status_label.text()


def test_window_kill_after_cancel_timeout(window, monkeypatch):
    monkeypatch.setenv("FAKE_HANG", "labelmap")
    window.process.cancel_grace_ms = 500
    assert window.start_run()
    wait_until(lambda: statuses(window)["labelmap"] == "running")
    window.cancel_run()
    wait_until(lambda: not window.process.is_running() and window.summary is not None)
    s = window.summary
    assert s.status == "interrupted" and "killed by the GUI" in (s.error or "")
    assert "process killed" in window.status_label.text()


def test_window_preflight_errors(window, monkeypatch):
    issue = api.Issue("error", "attachments", "no atlas")
    monkeypatch.setattr(api, "preflight", lambda prepared, registry=None: [issue])
    asked = []
    monkeypatch.setattr(window, "_confirm_errors", lambda errors: asked.append(errors) or False)
    assert window.start_run() is False
    assert asked and not window.process.is_running()


def test_window_input_errors(window, _no_dialogs):
    window.image_edit.setText("")
    assert window.start_run() is False
    assert _no_dialogs[-1][0] == "warning"
    window.image_edit.setText(str(Path(window.runs_edit.text()) / "missing.nii.gz"))
    assert window.start_run() is False


def test_window_settings_persist(qapp, tmp_path):
    path = str(tmp_path / "gui.ini")
    first = gui_app.MainWindow(
        QSettings(path, QSettings.Format.IniFormat), command=FAKE_CLI, probe_device=False
    )
    first.form.field("skeleton", "side").set("l")
    first.form.field("attachments", "params", "atlas_dir").set("D:/atlas")
    gui_app._select(first.modality_combo, "mri")
    first.runs_edit.setText(str(tmp_path / "out"))
    first._store()
    second = gui_app.MainWindow(
        QSettings(path, QSettings.Format.IniFormat), command=FAKE_CLI, probe_device=False
    )
    assert second.form.field("skeleton", "side").get() == "l"
    assert second.form.field("attachments", "params", "atlas_dir").get() == "D:/atlas"
    assert second.modality_combo.currentData() == "mri"
    assert second.runs_edit.text() == str(tmp_path / "out")


def test_window_device_detection(qapp, tmp_path):
    win = gui_app.MainWindow(
        QSettings(str(tmp_path / "g.ini"), QSettings.Format.IniFormat),
        command=FAKE_CLI,
        probe_device=False,
    )
    gui_app._select(win.device_combo, "cpu")
    win.detect_device()
    wait_until(lambda: win.device_label.text() not in ("", "detecting…"))
    assert win.device_label.text().startswith("cpu - requested")


def test_gui_does_not_change_registry_default():
    assert "other_method" not in default_registry().names("attachments")
    assert BoneRegistration.name in default_registry().names("attachments")


# ---------------------------------------------------------------------------- explanations


def test_form_explanations_behind_badges(qapp):
    from PySide6.QtWidgets import QLabel

    from mskpipe.gui.help_badge import HelpBadge

    form = ConfigForm()
    text = form.help_text("mesh", "bones", "smooth_iterations")
    assert "staircase" in text and "Example:" in text and "mesh.bones.smooth_iterations" in text
    assert "Example:" in form.help_text("attachments", "params", "nonrigid")
    badge = form._badges[("mesh", "bones", "smooth_iterations")]
    assert isinstance(badge, HelpBadge) and badge.text() == "?"
    assert form.field("mesh", "bones", "smooth_iterations").editor.toolTip() == text
    # no explanation text in the form itself, only behind the badges
    shown = [w.text() for w in form.findChildren(QLabel) if not isinstance(w, HelpBadge)]
    assert not any("staircase" in s for s in shown)
    # sections and groups: tooltip on the tab and on the group box
    tabs = form.tabs
    assert "label map" in tabs.tabToolTip(form.tab_names().index("Labelmap"))
    form.field("attachments", "params", "nonrigid").set("cpd")  # rebuild keeps explanations
    assert form.help_text("attachments", "params", "cpd_beta")
    assert ("attachments", "params", "coherent_beta_mm") in form._badges  # all params rebuilt


def test_badge_shows_help_on_hover(qapp, monkeypatch):
    from mskpipe.gui import help_badge

    shown = []
    monkeypatch.setattr(
        help_badge.QToolTip, "showText", staticmethod(lambda pos, text, w=None: shown.append(text))
    )
    badge = help_badge.HelpBadge("<p>explanation</p>")
    badge.show_help()
    assert shown == ["<p>explanation</p>"]


def test_window_badges(window):
    from PySide6.QtWidgets import QCheckBox

    assert set(window._badges) == {
        "gui.image",
        "gui.modality",
        "gui.subject",
        "gui.device",
        "gui.runs_dir",
        "gui.until",
    }
    assert "Hounsfield" in window._badges["gui.modality"].toolTip()
    assert "TotalSegmentator" in window.verbose_check.toolTip()
    texts = [c.text() for c in window.findChildren(QCheckBox)]
    assert "Show explanations" not in texts


# ---------------------------------------------------------------------------- muscles


def test_exported_muscles_list(qapp):
    form = ConfigForm()
    assert ("export", "muscles") not in form._fields  # the GUI keeps 'all'
    label = form.field("export", "exclude").label
    assert label is not None and label.text() == "Exported muscles"
    picker = form.picker("export", "exclude")
    default_out = {"piriformis", "obturator_internus", "obturator_externus", "tensor_fasciae_latae"}
    assert set(picker.names()) - set(picker.checked()) == default_out
    assert form.overrides() == []
    tip = picker.list.item(picker.names().index("piriformis")).toolTip()
    assert "Not exported by default" in tip and "greater trochanter" in tip

    picker.set_checked(["gluteus_maximus", "gluteus_medius", "gluteus_minimus"])
    (override,) = form.overrides()
    excluded = yaml.safe_load(override.split("=", 1)[1])
    assert set(excluded) == set(picker.names()) - {
        "gluteus_maximus",
        "gluteus_medius",
        "gluteus_minimus",
    }
    assert excluded[:4] == [  # default exclusions first, in their order
        "piriformis",
        "obturator_internus",
        "obturator_externus",
        "tensor_fasciae_latae",
    ]
    config = api.resolved_config(overrides=form.overrides())
    assert config.export.muscles == "all" and "gluteus_maximus" not in config.export.exclude

    picker._check_all()  # export everything, piriformis included
    assert form.overrides() == ["export.exclude=[]"]
    picker.set_checked([n for n in picker.names() if n not in default_out])  # Defaults
    assert form.overrides() == []


def test_explicit_muscle_list_from_config(qapp, tmp_path):
    source = tmp_path / "quick.yaml"
    source.write_text(
        "skeleton: {side: l}\n"
        "export: {muscles: [iliacus_l, pectineus_l, piriformis_l, unknown_muscle_l]}\n",
        encoding="utf-8",
    )
    form = ConfigForm()
    form.load(source)
    picker = form.picker("export", "exclude")
    # piriformis stays excluded by the default exclusions, unknown names are kept
    assert picker.checked() == ["iliacus", "pectineus", "unknown_muscle"]
    config = api.resolved_config(overrides=form.overrides())
    assert config.export.muscles == "all"
    assert "gluteus_maximus" in config.export.exclude and "iliacus" not in config.export.exclude


# ---------------------------------------------------------------------------- modality


def _ct(path):
    import nibabel as nib
    import numpy as np

    data = np.full((30, 30, 20), -1024, np.int16)
    data[8:22, 8:22, :] = 40
    affine = np.diag([0.8, 0.8, 1.5, 1.0])
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, 1)
    img.set_sform(affine, 1)
    nib.save(img, str(path))
    return path


def test_window_modality_auto(window, tmp_path, _no_dialogs):
    image = _ct(tmp_path / "ct.nii.gz")
    gui_app._select(window.modality_combo, "auto")
    window.image_edit.setText(str(image))
    window._image_changed()
    wait_until(lambda: "detected" in window.modality_label.text())
    assert "detected CT" in window.modality_label.text()
    assert window.effective_modality(image) == "ct"
    gui_app._select(window.modality_combo, "mri")
    assert "looks like CT" in window.modality_label.text()
    assert window.effective_modality(image) == "mri"  # the user's choice wins

    gui_app._select(window.modality_combo, "auto")
    import nibabel as nib
    import numpy as np

    odd = tmp_path / "odd.nii.gz"
    nib.save(nib.Nifti1Image(np.full((10, 10, 10), -400, np.int16), np.eye(4)), str(odd))
    window.image_edit.setText(str(odd))
    window._image_changed()
    wait_until(lambda: "not detected" in window.modality_label.text())
    assert window.start_run() is False
    assert _no_dialogs[-1][0] == "warning"


def test_window_run_with_auto_modality(window, tmp_path):
    image = _ct(tmp_path / "ct.nii.gz")
    gui_app._select(window.modality_combo, "auto")
    window.image_edit.setText(str(image))
    window._image_changed()
    assert window.start_run()  # detection finished or done synchronously
    wait_until(lambda: not window.process.is_running() and window.summary is not None)
    assert window.summary.status == "completed" and window.summary.modality == "ct"
