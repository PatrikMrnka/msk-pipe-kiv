# SPDX-License-Identifier: Apache-2.0
import json
import threading
from pathlib import Path

import pytest
from fake_pipeline import FakeStep, fake_steps, reset, write_nifti

from mskpipe import api
from mskpipe.core.manifest import Manifest
from mskpipe.core.registry import Registry
from mskpipe.plugins.base import AttachmentsPlugin, SkeletonPlugin


@pytest.fixture(autouse=True)
def _reset():
    reset()


@pytest.fixture
def image(tmp_path: Path) -> Path:
    return write_nifti(tmp_path / "data" / "s01.nii.gz")


def request(image: Path, tmp_path: Path, **kw) -> api.RunRequest:
    return api.RunRequest(image=image, modality="ct", runs_dir=tmp_path / "runs", **kw)


# ---------------------------------------------------------------------------- request


def test_request_shortcuts_override_set():
    req = api.RunRequest(
        image=Path("a.nii.gz"),
        modality="ct",
        overrides=("runtime.device=cpu", "mesh.bones.smooth_iterations=5"),
        device="gpu",
        runs_dir=Path("out/runs"),
        cache=False,
    )
    assert req.all_overrides()[-3:] == [
        "runtime.device=gpu",
        "runtime.runs_dir=out/runs",
        "runtime.cache=false",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "required"),
        ({"image": Path("a.nii.gz")}, "required"),
        ({"resume": Path("r"), "image": Path("a.nii.gz")}, "keeps its input"),
        ({"resume": Path("r"), "overrides": ("a.b=1",)}, "keeps its input"),
        ({"resume": Path("r"), "from_step": "nope"}, "not one of"),
        ({"resume": Path("r"), "from_step": "mesh", "until_step": "segment"}, "after"),
        ({"resume": Path("r"), "device": "tpu"}, "cpu, gpu or auto"),
    ],
)
def test_request_check(kwargs, message):
    with pytest.raises(api.SetupError, match=message):
        api.RunRequest(**kwargs).check()


def test_prepare_new(image, tmp_path):
    prepared = api.prepare_run(request(image, tmp_path, until_step="mesh", subject="S1"))
    assert prepared.steps == ("segment", "labelmap", "mesh")
    assert prepared.spec.subject_id == "S1"
    # plugin defaults are filled in (config.resolved.yaml records every parameter)
    assert "nonrigid" in prepared.config.attachments.params
    assert prepared.workspace is None


def test_prepare_errors(image, tmp_path):
    with pytest.raises(api.SetupError, match="not found"):
        api.prepare_run(request(tmp_path / "missing.nii.gz", tmp_path))
    with pytest.raises(api.SetupError, match=r"skeleton\.side"):
        api.prepare_run(request(image, tmp_path, overrides=("skeleton.side=x",)))
    bad = write_nifti(tmp_path / "bad name!.nii.gz")
    with pytest.raises(api.SetupError, match="subject_id"):
        api.prepare_run(request(bad, tmp_path, subject="bad name!"))
    with pytest.raises(api.SetupError, match="Cannot resume"):
        api.prepare_run(api.RunRequest(resume=tmp_path))


# ---------------------------------------------------------------------------- run


def test_run_completed_with_events(image, tmp_path):
    events: list[api.RunEvent] = []
    summary = api.run(request(image, tmp_path), on_event=events.append, steps=fake_steps())

    assert summary.status == "completed" and summary.exit_code == 0
    assert [s.name for s in summary.steps] == list(api.PIPELINE_STEPS)
    assert summary.mw2_dir == summary.run_dir / "06_mw2_input"
    assert summary.device == "cpu" and summary.subject == "s01"
    kinds = [e.kind for e in events]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished"
    assert events[0].run_dir == str(summary.run_dir)
    assert events[0].steps == list(api.PIPELINE_STEPS)
    done = [e for e in events if e.kind == "step" and e.status == "completed"]
    assert [e.step for e in done] == list(api.PIPELINE_STEPS)
    assert all(e.wall_s is not None for e in done)
    assert any(e.kind == "log" and "fake work" in (e.message or "") for e in events)
    assert events[-1].exit_code == 0
    # events round-trip through JSON lines (GUI <-> CLI)
    assert api.RunEvent.from_line(events[-1].to_line()) == events[-1]


def test_run_failure_is_reported_not_raised(image, tmp_path):
    FakeStep.fail = {"mesh"}
    summary = api.run(request(image, tmp_path), steps=fake_steps())
    assert summary.status == "failed" and summary.exit_code == 1
    assert summary.failed_step == "mesh"
    assert "mesh broken" in summary.error
    assert summary.mw2_dir is None
    assert "skeleton" not in FakeStep.calls


def test_run_cancel(image, tmp_path):
    FakeStep.cancel_in = {"labelmap"}
    events = []
    summary = api.run(
        request(image, tmp_path),
        steps=fake_steps(),
        cancel=threading.Event(),
        on_event=events.append,
    )
    assert summary.status == "interrupted" and summary.exit_code == 130
    assert summary.failed_step == "labelmap"
    assert events[-1].status == "interrupted"


def test_cancel_before_start(image, tmp_path):
    cancel = threading.Event()
    cancel.set()
    summary = api.run(request(image, tmp_path), steps=fake_steps(), cancel=cancel)
    assert summary.status == "interrupted"
    assert FakeStep.calls == {}


def test_resume_from_and_device(image, tmp_path):
    first = api.run(request(image, tmp_path, until_step="mesh"), steps=fake_steps())
    assert [s.name for s in first.steps] == ["segment", "labelmap", "mesh"]

    again = api.run(
        api.RunRequest(resume=first.run_dir, from_step="mesh", device="cpu"), steps=fake_steps()
    )
    assert again.run_dir == first.run_dir and again.status == "completed"
    assert FakeStep.calls["segment"] == 1  # up to date, not re-run
    assert FakeStep.calls["mesh"] == 2
    assert {s.name: s.status for s in again.steps}["segment"] == "completed"


def test_cache_between_runs_and_no_cache(image, tmp_path):
    api.run(request(image, tmp_path), steps=fake_steps())
    cached = api.run(request(image, tmp_path), steps=fake_steps())
    assert {s.status for s in cached.steps} == {"cached"}
    assert cached.executed_wall_s == 0
    assert cached.mw2_dir is not None
    fresh = api.run(request(image, tmp_path, cache=False), steps=fake_steps())
    assert {s.status for s in fresh.steps} == {"completed"}


def test_unusable_gpu_is_setup_error(image, tmp_path, monkeypatch):
    from mskpipe.core import runner
    from mskpipe.core.device import DeviceError

    def no_gpu(requested):
        raise DeviceError("no GPU")

    monkeypatch.setattr(runner, "resolve_device", no_gpu)
    with pytest.raises(api.SetupError, match="no GPU"):
        api.run(request(image, tmp_path, device="gpu"), steps=fake_steps())
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


# ---------------------------------------------------------------------------- summaries


def test_summarize_and_finalize_stale(image, tmp_path):
    summary = api.run(request(image, tmp_path), steps=fake_steps())
    path = summary.run_dir / "manifest.json"
    m = Manifest.load(path)
    m.status = "running"
    m.steps["export_mw2"].status = "running"
    m.save()
    assert api.summarize(summary.run_dir).status == "running"

    assert api.finalize_stale(summary.run_dir) is True
    after = api.summarize(summary.run_dir)
    assert after.status == "interrupted" and after.failed_step == "export_mw2"
    assert after.error == "process terminated"
    assert after.mw2_dir is None
    assert api.finalize_stale(summary.run_dir) is False


# ---------------------------------------------------------------------------- preflight


class _Skel(SkeletonPlugin):
    name = "pystaple"

    def build(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


class _Att(AttachmentsPlugin):
    name = "bone_registration"

    @classmethod
    def preflight(cls, config, modality):
        return ["no atlas"]

    def compute(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


class _Missing(_Skel):
    requires_modules = ("module_that_does_not_exist_xyz",)


def _registry(skeleton=_Skel) -> Registry:
    reg = Registry()
    reg.register(skeleton)
    reg.register(_Att)
    return reg


def test_preflight_plugins(image, tmp_path):
    reg = _registry()
    prepared = api.prepare_run(request(image, tmp_path, from_step="skeleton"))
    issues = api.preflight(prepared, reg)
    assert [(i.severity, i.where, i.message) for i in issues] == [
        ("error", "attachments", "no atlas")
    ]
    only_mesh = api.prepare_run(request(image, tmp_path, from_step="mesh", until_step="mesh"))
    assert api.preflight(only_mesh, reg) == []
    missing = api.preflight(prepared, _registry(_Missing))
    assert any("module_that_does_not_exist_xyz" in i.message for i in missing)


def test_preflight_segmenter_unknown(image, tmp_path):
    prepared = api.prepare_run(request(image, tmp_path, until_step="segment"))
    issues = api.preflight(prepared, Registry())  # no segmenters registered
    assert {i.where for i in issues} == {"segmentation"}
    assert all(i.severity == "error" for i in issues)


def test_preflight_image_header(tmp_path):
    ok = write_nifti(tmp_path / "ok.nii.gz")
    issues = api._check_image(ok)
    assert issues == []
    unit = write_nifti(tmp_path / "unit.nii.gz", spacing=(1.0, 1.0, 1.0), qform=False)
    messages = " ".join(i.message for i in api._check_image(unit))
    assert "qform/sform" in messages and "1 x 1 x 1" in messages
    broken = tmp_path / "broken.nii.gz"
    broken.write_bytes(b"not a nifti")
    assert api._check_image(broken)[0].severity == "error"


def test_totalsegmentator_licence(tmp_path, monkeypatch):
    from mskpipe.config import load_config
    from mskpipe.plugins.segmenters.totalsegmentator import TotalSegmentator

    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path))
    config = load_config()
    assert "licence key" in TotalSegmentator.preflight(config, "ct")[0]
    (tmp_path / "config.json").write_text(json.dumps({"license_number": "short"}), "utf-8")
    assert "Invalid" in TotalSegmentator.preflight(config, "ct")[0]
    (tmp_path / "config.json").write_text(json.dumps({"license_number": "a" * 18}), "utf-8")
    assert TotalSegmentator.preflight(config, "ct") == []
    no_key = load_config(overrides=["segmentation.tibia_fibula=musclemap"])
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path / "empty"))
    assert TotalSegmentator.preflight(no_key, "ct") == []


def test_bone_registration_preflight(tmp_path, monkeypatch):
    from mskpipe.config import load_config
    from mskpipe.plugins.attachments.bone_registration import BoneRegistration

    monkeypatch.delenv("MSKPIPE_ATLAS_DIR", raising=False)
    assert "No attachment atlas" in BoneRegistration.preflight(load_config(), "ct")[0]
    cfg = load_config(overrides=[f"attachments.params.atlas_dir={tmp_path.as_posix()}"])
    assert "index not found" in BoneRegistration.preflight(cfg, "ct")[0]
    (tmp_path / "atlas.json").write_text("{}", encoding="utf-8")
    assert BoneRegistration.preflight(cfg, "ct") == []


# ---------------------------------------------------------------------------- config


def test_config_helpers():
    assert "runtime:" in api.config_template()
    assert api.config_schema()["title"] == "PipelineConfig"
    text = api.config_yaml(api.resolved_config(overrides=["skeleton.side=l"]))
    assert "side: l" in text


def test_api_and_cli_import_no_heavy_modules():
    import subprocess
    import sys

    code = (
        "import sys, mskpipe.api, mskpipe.cli; "
        "print(','.join(m for m in ('vtk', 'torch', 'nibabel', 'SimpleITK', 'pystaple', "
        "'PySide6', 'scipy') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
