# SPDX-License-Identifier: Apache-2.0
import sys
import threading
from pathlib import Path

import pytest

from mskpipe.config import InputSpec, load_config
from mskpipe.core.manifest import Manifest
from mskpipe.core.runner import PipelineCancelled, PipelineError, run_pipeline, step_fingerprint
from mskpipe.core.step import Step, StepContext, StepError
from mskpipe.core.workspace import STEP_DIRS


class Recorder(Step):
    """Test step: writes one file, reads the upstream one, counts executions."""

    calls: dict[str, int]
    fail: set[str]

    def __init__(self, name: str, sections: tuple[str, ...], upstream: str | None) -> None:
        self.name = name  # type: ignore[misc]
        self.config_sections = sections  # type: ignore[misc]
        self.upstream = upstream

    def run(self, ctx: StepContext) -> None:
        Recorder.calls[self.name] = Recorder.calls.get(self.name, 0) + 1
        if self.name in Recorder.fail:
            raise StepError(f"{self.name} broken")
        if self.upstream:
            assert (ctx.step_dir(self.upstream) / f"{self.upstream}.txt").is_file()
        out = ctx.out_dir / f"{self.name}.txt"
        out.write_text(self.name, encoding="utf-8")
        ctx.record.add_output(out)
        ctx.record.metrics["value"] = 1


SECTIONS = {
    "prepare": (),
    "segment": ("segmentation",),
    "labelmap": ("labelmap",),
    "mesh": ("mesh",),
    "skeleton": ("skeleton",),
    "attachments": ("attachments",),
    "export_mw2": ("export",),
}


def make_steps() -> list[Step]:
    names = list(STEP_DIRS)
    return [Recorder(n, SECTIONS[n], names[i - 1] if i else None) for i, n in enumerate(names)]


@pytest.fixture(autouse=True)
def _reset():
    Recorder.calls = {}
    Recorder.fail = set()


@pytest.fixture
def runs(tmp_path: Path) -> str:
    return f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}"


@pytest.fixture
def spec(tmp_path: Path) -> InputSpec:
    image = tmp_path / "s01.nii.gz"
    image.write_bytes(b"volume")
    return InputSpec(image=image, modality="ct")


def statuses(result) -> dict[str, str]:
    return {k: v.status for k, v in result.manifest.steps.items()}


def test_full_run(spec, runs):
    events = []
    result = run_pipeline(
        make_steps(), spec, load_config(overrides=[runs]), progress=lambda s, st: events.append(st)
    )
    assert result.manifest.status == "completed"
    assert set(statuses(result).values()) == {"completed"}
    assert events.count("completed") == len(STEP_DIRS)
    assert (result.ws.step_dir("export_mw2") / "export_mw2.txt").is_file()
    assert (result.ws.logs_dir / "run.log").read_text(encoding="utf-8").count("completed") >= 7


def test_second_run_reuses_everything(spec, runs):
    cfg = load_config(overrides=[runs])
    first = run_pipeline(make_steps(), spec, cfg)
    second = run_pipeline(make_steps(), spec, cfg)
    assert set(statuses(second).values()) == {"cached"}
    assert all(v == 1 for v in Recorder.calls.values())
    rec = second.manifest.steps["mesh"]
    assert rec.metrics["cached_from"] == first.ws.root.name
    assert rec.outputs == ["03_mesh/mesh.txt"]
    assert (second.ws.step_dir("mesh") / "mesh.txt").is_file()
    assert second.ws.input_image.is_file()


def test_cache_disabled(spec, runs):
    run_pipeline(make_steps(), spec, load_config(overrides=[runs]))
    second = run_pipeline(make_steps(), spec, load_config(overrides=[runs, "runtime.cache=false"]))
    assert set(statuses(second).values()) == {"completed"}


def test_config_change_invalidates_only_downstream(spec, runs):
    run_pipeline(make_steps(), spec, load_config(overrides=[runs]))
    changed = run_pipeline(
        make_steps(), spec, load_config(overrides=[runs, "mesh.bones.smooth_iterations=3"])
    )
    st = statuses(changed)
    assert [st[n] for n in ("prepare", "segment", "labelmap")] == ["cached"] * 3
    assert [st[n] for n in ("mesh", "skeleton", "attachments", "export_mw2")] == ["completed"] * 4


def test_failure_then_resume(spec, runs):
    Recorder.fail = {"skeleton"}
    with pytest.raises(PipelineError, match="skeleton") as info:
        run_pipeline(make_steps(), spec, load_config(overrides=[runs]))
    ws = info.value.ws
    failed = Manifest.load(ws.manifest_path)
    assert failed.status == "failed"
    assert failed.steps["skeleton"].status == "failed"
    assert "attachments" not in failed.steps

    Recorder.fail = set()
    resumed = run_pipeline(make_steps(), resume=ws.root)
    assert resumed.ws.root == ws.root
    assert resumed.manifest.status == "completed"
    assert Recorder.calls["mesh"] == 1 and Recorder.calls["skeleton"] == 2


def test_from_and_until(spec, runs):
    first = run_pipeline(make_steps(), spec, load_config(overrides=[runs]), until_step="mesh")
    assert list(first.manifest.steps) == ["prepare", "segment", "labelmap", "mesh"]
    again = run_pipeline(make_steps(), resume=first.ws.root, from_step="labelmap")
    st = statuses(again)
    assert st["segment"] == "completed"  # original record kept
    assert Recorder.calls["segment"] == 1 and Recorder.calls["labelmap"] == 2
    assert st["export_mw2"] == "completed"


def test_from_without_upstream_outputs(spec, runs):
    with pytest.raises(PipelineError, match="no reusable outputs"):
        run_pipeline(
            make_steps(),
            spec,
            load_config(overrides=[runs, "runtime.cache=false"]),
            from_step="mesh",
        )


def test_rerun_clears_downstream(spec, runs):
    first = run_pipeline(make_steps(), spec, load_config(overrides=[runs]))
    stale = first.ws.step_dir("export_mw2") / "stale.obj"
    stale.write_text("x", encoding="utf-8")
    run_pipeline(make_steps(), resume=first.ws.root, from_step="mesh")
    assert not stale.exists()
    assert first.ws.input_image.is_file()


def test_cancel(spec, runs):
    cancel = threading.Event()

    def progress(step: str, status: str) -> None:
        if step == "segment" and status == "completed":
            cancel.set()

    with pytest.raises(PipelineCancelled) as info:
        run_pipeline(
            make_steps(), spec, load_config(overrides=[runs]), progress=progress, cancel=cancel
        )
    manifest = Manifest.load(info.value.ws.manifest_path)
    assert manifest.status == "interrupted"
    assert list(manifest.steps) == ["prepare", "segment"]


def test_invalid_step_order(spec, runs):
    steps = make_steps()
    with pytest.raises(ValueError, match="ordered"):
        run_pipeline([steps[1], steps[0]], spec, load_config(overrides=[runs]))


def test_fingerprint_chain(spec, runs):
    cfg = load_config(overrides=[runs])
    steps = make_steps()
    a = step_fingerprint(steps[3], cfg, "up1")
    assert a == step_fingerprint(steps[3], cfg, "up1")
    assert a != step_fingerprint(steps[3], cfg, "up2")


class Shell(Step):
    name = "prepare"

    def __init__(self, code: str) -> None:
        self.code = code

    def run(self, ctx: StepContext) -> None:
        ctx.run_command([sys.executable, "-c", self.code])


def test_run_command_logs_output(spec, runs):
    result = run_pipeline([Shell("print('hello from tool')")], spec, load_config(overrides=[runs]))
    log = (result.ws.logs_dir / "prepare.log").read_text(encoding="utf-8")
    assert "hello from tool" in log


def test_run_command_failure(spec, runs):
    with pytest.raises(PipelineError, match="exited with code 3") as info:
        run_pipeline([Shell("import sys; sys.exit(3)")], spec, load_config(overrides=[runs]))
    assert isinstance(info.value.__cause__, StepError)
