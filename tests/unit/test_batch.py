# SPDX-License-Identifier: Apache-2.0
import csv
import json
import threading
from pathlib import Path

import pytest
from fake_pipeline import FakeStep, fake_steps, reset, write_nifti
from typer.testing import CliRunner

from mskpipe import api
from mskpipe import batch as b
from mskpipe.cli import app

cli = CliRunner()


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    reset()
    monkeypatch.setattr(api, "default_steps", lambda registry=None: fake_steps())
    monkeypatch.setattr(api, "preflight", lambda prepared, registry=None: [])


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    write_nifti(tmp_path / "data" / "a.nii.gz")
    write_nifti(tmp_path / "data" / "b.nii.gz")
    (tmp_path / "cfg.yaml").write_text("skeleton:\n  side: r\n", encoding="utf-8")
    return tmp_path


def write(folder: Path, text: str, name: str = "batch.yaml") -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


BATCH = """
version: 1
experiment: E99
repeat: 2
defaults:
  config: cfg.yaml
  device: cpu
  set: [runtime.cache=false, mesh.bones.smooth_iterations=10]
jobs:
  - name: lhdl
    image: data/a.nii.gz
    modality: ct
    set: [mesh.bones.smooth_iterations=20]
    matrix:
      attachments.params.nonrigid: [coherent_icp, cpd]
  - image: data/b.nii.gz
    modality: ct
    subject: B
    repeat: 1
"""


# ---------------------------------------------------------------------------- loading


def test_expand_yaml(folder):
    spec, base = b.load_batch(write(folder, BATCH))
    planned = b.expand(spec, base, overrides=["export.num_of_lines=10"])
    assert [(p.job, p.variant, p.repeat) for p in planned] == [
        ("lhdl", "nonrigid=coherent_icp", 1),
        ("lhdl", "nonrigid=cpd", 1),
        ("B", "", 1),
        ("lhdl", "nonrigid=coherent_icp", 2),
        ("lhdl", "nonrigid=cpd", 2),
    ]
    assert [p.index for p in planned] == [1, 2, 3, 4, 5]
    req = planned[1].request
    assert req.image == folder / "data" / "a.nii.gz"
    assert req.config == folder / "cfg.yaml"
    assert req.overrides == (
        "runtime.cache=false",
        "mesh.bones.smooth_iterations=10",
        "mesh.bones.smooth_iterations=20",
        "attachments.params.nonrigid=cpd",
        "export.num_of_lines=10",
    )
    assert req.device == "cpu"
    prepared = api.prepare_run(req)
    assert prepared.config.mesh.bones.smooth_iterations == 20
    assert prepared.config.attachments.params["nonrigid"] == "cpd"
    assert b.expand(spec, base, device="gpu")[0].request.device == "gpu"


def test_job_names_unique(folder):
    text = "jobs:\n" + "  - {image: data/a.nii.gz, modality: ct}\n" * 2
    spec, base = b.load_batch(write(folder, text))
    assert [p.job for p in b.expand(spec, base)] == ["a", "a-2"]


def test_csv(folder):
    path = write(
        folder,
        "name,image,modality,subject,set,repeat\n"
        "one,data/a.nii.gz,ct,S1,runtime.cache=false; skeleton.side=l,2\n"
        ",data/b.nii.gz,mri,,,\n",
        "batch.csv",
    )
    spec, base = b.load_batch(path)
    planned = b.expand(spec, base)
    assert [(p.job, p.repeat) for p in planned] == [("one", 1), ("b", 1), ("one", 2)]
    assert planned[0].request.overrides == ("runtime.cache=false", "skeleton.side=l")
    assert planned[1].request.modality == "mri"
    bad = write(folder, "image,modality,colour\ndata/a.nii.gz,ct,red\n", "bad.csv")
    with pytest.raises(api.SetupError, match="colour"):
        b.load_batch(bad)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("jobs: []", "jobs"),
        ("jobs:\n  - {image: a.nii.gz, modality: xray}", "modality"),
        ("jobs:\n  - {image: a.nii.gz, modality: ct, matrix: {a.b: []}}", "no values"),
        ("jobs:\n  - {image: a.nii.gz, modality: ct, colour: red}", "colour"),
        ("- just a list", "mapping"),
        ("jobs: [", "Invalid batch file"),
    ],
)
def test_invalid_files(folder, text, message):
    with pytest.raises(api.SetupError, match=message):
        b.load_batch(write(folder, text))
    with pytest.raises(api.SetupError, match="Cannot read"):
        b.load_batch(folder / "missing.yaml")


# ---------------------------------------------------------------------------- running


def plan(folder: Path, text: str = BATCH) -> list[b.PlannedRun]:
    spec, base = b.load_batch(write(folder, text))
    return b.expand(spec, base, runs_dir=folder / "runs")


def test_run_batch(folder, monkeypatch):
    from mskpipe import batch as module

    calls = []
    real = module.resolve_device
    monkeypatch.setattr(module, "resolve_device", lambda d: calls.append(d) or real(d))
    events = []
    out = folder / "runs" / "batches" / "x"
    result = b.run_batch(
        plan(folder), out, experiment="E99", on_event=lambda e, ev: events.append((e.index, ev))
    )
    assert result.exit_code == 0 and result.index.status == "completed"
    assert calls == ["cpu"]  # one GPU detection for the whole batch
    assert {e.status for e in result.entries} == {"completed"}
    assert all(Path(e.run_dir).is_dir() for e in result.entries)
    assert any(ev is not None and ev.kind == "run_started" for _, ev in events)

    index = b.load_batch_index(out)
    assert index.experiment == "E99" and len(index.entries) == 5
    assert index.entries[1].matrix == {"attachments.params.nonrigid": "cpd"}
    with (out / "summary.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["status"] for r in rows] == ["completed"] * 5
    assert rows[0]["variant"] == "nonrigid=coherent_icp"


def test_failure_continues_or_stops(folder):
    FakeStep.fail = {"skeleton"}
    result = b.run_batch(plan(folder), folder / "out1")
    assert {e.status for e in result.entries} == {"failed"}
    assert result.exit_code == 1 and result.index.status == "failed"
    stopped = b.run_batch(plan(folder), folder / "out2", stop_on_error=True)
    assert [e.status for e in stopped.entries] == ["failed"] + ["skipped"] * 4


def test_setup_error_recorded(folder):
    text = BATCH.replace("data/b.nii.gz", "data/missing.nii.gz")
    result = b.run_batch(plan(folder, text), folder / "out")
    assert [e.status for e in result.entries][2] == "setup_error"
    assert "not found" in result.entries[2].error
    assert result.entries[3].status == "completed"


def test_cancel_skips_rest(folder):
    FakeStep.cancel_in = {"mesh"}
    result = b.run_batch(plan(folder), folder / "out", cancel=threading.Event())
    assert [e.status for e in result.entries] == ["interrupted"] + ["skipped"] * 4
    assert result.exit_code == 130 and result.index.status == "interrupted"


def test_check_batch_dedupes(folder, monkeypatch):
    issue = api.Issue("error", "attachments", "no atlas")
    monkeypatch.setattr(api, "preflight", lambda prepared, registry=None: [issue])
    found = b.check_batch(plan(folder))
    assert [(p.index, i.message) for p, i in found] == [(1, "no atlas")]


def test_batch_folder_unique(tmp_path):
    from datetime import datetime

    now = datetime(2026, 10, 8, 9, 30, 0)
    first = b.batch_folder(tmp_path, "E1", now)
    assert first == tmp_path / "batches" / "20261008-093000_E1"
    first.mkdir(parents=True)
    assert b.batch_folder(tmp_path, "E1", now).name == "20261008-093000_E1-1"
    assert b.batch_folder(tmp_path, None, now).name == "20261008-093000_batch"


# ---------------------------------------------------------------------------- CLI


def test_cli_dry_run_and_run(folder):
    path = write(folder, BATCH)
    dry = cli.invoke(app, ["batch", str(path), "--dry-run", "-o", str(folder / "runs")])
    assert dry.exit_code == 0, dry.output
    assert "5 run(s), experiment: E99" in dry.stdout
    assert not (folder / "runs").exists()

    out = folder / "bat"
    result = cli.invoke(app, ["batch", str(path), "-o", str(folder / "runs"), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "=== [5/5] completed" in result.stdout
    assert (out / "batch.json").is_file()

    as_json = cli.invoke(app, ["batch", str(path), "-o", str(folder / "runs"), "--json", "-q"])
    entries = json.loads(as_json.stdout)
    assert len(entries) == 5 and entries[0]["status"] == "completed"


def test_cli_errors(folder, monkeypatch):
    assert cli.invoke(app, ["batch", str(folder / "nope.yaml")]).exit_code == 2
    path = write(folder, BATCH)
    issue = api.Issue("error", "input", "bad header")
    monkeypatch.setattr(api, "preflight", lambda prepared, registry=None: [issue])
    blocked = cli.invoke(app, ["batch", str(path), "-o", str(folder / "runs")])
    assert blocked.exit_code == 2 and "bad header" in blocked.output
    FakeStep.fail = {"mesh"}
    forced = cli.invoke(app, ["batch", str(path), "-o", str(folder / "runs"), "--no-preflight"])
    assert forced.exit_code == 1


def test_cli_warns_about_cache_with_repeats(folder):
    text = "repeat: 2\njobs:\n  - {image: data/a.nii.gz, modality: ct}\n"
    dry = cli.invoke(app, ["batch", str(write(folder, text)), "--dry-run"])
    assert dry.exit_code == 0 and "runtime.cache=true" in dry.output
    quiet = cli.invoke(app, ["batch", str(write(folder, text)), "--dry-run", "--no-cache"])
    assert "runtime.cache=true" not in quiet.output


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).parents[2] / "paper" / "experiments").glob("*.yaml"))
)
def test_example_batch_files(path):
    spec, base = b.load_batch(path)
    planned = b.expand(spec, base)
    assert planned and spec.experiment
