# SPDX-License-Identifier: Apache-2.0
import csv
from pathlib import Path

import pytest
from fake_pipeline import FakeStep, fake_steps, reset, write_nifti
from typer.testing import CliRunner

from mskpipe import api
from mskpipe import batch as b
from mskpipe import stats as s
from mskpipe.cli import app

cli = CliRunner()

BATCH = """
experiment: E50
repeat: 2
defaults: {set: [runtime.cache=false]}
jobs:
  - name: ct
    image: a.nii.gz
    modality: ct
    matrix: {attachments.params.nonrigid: [coherent_icp, cpd]}
"""


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    reset()
    monkeypatch.setattr(api, "default_steps", lambda registry=None: fake_steps())
    monkeypatch.setattr(api, "preflight", lambda prepared, registry=None: [])


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    """Batch of 4 runs + one stand-alone run with a cached segmentation."""
    write_nifti(tmp_path / "a.nii.gz")
    (tmp_path / "batch.yaml").write_text(BATCH, encoding="utf-8")
    spec, base = b.load_batch(tmp_path / "batch.yaml")
    runs = tmp_path / "runs"
    planned = b.expand(spec, base, runs_dir=runs)
    b.run_batch(planned, b.batch_folder(runs, spec.experiment), experiment=spec.experiment)
    api.run(api.RunRequest(image=tmp_path / "a.nii.gz", modality="ct", runs_dir=runs))
    return runs


def test_collect(runs):
    tables = s.collect([runs])
    assert len(tables["runs"]) == 5 and tables.skipped == []

    batch_runs = [r for r in tables["runs"] if r["experiment"] == "E50"]
    assert len(batch_runs) == 4
    first = batch_runs[0]
    assert first["status"] == "completed" and first["device"] == "cpu"
    assert first["job"] == "ct" and first["repeat"] in (1, 2)
    assert first["pipeline_wall_s"] is not None and first["n_steps_completed"] == 6
    assert first["n_muscles_selected"] == 2 and first["n_muscles_exported"] == 1
    assert first["n_areas_inflated"] == 1 and first["n_body_mismatch"] == 1
    assert first["cache"] is False
    diffs = {r["variant"]: r["config_diff"] for r in batch_runs}
    assert diffs["nonrigid=coherent_icp"] == ""
    assert diffs["nonrigid=cpd"] == 'attachments.params.nonrigid="cpd"'

    alone = next(r for r in tables["runs"] if r["experiment"] is None)
    assert alone["n_steps_cached"] == 6 and alone["pipeline_wall_s"] is None
    cached = [r for r in tables["steps"] if r["run_id"] == alone["run_id"]]
    assert {r["status"] for r in cached} == {"cached"}
    assert all(r["wall_s"] is None and r["cached_from"] for r in cached)

    seg = [r for r in tables["steps"] if r["step"] == "segment" and r["status"] == "completed"]
    assert seg[0]["step_time_s"] == 3.0
    export = [r for r in tables["steps"] if r["step"] == "export_mw2"]
    assert export[0]["step_time_s"] == 0.5

    tools = tables["tools"]
    assert {t["tool"] for t in tools} == {"totalsegmentator", "musclemap"}
    ts = next(t for t in tools if t["tool"] == "totalsegmentator")
    assert ts["tasks"] == "appendicular_bones;total" and ts["empty"] == "vertebrae_S1"

    reg = tables["registration"]
    femur = next(r for r in reg if r["bone"] == "femur" and r["run_id"] == first["run_id"])
    assert femur["nonrigid_mean_mm"] == 0.69 and femur["method"] in ("coherent_icp", "cpd")
    assert next(r for r in reg if r["bone"] == "pelvis")["nonrigid_mean_mm"] is None

    muscles = [m for m in tables["muscles"] if m["run_id"] == first["run_id"]]
    assert {m["muscle"]: m["status"] for m in muscles} == {
        "gluteus_medius_r": "ok",
        "piriformis_r": "excluded",
    }
    areas = [a for a in tables["areas"] if a["run_id"] == first["run_id"]]
    ins = next(a for a in areas if a["kind"] == "Ins")
    assert ins["inflated"] is True and ins["inflated_distinct"] == 18
    assert ins["body_mismatch"] is True and ins["gap_mean_mm"] == 97.0


def test_summary(runs):
    summary = s.collect([runs])["summary"]
    rows = [r for r in summary if r["experiment"] == "E50"]
    steps = [r["step"] for r in rows if r["variant"] == "nonrigid=cpd"]
    assert steps == [*api.PIPELINE_STEPS, "total"]
    seg = next(r for r in rows if r["step"] == "segment" and r["variant"] == "nonrigid=cpd")
    assert seg["n"] == 2 and seg["wall_sd_s"] is not None
    assert seg["wall_min_s"] <= seg["wall_mean_s"] <= seg["wall_max_s"]
    # the stand-alone run reused everything: no executed steps, not in the summary
    assert all(r["experiment"] == "E50" for r in summary)


def test_filters_and_sources(runs):
    batch_dir = next((runs / "batches").iterdir())
    assert len(s.collect([batch_dir])["runs"]) == 4
    assert len(s.collect([batch_dir / "batch.json"])["runs"]) == 4
    assert len(s.collect([runs], experiment="E50")["runs"]) == 4
    assert s.collect([runs], experiment="nope")["runs"] == []
    one = next(p for p in runs.iterdir() if (p / "manifest.json").is_file())
    assert len(s.collect([one])["runs"]) == 1


def test_failed_and_broken_runs(tmp_path):
    FakeStep.fail = {"mesh"}
    image = write_nifti(tmp_path / "a.nii.gz")
    runs = tmp_path / "runs"
    api.run(api.RunRequest(image=image, modality="ct", runs_dir=runs))
    broken = runs / "20260101-000000_x_00000000"
    broken.mkdir()
    (broken / "input.json").write_text("{}", encoding="utf-8")
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    tables = s.collect([runs])
    assert len(tables.skipped) == 1
    run = tables["runs"][0]
    assert run["status"] == "failed" and "mesh broken" in run["error"]
    assert run["n_muscles_selected"] is None and tables["muscles"] == []
    assert tables["summary"] and all(r["step"] != "total" for r in tables["summary"])


def test_config_diff():
    ref = {"a": {"b": 1, "c": [1, 2]}, "runtime": {"device": "cpu"}, "config_version": 1}
    cfg = {"a": {"b": 2, "c": [1, 2]}, "runtime": {"device": "gpu"}, "config_version": 1}
    assert s.config_diff(cfg, s._flatten(ref)) == "a.b=2"


def test_write_tables_and_cli(runs, tmp_path):
    out = tmp_path / "stats"
    written = s.write_tables(s.collect([runs]), out)
    assert {p.name for p in written} == {f"{n}.csv" for n in s.COLUMNS}
    with (out / "runs.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5 and rows[0]["cache"] in ("true", "false")
    assert list(rows[0]) == list(s.COLUMNS["runs"])

    result = cli.invoke(app, ["stats", str(runs), "-o", str(tmp_path / "cli"), "-e", "E50"])
    assert result.exit_code == 0, result.output
    assert "runs.csv (4 rows)" in result.stdout and "total" in result.stdout
    empty = cli.invoke(app, ["stats", str(tmp_path / "nothing"), "-o", str(tmp_path / "x")])
    assert empty.exit_code == 1
