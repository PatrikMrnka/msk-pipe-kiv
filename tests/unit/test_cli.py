# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from typer.testing import CliRunner

from mskpipe import __version__
from mskpipe.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"mskpipe {__version__}"


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_plugins_lists_builtins() -> None:
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0
    assert "bone_registration" in result.stdout and "totalsegmentator" in result.stdout


def test_device_cpu() -> None:
    result = runner.invoke(app, ["device", "--device", "cpu"])
    assert result.exit_code == 0
    assert "Selected: cpu - requested" in result.stdout


def test_device_json_and_bad_value() -> None:
    assert '"requested": "cpu"' in runner.invoke(app, ["device", "-d", "cpu", "--json"]).stdout
    assert runner.invoke(app, ["device", "-d", "tpu"]).exit_code == 1


# ---------------------------------------------------------------------------- run / config


@pytest.fixture
def fake(monkeypatch):
    from fake_pipeline import fake_steps, reset

    from mskpipe import api

    reset()
    monkeypatch.setattr(api, "default_steps", lambda registry=None: fake_steps())
    monkeypatch.setattr(api, "preflight", lambda prepared, registry=None: [])
    return api


@pytest.fixture
def image(tmp_path):
    from fake_pipeline import write_nifti

    return write_nifti(tmp_path / "s01.nii.gz")


def _run(tmp_path, *args):
    return runner.invoke(app, ["run", *args, "--runs-dir", str(tmp_path / "runs")])


def test_run_completed(fake, image, tmp_path):
    result = _run(tmp_path, str(image), "-m", "ct", "-d", "cpu")
    assert result.exit_code == 0, result.output
    assert "Status: completed" in result.stdout
    assert "Muscle Wrapping input:" in result.stdout
    assert "export_mw2" in result.stdout


def test_run_json_and_failure(fake, image, tmp_path):
    from fake_pipeline import FakeStep

    FakeStep.fail = {"skeleton"}
    result = _run(tmp_path, str(image), "-m", "ct", "--json", "-q")
    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert summary["status"] == "failed" and summary["failed_step"] == "skeleton"


def test_run_until_then_resume(fake, image, tmp_path):
    first = _run(tmp_path, str(image), "-m", "ct", "--until", "mesh", "--json", "-q")
    run_dir = json.loads(first.stdout)["run_dir"]
    result = runner.invoke(app, ["run", "--resume", run_dir, "--from", "skeleton", "--json", "-q"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["mw2_dir"]


def test_run_setup_errors(fake, image, tmp_path):
    assert _run(tmp_path, str(image)).exit_code == 2  # no modality
    assert _run(tmp_path, str(image), "-m", "ct", "--set", "skeleton.side=x").exit_code == 2
    assert _run(tmp_path, str(tmp_path / "nope.nii.gz"), "-m", "ct").exit_code == 2
    resumed = runner.invoke(app, ["run", "--resume", str(tmp_path), "-m", "ct"])
    assert resumed.exit_code == 2


def test_run_preflight_blocks(fake, image, tmp_path, monkeypatch):
    issue = fake.Issue("error", "attachments", "no atlas")
    monkeypatch.setattr(fake, "preflight", lambda prepared, registry=None: [issue])
    blocked = _run(tmp_path, str(image), "-m", "ct")
    assert blocked.exit_code == 2
    assert "no atlas" in blocked.output
    forced = _run(tmp_path, str(image), "-m", "ct", "--no-preflight")
    assert forced.exit_code == 0


def test_config_init_show_schema(tmp_path):
    target = tmp_path / "cfg" / "my.yaml"
    assert runner.invoke(app, ["config", "init", str(target)]).exit_code == 0
    assert target.read_text(encoding="utf-8").startswith("# msk-pipe configuration")
    assert runner.invoke(app, ["config", "init", str(target)]).exit_code == 1
    assert runner.invoke(app, ["config", "init", str(target), "--force"]).exit_code == 0

    shown = runner.invoke(app, ["config", "show", "-c", str(target), "--set", "skeleton.side=l"])
    assert shown.exit_code == 0 and "side: l" in shown.stdout
    as_json = runner.invoke(app, ["config", "show", "--json"])
    assert json.loads(as_json.stdout)["skeleton"]["side"] == "r"
    assert runner.invoke(app, ["config", "show", "--set", "nope=1"]).exit_code == 2

    schema = runner.invoke(app, ["config", "schema"])
    assert json.loads(schema.stdout)["title"] == "PipelineConfig"
    out = tmp_path / "schema.json"
    assert runner.invoke(app, ["config", "schema", "-o", str(out)]).exit_code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["type"] == "object"
