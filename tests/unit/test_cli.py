# SPDX-License-Identifier: Apache-2.0
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
