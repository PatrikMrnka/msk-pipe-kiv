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
