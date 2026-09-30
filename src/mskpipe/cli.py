# SPDX-License-Identifier: Apache-2.0
"""Command-line interface."""

from typing import Annotated

import typer

from mskpipe import __version__

app = typer.Typer(
    name="mskpipe",
    help="CT/MRI (NIfTI) -> Muscle Wrapping 2.x input (.osim, .xml, .obj).",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"mskpipe {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """mskpipe command-line interface."""
