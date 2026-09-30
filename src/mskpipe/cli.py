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


@app.command()
def device(
    requested: Annotated[
        str, typer.Option("--device", "-d", help="Device to test: cpu, gpu or auto.")
    ] = "auto",
    as_json: Annotated[bool, typer.Option("--json", help="Print the full report as JSON.")] = False,
) -> None:
    """Show detected NVIDIA GPUs and which device a run would use."""
    from mskpipe.core.device import DeviceError, resolve_device

    try:
        report = resolve_device(requested)
    except (DeviceError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    driver = report.driver_version or "not found"
    if report.driver_cuda:
        driver += f" (CUDA {report.driver_cuda})"
    typer.echo(f"NVIDIA driver: {driver}")
    for gpu in report.gpus:
        mem = f", {gpu.memory_total_bytes / 2**30:.1f} GiB" if gpu.memory_total_bytes else ""
        cc = f", sm {gpu.compute_capability}" if gpu.compute_capability else ""
        typer.echo(f"GPU {gpu.index}: {gpu.name}{mem}{cc}")
    if report.torch and report.torch.version:
        t = report.torch
        typer.echo(f"PyTorch: {t.version} (CUDA build {t.cuda or 'none'})")
    typer.echo(f"Selected: {report.summary()}")


@app.command()
def plugins() -> None:
    """List segmenters, skeleton backends and attachment methods."""
    from mskpipe.core.registry import default_registry, format_plugins

    typer.echo(format_plugins(default_registry().describe()))
