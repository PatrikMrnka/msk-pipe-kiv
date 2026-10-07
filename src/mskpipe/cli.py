# SPDX-License-Identifier: Apache-2.0
"""Command-line interface (``mskpipe``). Thin layer over :mod:`mskpipe.api`.

Exit codes: 0 completed, 1 a step failed, 2 invalid request/config/device (nothing run),
130 cancelled (Ctrl+C).
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from mskpipe import __version__
from mskpipe.config import Device, Modality
from mskpipe.core.workspace import STEPS

if TYPE_CHECKING:
    from mskpipe.api import RunSummary

StepName = StrEnum("StepName", {s: s for s in STEPS[1:]})  # type: ignore[misc]

app = typer.Typer(
    name="mskpipe",
    help="CT/MRI (NIfTI) -> Muscle Wrapping 2.x input (.osim, .xml, .obj).",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(
    help="Create, show and validate configuration files.", no_args_is_help=True
)
app.add_typer(config_app, name="config")

EXIT_SETUP = 2


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


# ---------------------------------------------------------------------------- run


@app.command()
def run(
    image: Annotated[
        Path | None,
        typer.Argument(
            help="Input volume (.nii/.nii.gz). Not used with --resume.", show_default=False
        ),
    ] = None,
    modality: Annotated[
        Modality | None, typer.Option("--modality", "-m", help="Modality of the input.")
    ] = None,
    subject: Annotated[
        str | None, typer.Option("--subject", "-s", help="Subject id (default: file name).")
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="YAML config (see `mskpipe config init`).")
    ] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", metavar="KEY=VALUE", help="Override a config value (repeatable)."),
    ] = None,
    device: Annotated[
        Device | None, typer.Option("--device", "-d", help="Compute device (runtime.device).")
    ] = None,
    runs_dir: Annotated[
        Path | None,
        typer.Option("--runs-dir", "-o", help="Folder for run folders (runtime.runs_dir)."),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Do not reuse results of earlier runs (timings)."),
    ] = False,
    from_step: Annotated[
        StepName | None, typer.Option("--from", help="Re-run from this step.")
    ] = None,
    until_step: Annotated[
        StepName | None, typer.Option("--until", help="Last step to run.")
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option("--resume", help="Continue an existing run folder (keeps its config)."),
    ] = None,
    no_preflight: Annotated[
        bool, typer.Option("--no-preflight", help="Start even if pre-run checks report errors.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Also print the output of external tools.")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only warnings and errors.")] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the run summary as JSON (stdout).")
    ] = False,
) -> None:
    """Run the pipeline on one volume: segmentation -> ... -> Muscle Wrapping 2.x input.

    \b
    Examples:
      mskpipe run ct.nii.gz -m ct -d gpu
      mskpipe run ct.nii.gz -m ct -c lhdl.yaml --set attachments.params.nonrigid=cpd
      mskpipe run --resume runs/<run> --from export_mw2
    """
    from mskpipe import api

    request = api.RunRequest(
        image=image,
        modality=modality,
        subject=subject,
        config=config,
        overrides=tuple(overrides or ()),
        device=device,
        runs_dir=runs_dir,
        cache=False if no_cache else None,
        from_step=from_step.value if from_step else None,
        until_step=until_step.value if until_step else None,
        resume=resume,
    )
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    try:
        prepared = api.prepare_run(request)
        if not _report_preflight(api.preflight(prepared), no_preflight):
            raise typer.Exit(EXIT_SETUP)
        with _console_log(level, sys.stderr if as_json else sys.stdout):
            summary = api.run(prepared)
    except api.SetupError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_SETUP) from None
    if as_json:
        typer.echo(summary.model_dump_json(indent=2))
    else:
        _print_summary(summary)
    raise typer.Exit(summary.exit_code)


def _report_preflight(issues: list, force: bool) -> bool:
    """Print issues; False = do not start."""
    errors = [i for i in issues if i.severity == "error"]
    for issue in issues:
        typer.echo(f"{issue.severity.upper()}: {issue.where}: {issue.message}", err=True)
    if errors and not force:
        typer.echo(
            f"{len(errors)} pre-run check(s) failed; fix them or use --no-preflight.", err=True
        )
        return False
    return True


@contextmanager
def _console_log(level: int, stream) -> Iterator[None]:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    handler.setLevel(level)
    log = logging.getLogger("mskpipe")
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        log.removeHandler(handler)


def _gb(value: int | None) -> str:
    return f"{value / 2**30:.2f}" if value else "-"


def _print_summary(summary: RunSummary) -> None:
    typer.echo(f"\nRun:    {summary.run_dir}")
    typer.echo(f"Status: {summary.status}")
    if summary.device:
        typer.echo(f"Device: {summary.device} - {summary.device_reason or ''}")
    typer.echo(f"{'step':<13}{'status':<12}{'wall s':>9}{'CPU s':>9}{'peak RAM GB':>13}")
    for s in summary.steps:
        wall = f"{s.wall_s:.1f}" if s.wall_s is not None and s.status == "completed" else "-"
        cpu = f"{s.cpu_s:.1f}" if s.cpu_s is not None and s.status == "completed" else "-"
        typer.echo(f"{s.name:<13}{s.status:<12}{wall:>9}{cpu:>9}{_gb(s.peak_rss_bytes):>13}")
    typer.echo(f"{'total':<25}{summary.executed_wall_s:>9.1f}")
    if summary.error:
        typer.echo(f"Error in '{summary.failed_step}': {summary.error}", err=True)
    if summary.mw2_dir:
        typer.echo(f"Muscle Wrapping input: {summary.mw2_dir}")


# ---------------------------------------------------------------------------- config


@config_app.command("init")
def config_init(
    path: Annotated[Path, typer.Argument(help="File to create; '-' prints to stdout.")] = Path(
        "mskpipe.yaml"
    ),
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite an existing file.")
    ] = False,
) -> None:
    """Write a commented config file with every key and its default."""
    from mskpipe import api

    text = api.config_template()
    if str(path) == "-":
        typer.echo(text, nl=False)
        return
    if path.exists() and not force:
        typer.echo(f"Error: {path} exists (use --force to overwrite)", err=True)
        raise typer.Exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    typer.echo(f"Written {path}")


@config_app.command("show")
def config_show(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML config.")] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", metavar="KEY=VALUE", help="Override a config value (repeatable)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print as JSON.")] = False,
) -> None:
    """Validate a config and print it fully resolved (defaults and plugin parameters)."""
    from mskpipe import api

    try:
        resolved = api.resolved_config(config, overrides or ())
    except api.SetupError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_SETUP) from None
    if as_json:
        typer.echo(json.dumps(resolved.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        typer.echo(api.config_yaml(resolved), nl=False)


@config_app.command("schema")
def config_schema(
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write to a file instead of stdout.")
    ] = None,
) -> None:
    """Print the JSON Schema of the config file (editor completion, GUI forms)."""
    from mskpipe import api

    text = json.dumps(api.config_schema(), indent=2, ensure_ascii=False) + "\n"
    if output is None:
        typer.echo(text, nl=False)
        return
    output.write_text(text, encoding="utf-8")
    typer.echo(f"Written {output}")


# ---------------------------------------------------------------------------- info


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
