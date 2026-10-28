# SPDX-License-Identifier: Apache-2.0
"""Command-line interface (``mskpipe``). Thin layer over :mod:`mskpipe.api`.

Exit codes: 0 completed, 1 a step failed, 2 invalid request/config/device (nothing run),
130 cancelled (Ctrl+C).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TextIO

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
setup_app = typer.Typer(
    help="Model weights, TotalSegmentator licence key and attachment atlas.",
    no_args_is_help=True,
)
app.add_typer(setup_app, name="setup")

EXIT_SETUP = 2
EXIT_CANCELLED = 130
EXIT_BUSY = 3  # another msk-pipe process holds the setup lock


class SetupModality(StrEnum):
    CT = "ct"
    MRI = "mri"
    ALL = "all"


class SetupTool(StrEnum):
    ALL = "all"
    TOTALSEGMENTATOR = "totalsegmentator"
    MUSCLEMAP = "musclemap"


class EventFormat(StrEnum):
    JSONL = "jsonl"


ModalityChoice = StrEnum("ModalityChoice", {m: m for m in (*(x.value for x in Modality), "auto")})  # type: ignore[misc]


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
            help="Input volume (.nii/.nii.gz) or a folder of DICOM files. Not used with --resume.",
            show_default=False,
        ),
    ] = None,
    modality: Annotated[
        ModalityChoice | None,
        typer.Option(
            "--modality",
            "-m",
            help="Modality of the input; auto = from DICOM, the dcm2niix JSON, header or "
            "intensities.",
        ),
    ] = None,
    subject: Annotated[
        str | None, typer.Option("--subject", "-s", help="Subject id (default: file name).")
    ] = None,
    series: Annotated[
        str | None,
        typer.Option(
            "--series",
            help="DICOM folder with several series: the one to process (see `mskpipe dicom`).",
        ),
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
    events: Annotated[
        EventFormat | None,
        typer.Option(
            "--events",
            help="Stream progress as one JSON object per line on stdout (for GUIs); "
            "log messages are part of the stream.",
        ),
    ] = None,
    cancel_on_stdin: Annotated[
        bool,
        typer.Option(
            "--cancel-on-stdin",
            help="Cancel the run when 'cancel' or end of input arrives on stdin (for GUIs).",
        ),
    ] = False,
) -> None:
    """Run the pipeline on one volume: segmentation -> ... -> Muscle Wrapping 2.x input.

    \b
    Examples:
      mskpipe run ct.nii.gz -m ct -d gpu
      mskpipe run scan.nii.gz -m auto
      mskpipe run D:/data/patient01_dicom -m auto --series 3
      mskpipe run ct.nii.gz -m ct -c lhdl.yaml --set attachments.params.nonrigid=cpd
      mskpipe run --resume runs/<run> --from export_mw2
    """
    from mskpipe import api

    request = api.RunRequest(
        image=image,
        modality=modality.value if modality else None,
        subject=subject,
        series=series,
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
    cancel = threading.Event()
    if cancel_on_stdin:
        threading.Thread(
            target=watch_stdin, args=(sys.stdin, cancel), name="mskpipe-stdin", daemon=True
        ).start()
    try:
        prepared = api.prepare_run(request)
        if not _report_preflight(api.preflight(prepared), no_preflight):
            raise typer.Exit(EXIT_SETUP)
        if events is not None:
            summary = api.run(
                prepared, on_event=_event_writer(sys.stdout), log_level=level, cancel=cancel
            )
            raise typer.Exit(summary.exit_code)
        with _console_log(level, sys.stderr if as_json else sys.stdout):
            summary = api.run(prepared, cancel=cancel)
    except api.SetupCancelled as exc:
        typer.echo(f"Cancelled: {exc}", err=True)
        raise typer.Exit(EXIT_CANCELLED) from None
    except api.SetupError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_SETUP) from None
    if as_json:
        typer.echo(summary.model_dump_json(indent=2))
    else:
        _print_summary(summary)
    raise typer.Exit(summary.exit_code)


def watch_stdin(stream: TextIO, cancel: threading.Event, poll_s: float = 0.25) -> None:
    """Set ``cancel`` on a line 'cancel' or at end of input (the parent GUI is gone).

    On Windows a pipe is polled (``PeekNamedPipe``) instead of read with a blocking call:
    while one thread waits in a synchronous ``ReadFile`` on the inherited stdin pipe,
    ``subprocess.Popen`` in another thread hangs duplicating that handle, which froze runs
    started from the GUI before the first step (device detection starts nvidia-smi).
    """
    pipe = _windows_pipe(stream)
    if pipe is not None:
        poll_pipe(*pipe, cancel, poll_s)
        return
    try:
        for line in stream:
            if line.strip().lower() == "cancel":
                break
    except (OSError, ValueError):  # closed or invalid handle: treat as end of input
        pass
    cancel.set()


def poll_pipe(
    peek: Callable[[], int],
    read: Callable[[int], bytes],
    cancel: threading.Event,
    poll_s: float = 0.25,
) -> None:
    """Poll a pipe without blocking: ``peek()`` = bytes available (raises OSError when the
    writer is gone), ``read(n)`` returns available bytes. Sets ``cancel`` on a 'cancel'
    line or end of input; returns early if ``cancel`` is set by someone else."""
    buffer = b""
    while not cancel.is_set():
        try:
            available = peek()
            data = read(available) if available else b""
        except OSError:  # broken pipe: the parent closed stdin or died
            break
        if available and not data:
            break
        buffer += data
        *lines, buffer = buffer.split(b"\n")
        if any(line.strip().lower() == b"cancel" for line in lines):
            break
        if not available:
            time.sleep(poll_s)
    cancel.set()


def _windows_pipe(stream: TextIO) -> tuple[Callable[[], int], Callable[[int], bytes]] | None:
    """(peek, read) for stdin if it is a Windows pipe, else None (POSIX, console, tests)."""
    if sys.platform != "win32":
        return None
    try:
        import _winapi
        import msvcrt

        fd = stream.fileno()
        handle = msvcrt.get_osfhandle(fd)
        _winapi.PeekNamedPipe(handle, 0)
    except (AttributeError, ImportError, OSError, ValueError):  # not a pipe (console)
        return None

    def peek() -> int:
        return int(_winapi.PeekNamedPipe(handle, 0)[0])

    def read(n: int) -> bytes:
        return os.read(fd, n)

    return peek, read


def _event_writer(stream: TextIO):
    """Thread-safe writer of RunEvents as JSON lines (logs come from tool reader threads)."""
    lock = threading.Lock()
    with suppress(AttributeError, OSError):
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    def write(event) -> None:
        with lock:
            stream.write(event.to_line() + "\n")
            stream.flush()

    return write


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


# ---------------------------------------------------------------------------- batch


@app.command()
def batch(
    file: Annotated[Path, typer.Argument(help="Batch file (.yaml or .csv).", show_default=False)],
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", metavar="KEY=VALUE", help="Override for every run (repeatable)."),
    ] = None,
    device: Annotated[
        Device | None, typer.Option("--device", "-d", help="Compute device for every run.")
    ] = None,
    runs_dir: Annotated[
        Path | None,
        typer.Option("--runs-dir", "-o", help="Folder for run folders (runtime.runs_dir)."),
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Do not reuse results of earlier runs (timings).")
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Batch folder (default: <runs_dir>/batches/<time>_<exp>)."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Only list the runs and the pre-run check results.")
    ] = False,
    stop_on_error: Annotated[
        bool, typer.Option("--stop-on-error", help="Stop at the first run that does not complete.")
    ] = False,
    no_preflight: Annotated[
        bool, typer.Option("--no-preflight", help="Start even if pre-run checks report errors.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Also print the output of external tools.")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only warnings and errors.")] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the batch entries as JSON (stdout).")
    ] = False,
) -> None:
    """Run several inputs and/or configurations one after another (format: mskpipe.batch).

    \b
    Examples:
      mskpipe batch experiments/e13_timing.yaml --dry-run
      mskpipe batch experiments/e13_timing.yaml -d gpu --no-cache
    """
    from mskpipe import api
    from mskpipe import batch as batch_mod

    try:
        spec, base = batch_mod.load_batch(file)
        planned = batch_mod.expand(
            spec,
            base,
            overrides=tuple(overrides or ()),
            device=device,
            runs_dir=runs_dir,
            cache=False if no_cache else None,
        )
        first = api.prepare_run(planned[0].request)
    except api.SetupError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_SETUP) from None

    typer.echo(f"{len(planned)} run(s), experiment: {spec.experiment or '-'}", err=as_json)
    for p in planned:
        variant = f" [{p.variant}]" if p.variant else ""
        typer.echo(f"  {p.index:>3}. {p.job}{variant} #{p.repeat}", err=as_json)
    repeats = max(p.repeat for p in planned)
    if repeats > batch_mod.RECOMMENDED_REPEAT:
        typer.echo(
            f"WARNING: repeat {repeats} > {batch_mod.RECOMMENDED_REPEAT}: timings settle after "
            "3 runs, more repetitions only take time",
            err=True,
        )
    if repeats > 1 and first.config.runtime.cache:
        typer.echo(
            "WARNING: repeated runs with runtime.cache=true reuse earlier results "
            "(use --no-cache for timings)",
            err=True,
        )
    issues = [] if no_preflight and not dry_run else batch_mod.check_batch(planned)
    for p, issue in issues:
        typer.echo(
            f"{issue.severity.upper()}: run {p.index} ({p.job}): {issue.where}: {issue.message}",
            err=True,
        )
    errors = [i for _, i in issues if i.severity == "error"]
    if dry_run:
        raise typer.Exit(EXIT_SETUP if errors else 0)
    if errors and not no_preflight:
        typer.echo(
            f"{len(errors)} pre-run check(s) failed; fix them or use --no-preflight.", err=True
        )
        raise typer.Exit(EXIT_SETUP)

    folder = out or batch_mod.batch_folder(first.config.runtime.runs_dir, spec.experiment)
    total = len(planned)

    def progress(entry, event) -> None:
        if event is not None:
            return
        if entry.status == "running":
            variant = f" [{entry.variant}]" if entry.variant else ""
            typer.echo(
                f"=== [{entry.index}/{total}] {entry.job}{variant} #{entry.repeat}", err=as_json
            )
        else:
            wall = f", {entry.wall_s:.1f} s" if entry.wall_s is not None else ""
            typer.echo(f"=== [{entry.index}/{total}] {entry.status}{wall}", err=as_json)

    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    with _console_log(level, sys.stderr if as_json else sys.stdout):
        result = batch_mod.run_batch(
            planned,
            folder,
            experiment=spec.experiment,
            batch_file=file.resolve(),
            stop_on_error=stop_on_error,
            on_event=progress,
        )
    if as_json:
        typer.echo(batch_mod.entries_json(result.entries))
    else:
        _print_batch(result)
    raise typer.Exit(result.exit_code)


def _print_batch(result) -> None:
    typer.echo(f"\nBatch: {result.folder}")
    typer.echo(f"{'#':>4}  {'job':<16}{'variant':<28}{'rep':>4}  {'status':<12}{'wall s':>9}")
    for e in result.entries:
        wall = f"{e.wall_s:.1f}" if e.wall_s is not None else "-"
        typer.echo(
            f"{e.index:>4}  {e.job[:15]:<16}{e.variant[:27]:<28}{e.repeat:>4}  "
            f"{e.status:<12}{wall:>9}"
        )
    for e in result.entries:
        if e.error:
            typer.echo(f"run {e.index}: {e.error}", err=True)


# ---------------------------------------------------------------------------- stats


@app.command()
def stats(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Run folders, folders of runs (runs/) or batch folders. Default: runs.",
            show_default=False,
        ),
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Folder for the CSV tables.")
    ] = Path("stats"),
    experiment: Annotated[
        str | None,
        typer.Option("--experiment", "-e", help="Only runs of this batch experiment."),
    ] = None,
) -> None:
    """Collect timings, resources and Muscle Wrapping export checks into CSV tables.

    \b
    Tables: runs, steps, tools, muscles, areas, registration, summary (mean +- SD per step).
    Example:
      mskpipe stats runs -e E13 -o paper/stats/E13
    """
    from mskpipe import stats as stats_mod

    tables = stats_mod.collect(paths or [Path("runs")], experiment=experiment)
    for message in tables.skipped:
        typer.echo(f"WARNING: skipped {message}", err=True)
    runs = tables["runs"]
    if not runs:
        typer.echo("No runs found.", err=True)
        raise typer.Exit(1)
    for path in stats_mod.write_tables(tables, output):
        typer.echo(f"Written {path} ({len(tables[path.stem])} rows)")
    typer.echo("")
    typer.echo(
        f"{'job':<14}{'variant':<24}{'device':<7}{'step':<13}{'n':>3}{'mean s':>10}{'SD':>8}"
    )
    for row in tables["summary"]:
        sd = f"{row['wall_sd_s']:.1f}" if row["wall_sd_s"] is not None else "-"
        job = str(row["job"] or row["subject"])[:13]
        variant = str(row["variant"] or "")[:23]
        dev = str(row["device"] or "-")
        typer.echo(
            f"{job:<14}{variant:<24}{dev:<7}{row['step']:<13}{row['n']:>3}"
            f"{row['wall_mean_s']:>10.1f}{sd:>8}"
        )


# ---------------------------------------------------------------------------- qc


@app.command()
def qc(
    run_dir: Annotated[Path, typer.Argument(help="Run folder.", show_default=False)],
    show: Annotated[
        bool, typer.Option("--show", help="Open the 3D view focused on the defects.")
    ] = False,
    png: Annotated[
        bool, typer.Option("--png", help="Save one image per defect into <run>/qc/.")
    ] = False,
    defect: Annotated[
        int | None, typer.Option("--defect", "-n", help="Start the 3D view at defect N.")
    ] = None,
) -> None:
    """List the geometry defects of a run (bone holes, gaps, short bones, surfaces).

    \b
    Exit code 0 = nothing left to fix, 2 = a defect was not repaired.
    In the 3D view: mouse = rotate / pan / zoom, n / p = next / previous defect,
    a = whole model, o = other structures solid / transparent.
    Example:
      mskpipe qc runs/20261021-162806_MRI_97094935 --show
    """
    from mskpipe.qc.command import qc_command

    if not run_dir.is_dir():
        typer.echo(f"Error: not a folder: {run_dir}", err=True)
        raise typer.Exit(1)
    raise typer.Exit(qc_command(run_dir, show=show, png=png, defect=defect))


# ---------------------------------------------------------------------------- view


@app.command()
def view(
    target: Annotated[
        Path | None,
        typer.Argument(
            help="Volume (.nii/.nii.gz), DICOM folder or run folder.", show_default=False
        ),
    ] = None,
    labels: Annotated[
        list[Path] | None,
        typer.Option("--labels", "-l", help="Label map over the image (repeatable)."),
    ] = None,
    series: Annotated[
        str | None, typer.Option("--series", help="DICOM folder: the series to show.")
    ] = None,
) -> None:
    """Show a volume in axial, coronal and sagittal slices, with label maps.

    \b
    A run folder opens its input image with the unified label map; the raw segmentations
    and the defect repairs can be chosen in the window. Label maps need not share the
    image grid (nearest voxel). Double click a structure to go to it, L = labels on/off.
    Examples:
      mskpipe view runs/20261021-162806_MRI_97094935
      mskpipe view CT.nii.gz -l runs/<run>/02_labelmap/labelmap.nii.gz
      mskpipe view D:/data/patient01_dicom --series 3
    """
    for path in [target, *(labels or [])]:
        if path is not None and not path.exists():
            typer.echo(f"Error: not found: {path}", err=True)
            raise typer.Exit(1)
    if target is not None and _is_dicom_input(target):
        from mskpipe import api

        try:
            target = api.input_volume(target, series)
        except api.SetupError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from None
    try:
        from mskpipe.gui.slice_viewer import main as view_main
    except ImportError as exc:
        typer.echo(f"Error: the viewer needs PySide6 ({exc}).", err=True)
        raise typer.Exit(1) from None
    raise typer.Exit(view_main(target, labels or []))


def _is_dicom_input(path: Path) -> bool:
    """A DICOM folder (not a run folder) or one DICOM file."""
    from mskpipe.core.workspace import INPUT_FILE
    from mskpipe.io.dicom import DICOM_SUFFIXES

    if path.is_dir():
        return not (path / INPUT_FILE).is_file()
    return path.suffix.lower() in DICOM_SUFFIXES


# ---------------------------------------------------------------------------- dicom


@app.command()
def dicom(
    folder: Annotated[Path, typer.Argument(help="Folder with DICOM files (subfolders included).")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the chosen series as NIfTI (.nii.gz)."),
    ] = None,
    series: Annotated[
        str | None, typer.Option("--series", help="Series to write (see the list).")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the series as JSON.")] = False,
) -> None:
    """List the series of a DICOM folder and check them (converted with dcm2niix).

    \b
    The id in the first column is what --series takes (also for `mskpipe run`).
    Examples:
      mskpipe dicom D:/data/patient01_dicom
      mskpipe dicom D:/data/patient01_dicom --series 3 -o ct.nii.gz
    """
    import dataclasses
    import shutil

    from mskpipe import api

    try:
        result = api.scan_dicom(folder)
    except api.SetupError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_SETUP) from None
    if as_json:
        data = dataclasses.asdict(result)
        data.pop("log", None)
        for item in data["volumes"]:
            item.pop("nifti", None)
            item.pop("sidecar", None)
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"{folder} (dcm2niix {result.version})")
        for vol in result.volumes:
            mark = "ok " if vol.usable else "-- "
            typer.echo(f"{mark} {vol.label()}")
            if not vol.usable:
                typer.echo(f"      not usable: {vol.status()}")
            for warning in vol.warnings:
                typer.echo(f"      warning: {warning}")
    if output is None:
        raise typer.Exit(0 if result.usable else EXIT_SETUP)
    try:
        chosen = result.select(series)
    except ValueError as exc:  # the series are listed above
        typer.echo(f"Error: {str(exc).splitlines()[0]}", err=True)
        raise typer.Exit(EXIT_SETUP) from None
    if not output.name.lower().endswith((".nii", ".nii.gz")):
        output = output.with_name(output.name + ".nii.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.name.lower().endswith(".nii.gz"):
        shutil.copyfile(chosen.nifti_path, output)
    else:
        import nibabel as nib

        nib.save(nib.load(str(chosen.nifti_path)), str(output))
    if chosen.sidecar:
        name = output.name[: -len(".nii.gz")] if output.name.endswith(".gz") else output.stem
        shutil.copyfile(chosen.sidecar, output.with_name(f"{name}.json"))
    typer.echo(f"Series {chosen.id} written to {output}", err=as_json)


# ---------------------------------------------------------------------------- gui


@app.command()
def gui() -> None:
    """Open the graphical user interface."""
    try:
        from mskpipe.gui.app import main as gui_main
    except ImportError as exc:
        typer.echo(
            f"Error: the GUI needs PySide6 ({exc}). Use a pixi environment with the GUI: "
            "pixi run -e cpu mskpipe gui",
            err=True,
        )
        raise typer.Exit(1) from None
    raise typer.Exit(gui_main([]))


# ---------------------------------------------------------------------------- setup


def _modalities(choice: SetupModality) -> tuple[str, ...]:
    return ("ct", "mri") if choice == SetupModality.ALL else (choice.value,)


def _setup_config(config: Path | None, overrides: list[str] | None):
    from mskpipe import api

    try:
        return api.resolved_config(config, overrides or ())
    except api.SetupError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_SETUP) from None


@contextmanager
def _setup_locked(purpose: str) -> Iterator[None]:
    """Hold the msk-pipe setup lock (one download/installation at a time across all GUI
    windows, consoles and runs); exit code 3 if another process holds it."""
    from mskpipe.resources.lock import BusyError, setup_lock

    try:
        lock = setup_lock(purpose).acquire()
    except BusyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_BUSY) from None
    try:
        yield
    finally:
        lock.release()


@setup_app.command("status")
def setup_status(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML config.")] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", metavar="KEY=VALUE", help="Override a config value (repeatable)."),
    ] = None,
    modality: Annotated[
        SetupModality, typer.Option("--modality", "-m", help="Modalities to check.")
    ] = SetupModality.ALL,
    as_json: Annotated[bool, typer.Option("--json", help="Print as JSON.")] = False,
) -> None:
    """Show whether weights, licence key and atlas are in place.

    Also prints where msk-pipe keeps its data. Exit code 0 = ready, 1 = something is
    missing (missing weights are otherwise downloaded on the first run, with internet).
    """
    from mskpipe.resources.status import collect

    status = collect(_setup_config(config, overrides), _modalities(modality))
    if as_json:
        typer.echo(status.model_dump_json(indent=2))
        raise typer.Exit(0 if status.ready else 1)
    typer.echo(f"msk-pipe data:    {status.home}")
    typer.echo(f"TotalSegmentator: {status.totalseg_home}")
    lic = status.licence
    state = "set" if lic.present and not lic.problem else lic.problem or "not set"
    need = "" if lic.needed else " (not needed by this config)"
    typer.echo(f"Licence key:      {state}{need}")
    atlas = status.atlas
    where = f"{atlas.path} ({atlas.source})" if atlas.path else "not found"
    mark = "ok" if atlas.ok else "MISSING"
    typer.echo(f"Atlas:            {mark} - {where}" if atlas.needed else "Atlas: not needed")
    gpu = status.gpu
    if gpu is not None and gpu.installed:
        typer.echo(f"GPU support:      {gpu.tag} - {gpu.path}")
    elif gpu is not None and gpu.supported:
        typer.echo(f"GPU support:      not installed (optional: mskpipe setup gpu, {gpu.tag})")
    rnrr = status.fast_rnrr
    if rnrr is not None and (rnrr.needed or rnrr.ok):
        if rnrr.ok:
            text = f"ok - {rnrr.path} ({rnrr.source})"
        else:
            text = "MISSING (research use only: mskpipe setup fast-rnrr)"
        typer.echo(f"Fast-RNRR:        {text}")
    typer.echo("Weights:")
    for w in status.weights:
        lock = " [licence]" if w.licensed else ""
        detail = f" - {w.detail}" if w.detail else ""
        typer.echo(f"  {w.tool:<17}{w.name + lock:<32}{w.status:<9}{detail}")
    if not status.ready:
        typer.echo("\nGet what is missing: mskpipe setup weights / licence <key> / atlas <zip>")
    raise typer.Exit(0 if status.ready else 1)


@setup_app.command("weights")
def setup_weights(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML config.")] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", metavar="KEY=VALUE", help="Override a config value (repeatable)."),
    ] = None,
    modality: Annotated[
        SetupModality, typer.Option("--modality", "-m", help="Modalities to prepare.")
    ] = SetupModality.ALL,
    tool: Annotated[SetupTool, typer.Option("--tool", "-t", help="Segmenter.")] = SetupTool.ALL,
) -> None:
    """Download the model weights the configuration needs, in advance.

    Use it before working offline; weights already present are skipped.

    \b
    Examples:
      mskpipe setup weights
      mskpipe setup weights -m ct --set segmentation.bones=musclemap
    """
    resolved = _setup_config(config, overrides)
    with _setup_locked("downloading model weights"):
        failed = _download_weights(resolved, modality, tool)
    raise typer.Exit(1 if failed else 0)


def _download_weights(resolved, modality: SetupModality, tool: SetupTool) -> int:
    from mskpipe.plugins.segmenters.musclemap import model_version
    from mskpipe.resources.licence import licence_number
    from mskpipe.resources.weights import (
        WeightsError,
        download_musclemap,
        download_ts,
        weights_status,
    )

    failed = 0
    for item in weights_status(resolved, _modalities(modality)):
        if tool != SetupTool.ALL and item.tool != tool.value:
            continue
        if item.status == "unknown":
            typer.echo(f"{item.tool} {item.name}: skipped ({item.detail})", err=True)
            failed += 1
            continue
        if item.licensed and licence_number() is None:
            typer.echo(
                f"totalsegmentator {item.name}: skipped, needs the licence key "
                "(mskpipe setup licence <key>)",
                err=True,
            )
            failed += 1
            continue
        try:
            if item.tool == "totalsegmentator":
                download_ts(item.name, typer.echo)
            else:
                download_musclemap(model_version(resolved), typer.echo)
        except WeightsError as exc:
            typer.echo(f"Error: {exc}", err=True)
            failed += 1
    return failed


@setup_app.command("licence")
def setup_licence(
    key: Annotated[str, typer.Argument(help="Key such as aca_XXXXXXXXXXXXXX.", show_default=False)],
    skip_validation: Annotated[
        bool, typer.Option("--skip-validation", help="Do not check the key online.")
    ] = False,
) -> None:
    """Store the TotalSegmentator licence key (free for non-commercial use).

    Needed for bones from TotalSegmentator (segmentation.bones=totalsegmentator, default).
    """
    from mskpipe.resources.licence import LICENCE_URL, LicenceError, set_licence

    try:
        with _setup_locked("storing the TotalSegmentator licence key"):
            set_licence(key, typer.echo, skip_validation=skip_validation)
    except LicenceError as exc:
        typer.echo(f"Error: {exc}. Get a key at {LICENCE_URL}", err=True)
        raise typer.Exit(1) from None
    typer.echo("Licence key stored.")


@setup_app.command("atlas")
def setup_atlas(
    source: Annotated[
        str, typer.Argument(help="Atlas folder, .zip file or URL of a zip.", show_default=False)
    ],
    name: Annotated[str, typer.Option("--name", help="Folder name under <home>/atlas.")] = "lhdl",
    sha256: Annotated[
        str | None, typer.Option("--sha256", help="Expected SHA-256 of the zip.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Replace an installed atlas.")
    ] = False,
) -> None:
    """Install an attachment atlas from a folder, zip or URL.

    It is used when neither attachments.params.atlas_dir nor MSKPIPE_ATLAS_DIR is set.
    """
    from mskpipe.resources.atlas import AtlasError, install_atlas
    from mskpipe.resources.download import DownloadError

    try:
        with _setup_locked("installing an attachment atlas"):
            install_atlas(source, typer.echo, name=name, sha256=sha256, force=force)
    except (AtlasError, DownloadError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None


@setup_app.command("gpu")
def setup_gpu(
    remove: Annotated[bool, typer.Option("--remove", help="Delete the GPU support.")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Reinstall.")] = False,
    wheel_dir: Annotated[
        Path | None,
        typer.Option("--wheel-dir", help="Folder with the wheels already downloaded (offline)."),
    ] = None,
    keep_downloads: Annotated[
        bool, typer.Option("--keep-downloads", help="Keep the downloaded wheels.")
    ] = False,
) -> None:
    """Install GPU support for the Windows release (CUDA build of PyTorch, ~3.5 GB).

    Downloads the torch/torchvision CUDA wheels pinned in pixi.lock, checks their SHA-256
    and unpacks them next to the environment; runs on the GPU use them, CPU runs do not.
    Needs an NVIDIA driver 570 or newer. From the sources use `pixi run -e gpu` instead.
    """
    from mskpipe.resources.download import DownloadError
    from mskpipe.resources.gpu import GpuSetupError, install_gpu, remove_gpu

    if remove:
        try:
            with _setup_locked("removing GPU support"):
                remove_gpu(typer.echo)
        except (GpuSetupError, OSError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from None
        return

    def progress(wheel):
        last = [-1]

        def report(done: int, total: int | None) -> None:
            if not total:
                return
            pct = int(100 * done / total) // 5 * 5
            if pct != last[0]:
                last[0] = pct
                typer.echo(f"  {wheel.name}: {pct:3d} % of {total / 2**30:.2f} GiB")

        return report

    try:
        with _setup_locked("installing GPU support"):
            install_gpu(
                typer.echo,
                progress=progress,
                force=force,
                keep_downloads=keep_downloads,
                wheel_dir=wheel_dir,
            )
    except (GpuSetupError, DownloadError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo("Check it with: mskpipe device -d gpu")


@setup_app.command("fast-rnrr")
def setup_fast_rnrr(
    accept: Annotated[
        bool,
        typer.Option(
            "--accept-research-licence",
            help="Confirm that you use Fast-RNRR for research purposes only (its authors' terms).",
        ),
    ] = False,
    remove: Annotated[bool, typer.Option("--remove", help="Delete the built program.")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Build again.")] = False,
    keep_build_files: Annotated[
        bool, typer.Option("--keep-build-files", help="Keep sources, compiler and objects.")
    ] = False,
    jobs: Annotated[
        int | None, typer.Option("--jobs", "-j", min=1, help="Parallel compiler processes.")
    ] = None,
) -> None:
    """Build Fast-RNRR on this computer (optional non-rigid registration of attachments).

    Fast-RNRR (Yao et al., CVPR 2020) is patented and may be used for research purposes only,
    so msk-pipe does not distribute it. This downloads the authors' source (pinned commit),
    OpenMesh, Eigen and the Zig compiler (~110 MB, SHA-256 checked), builds the program in a
    few minutes and checks it against the authors' test result. Nothing else is installed.
    Then use: --set attachments.params.nonrigid=fast_rnrr
    """
    from mskpipe.resources.download import DownloadError
    from mskpipe.resources.fast_rnrr import (
        FastRnrrSetupError,
        build_fast_rnrr,
        load_spec,
        remove_fast_rnrr,
    )
    from mskpipe.resources.tool import ToolCancelled

    if remove:
        try:
            with _setup_locked("removing Fast-RNRR"):
                remove_fast_rnrr(typer.echo)
        except (FastRnrrSetupError, OSError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from None
        return
    if not accept:
        spec = load_spec()
        typer.echo(spec.notice)
        typer.echo(
            "\nTo build it, confirm that you use it for research purposes only:\n"
            "  mskpipe setup fast-rnrr --accept-research-licence",
            err=True,
        )
        raise typer.Exit(EXIT_SETUP)
    try:
        with _setup_locked("building Fast-RNRR"):
            build_fast_rnrr(
                typer.echo,
                accepted=True,
                force=force,
                keep_build_files=keep_build_files,
                jobs=jobs,
            )
    except ToolCancelled:
        raise typer.Exit(EXIT_CANCELLED) from None
    except (FastRnrrSetupError, DownloadError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo("Use it with: --set attachments.params.nonrigid=fast_rnrr")


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


@config_app.command("check")
def config_check(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML config.")] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option("--set", metavar="KEY=VALUE", help="Override a config value (repeatable)."),
    ] = None,
    image: Annotated[
        Path | None,
        typer.Option(
            "--image",
            "-i",
            help="Also check this input volume or DICOM folder (spacing, modality).",
        ),
    ] = None,
    modality: Annotated[
        ModalityChoice | None,
        typer.Option("--modality", "-m", help="Modality given for --image (checked against it)."),
    ] = None,
) -> None:
    """Check a config: errors, values outside the recommended ranges, settings that are
    probably not meant, this computer (threads, RAM, disk, GPU) and optionally an input."""
    from mskpipe import api
    from mskpipe.config import sanity

    try:
        resolved = api.resolved_config(config, overrides or ())
    except api.SetupError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(EXIT_SETUP) from None
    issues: list[sanity.Issue] = []
    if image is not None:
        try:
            image = api.input_volume(image)
        except api.SetupError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(EXIT_SETUP) from None
        issues += api.check_image(image)
    given = modality.value if modality is not None and modality.value != "auto" else None
    issues += sanity.check(resolved, image, given)
    for issue in issues:
        typer.echo(f"{issue.severity.upper()}: {issue.where}: {issue.message}")
    errors = sum(i.severity == "error" for i in issues)
    warnings = sum(i.severity == "warning" for i in issues)
    typer.echo(f"{errors} error(s), {warnings} warning(s)" if issues else "OK: no issues found")
    if errors:
        raise typer.Exit(EXIT_SETUP)


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
