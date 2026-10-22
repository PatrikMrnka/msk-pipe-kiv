# SPDX-License-Identifier: Apache-2.0
"""``mskpipe qc RUN_DIR [--show] [--png]``: list the geometry defects of a run, open the
3D view focused on them, or render one PNG per defect (report, paper)."""

from __future__ import annotations

from pathlib import Path

from mskpipe.qc.defects import collect


def qc_command(
    run_dir: Path, show: bool = False, png: bool = False, defect: int | None = None
) -> int:
    """Exit code 0 = no unfixed defect, 2 = some defect needs attention."""
    report = collect(run_dir)
    if not report.defects:
        print("No geometry defects.")
    for i, d in enumerate(report.defects, 1):
        print(f"{i:>3}  {'ok ' if d.fixed else 'ERR'}  {d.message()}\n        {d.detail}")
    if png:
        from mskpipe.qc.viewer import snapshots

        for f in snapshots(run_dir):
            print(f"  {f}")
    if show:
        from mskpipe.qc.viewer import show as show_window

        show_window(run_dir, None if defect is None else defect - 1)
    return 2 if report.unfixed else 0


__all__ = ["qc_command"]
