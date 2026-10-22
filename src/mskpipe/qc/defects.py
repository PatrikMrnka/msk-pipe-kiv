# SPDX-License-Identifier: Apache-2.0
"""Geometric defects found in the run: what, where, fixed or not.

Every step that finds defects writes ``defects.json`` into its own folder (``labelmap``:
holes, gaps, short bones, with ``patches.nii.gz`` of the added voxels; ``mesh``: bone
surfaces with a wrong genus; ``export_mw2``: muscles Muscle Wrapping cannot use), so a
cached step keeps its report. :func:`collect` joins them in step order; it is read by the
CLI (``mskpipe qc``), the GUI (3D view focused on each defect) and the statistics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

DEFECTS_FILE = "defects.json"
PATCHES_FILE = "patches.nii.gz"  # labelmap step: value i = voxels added for defect i (1-based)
Kind = Literal["hole", "gap", "short", "surface", "missing"]


@dataclass
class Defect:
    structure: str
    kind: Kind
    center_ijk: list[float] | None
    center_mm: list[float] | None  # world (NIfTI RAS) coordinates of the input image
    size_mm: float | None  # hole extent / gap length
    fixed: bool
    method: str
    voxels: int = 0  # voxels added by the repair
    detail: str = ""
    step: str = "labelmap"
    end_a_mm: list[float] | None = None  # gaps: facing ends of the fragments
    end_b_mm: list[float] | None = None

    @property
    def severity(self) -> Literal["warning", "error"]:
        """``error`` = the output is not usable as is (unfixed defect)."""
        return "warning" if self.fixed else "error"

    def message(self) -> str:
        where = (
            ""
            if self.center_mm is None
            else " at ({:.0f}, {:.0f}, {:.0f}) mm".format(*self.center_mm)
        )
        size = "" if self.size_mm is None else f", {self.size_mm:g} mm"
        state = f"fixed ({self.method})" if self.fixed else "NOT fixed"
        return f"{self.structure}: {self.kind}{size}{where} - {state}"


@dataclass
class DefectReport:
    defects: list[Defect] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def extend(self, items: list[Defect]) -> None:
        self.defects.extend(items)

    @property
    def unfixed(self) -> list[Defect]:
        return [d for d in self.defects if not d.fixed]

    def summary(self) -> dict[str, Any]:
        """Counts for the run manifest and the paper statistics."""
        out: dict[str, Any] = {"n": len(self.defects), "unfixed": len(self.unfixed)}
        for d in self.defects:
            key = f"{d.kind}s"
            out.setdefault(key, {}).setdefault(d.structure, 0)
            out[key][d.structure] += 1
        return out

    def write(self, qc_dir: Path) -> Path:
        qc_dir.mkdir(parents=True, exist_ok=True)
        path = qc_dir / DEFECTS_FILE
        data = {
            "summary": self.summary(),
            "meta": self.meta,
            "defects": [asdict(d) for d in self.defects],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> DefectReport:
        if path.is_dir():
            path = path / DEFECTS_FILE
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([Defect(**d) for d in data.get("defects", [])], data.get("meta", {}))


def step_dirs(run_dir: Path) -> list[Path]:
    from mskpipe.core.workspace import STEP_DIRS

    return [Path(run_dir) / d for d in STEP_DIRS.values()]


def collect(run_dir: Path) -> DefectReport:
    """All defects of a run, in step order (labelmap defects first: patch values match)."""
    report = DefectReport(meta={"run": Path(run_dir).name})
    for d in step_dirs(run_dir):
        if (d / DEFECTS_FILE).exists():
            report.extend(DefectReport.read(d).defects)
    return report


def length_check(
    structure: str, length_mm: float, reference: str, reference_mm: float, min_ratio: float
) -> Defect | None:
    """``short`` defect when a bone is clearly shorter than its reference (fibula vs tibia):
    part of it was not segmented or lies outside the field of view."""
    if reference_mm <= 0 or length_mm >= min_ratio * reference_mm:
        return None
    return Defect(
        structure=structure,
        kind="short",
        center_ijk=None,
        center_mm=None,
        size_mm=round(reference_mm - length_mm, 1),
        fixed=False,
        method="check only",
        detail=f"length {length_mm:.0f} mm = {100 * length_mm / reference_mm:.0f} % of "
        f"{reference} ({reference_mm:.0f} mm), expected >= {100 * min_ratio:.0f} %",
    )


__all__ = [
    "DEFECTS_FILE",
    "PATCHES_FILE",
    "Defect",
    "DefectReport",
    "collect",
    "length_check",
    "step_dirs",
]
