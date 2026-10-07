# SPDX-License-Identifier: Apache-2.0
"""Fake pipeline for tests of the API, CLI, batch, stats and GUI (no tools, no data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import nibabel as nib
import numpy as np

from mskpipe.core.step import Step, StepContext, StepError
from mskpipe.core.workspace import STEPS

PIPELINE = STEPS[1:]


class FakeStep(Step):
    """Writes ``<name>.txt`` (export_mw2: the MW2 setup file) and one metric.

    ``FakeStep.fail``: names of steps that raise; ``FakeStep.cancel_in``: steps that
    set the run's cancel event and then check it (as a long step would).
    """

    calls: ClassVar[dict[str, int]] = {}
    fail: ClassVar[set[str]] = set()
    cancel_in: ClassVar[set[str]] = set()

    def __init__(self, name: str) -> None:
        self.name = name  # type: ignore[misc]
        self.config_sections = {  # type: ignore[misc]
            "segment": ("segmentation",),
            "labelmap": ("labelmap",),
            "mesh": ("mesh",),
            "skeleton": ("skeleton",),
            "attachments": ("attachments",),
            "export_mw2": ("export",),
        }[name]

    def run(self, ctx: StepContext) -> None:
        FakeStep.calls[self.name] = FakeStep.calls.get(self.name, 0) + 1
        if self.name in FakeStep.fail:
            raise StepError(f"{self.name} broken")
        if self.name in FakeStep.cancel_in and ctx.cancel is not None:
            ctx.cancel.set()
            ctx.check_cancel()
        ctx.logger.info("[%s] fake work", self.name)
        out = ctx.out_dir / (
            "setup_MuscleGeneratorTool.xml" if self.name == "export_mw2" else f"{self.name}.txt"
        )
        out.write_text(self.name, encoding="utf-8")
        ctx.record.add_output(out)
        ctx.record.metrics.update(METRICS.get(self.name, {}), value=1)
        if self.name == "export_mw2":
            (ctx.out_dir / "export.json").write_text(json.dumps(EXPORT_INDEX), encoding="utf-8")


# metrics shaped like those of the real steps (see mskpipe.steps.*)
METRICS: dict[str, dict] = {
    "segment": {
        "segment_time_s": 3.0,
        "tools": {
            "totalsegmentator": {
                "tasks": ["appendicular_bones", "total"],
                "time_s": 1.25,
                "device": "cpu",
                "n_requested": 10,
                "not_provided": [],
                "empty": ["vertebrae_S1"],
            },
            "musclemap": {
                "tasks": ["wholebody"],
                "time_s": 1.75,
                "device": "cpu",
                "n_requested": 40,
                "not_provided": [],
                "empty": [],
            },
        },
    },
    "labelmap": {"labelmap_time_s": 0.5},
    "mesh": {"mesh_time_s": 0.25, "n_meshed": 26},
    "attachments": {
        "attachments_time_s": 0.75,
        "registration": {
            "femur": {
                "status": "ok",
                "scale": 1.002,
                "rigid": {"mean_mm": 0.88, "p95_mm": 2.1, "trimmed_mean_mm": 0.7},
                "nonrigid": {"mean_mm": 0.69, "p95_mm": 1.6, "trimmed_mean_mm": 0.55},
            },
            "pelvis": {
                "status": "ok",
                "scale": 1.005,
                "rigid": {"mean_mm": 1.25, "p95_mm": 3.0, "trimmed_mean_mm": 1.0},
            },
        },
    },
    "export_mw2": {"export_time_s": 0.5, "n_selected": 2, "n_exported": 1},
}

_AREA = {
    "n_points": 12,
    "distinct": 12,
    "span_mm": 30.0,
    "shrink": 0.9,
    "crossing": False,
    "patch_fraction": 0.05,
    "gap_mean_mm": 2.0,
    "gap_max_mm": 4.0,
    "body": "pelvis",
    "nearest_body": "pelvis",
}
EXPORT_INDEX: dict = {
    "format": "mskpipe.mw2_input",
    "version": 1,
    "muscles": [
        {
            "name": "gluteus_medius_r",
            "status": "ok",
            "mesh": {"check": {"ok": True, "genus": 0, "components": 1}, "repaired": True},
            "areas": {
                "Ori": _AREA,
                "Ins": {
                    **_AREA,
                    "distinct": 1,
                    "body": "tibia_r",
                    "nearest_body": "femur_r",
                    "gap_mean_mm": 97.0,
                    "inflated": {
                        "radius_mm": 5.0,
                        "patch_fraction": 0.01,
                        "before": {"n_points": 12, "distinct": 1, "span_mm": 0.0},
                        "after": {"n_points": 18, "distinct": 18, "span_mm": 9.5},
                    },
                },
            },
            "problems": [],
            "warnings": ["insertion collapsed", "gluteus_medius_r insertion: area inflated"],
            "ori_ins_distance_mm": 150.0,
            "mean_edge_mm": 2.0,
        },
        {
            "name": "piriformis_r",
            "status": "excluded",
            "mesh": {},
            "areas": {},
            "problems": ["export.exclude: tendon around a bony pulley"],
            "excluded_by_config": "tendon around a bony pulley",
        },
    ],
}


def fake_steps() -> list[Step]:
    return [FakeStep(n) for n in PIPELINE]


def reset() -> None:
    FakeStep.calls = {}
    FakeStep.fail = set()
    FakeStep.cancel_in = set()


def write_nifti(
    path: Path, *, spacing: tuple[float, float, float] = (0.8, 0.8, 1.5), qform: bool = True
) -> Path:
    """Small CT-like volume with a valid header."""
    data = np.zeros((8, 8, 6), dtype=np.int16)
    affine = np.diag([*spacing, 1.0])
    img = nib.Nifti1Image(data, affine)
    if qform:
        img.set_qform(affine, code=1)
        img.set_sform(affine, code=1)
    else:
        img.set_qform(None, code=0)
        img.set_sform(None, code=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path
