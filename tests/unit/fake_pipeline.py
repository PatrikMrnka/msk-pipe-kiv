# SPDX-License-Identifier: Apache-2.0
"""Fake pipeline for tests of the API, CLI, batch, stats and GUI (no tools, no data)."""

from __future__ import annotations

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
        ctx.record.metrics["value"] = 1


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
