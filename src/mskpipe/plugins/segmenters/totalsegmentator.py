# SPDX-License-Identifier: Apache-2.0
"""TotalSegmentator (tasks ``total``/``total_mr``, optionally ``appendicular_bones``)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mskpipe.plugins.base import PluginParams, SegmentationOutput, SegmenterPlugin

if TYPE_CHECKING:
    from mskpipe.core.step import StepContext


class TotalSegmentator(SegmenterPlugin):
    name = "totalsegmentator"
    description = "TotalSegmentator: pelvis, femur, gluteal muscles; tibia/fibula with licence."
    requires_modules = ("totalsegmentator",)
    supports_gpu = True

    def segment(
        self, ctx: StepContext, image: Path, out_dir: Path, params: PluginParams
    ) -> list[SegmentationOutput]:
        raise NotImplementedError("ported in phase 4 (step segment)")
