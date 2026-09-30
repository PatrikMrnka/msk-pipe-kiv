# SPDX-License-Identifier: Apache-2.0
"""MuscleMap (CLI ``mm_segment``)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mskpipe.plugins.base import PluginParams, SegmentationOutput, SegmenterPlugin

if TYPE_CHECKING:
    from mskpipe.core.step import StepContext


class MuscleMap(SegmenterPlugin):
    name = "musclemap"
    description = "MuscleMap: hip and thigh muscles, pelvis, femur, tibia, fibula."
    requires_executables = ("mm_segment",)
    supports_gpu = True

    def segment(
        self, ctx: StepContext, image: Path, out_dir: Path, params: PluginParams
    ) -> list[SegmentationOutput]:
        raise NotImplementedError("ported in phase 4 (step segment)")
