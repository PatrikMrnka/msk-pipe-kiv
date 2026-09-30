# SPDX-License-Identifier: Apache-2.0
"""Atlas-based attachment areas (method of the 2026 bachelor thesis)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from mskpipe.plugins.base import AttachmentsPlugin, PluginParams

if TYPE_CHECKING:
    from mskpipe.core.step import StepContext


class AtlasBasedParams(PluginParams):
    # Defaults from config.json of the thesis.
    threshold_mm: float = Field(
        5.0, gt=0.0, le=50.0, description="Distance threshold between muscle and bone [mm]."
    )
    clip_fraction: float = Field(
        0.22, ge=0.0, lt=0.5, description="Fraction of the muscle clipped at each end."
    )


class AtlasBased(AttachmentsPlugin):
    name = "atlas_based"
    description = "Attachment areas transferred from an atlas (default LHDL)."
    Params = AtlasBasedParams
    requires_modules = ("vtk",)

    def compute(
        self,
        ctx: StepContext,
        bones: Mapping[str, Path],
        muscles: Mapping[str, Path],
        out_dir: Path,
        params: PluginParams,
    ) -> Mapping[str, Path]:
        raise NotImplementedError("ported in phase 4 (step attachments)")
