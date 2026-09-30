# SPDX-License-Identifier: Apache-2.0
"""pystaple (Python port of msk-STAPLE); writes the ``.osim`` itself, no OpenSim needed."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from mskpipe.plugins.base import PluginParams, SkeletonPlugin

if TYPE_CHECKING:
    from mskpipe.core.step import StepContext


class PyStaple(SkeletonPlugin):
    name = "pystaple"
    description = "Lower-limb skeletal model (STAPLE) from bone meshes; options in `skeleton`."
    requires_modules = ("pystaple",)

    def build(
        self, ctx: StepContext, bones: Mapping[str, Path], out_dir: Path, params: PluginParams
    ) -> Path:
        raise NotImplementedError("ported in phase 4 (step skeleton)")
