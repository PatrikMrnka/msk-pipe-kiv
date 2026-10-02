# SPDX-License-Identifier: Apache-2.0
"""pystaple (Python port of msk-STAPLE); writes the ``.osim`` itself, no OpenSim needed.

Runs the workflow of STAPLE's ``hip_model.m`` (pelvis: STAPLE, femur: GIBOC-cylinder,
tibia: Kai2014, joints: auto2020), which pystaple reproduces to 1e-7 mm of MATLAB STAPLE.
The bodies of the model keep the frame of the input meshes: joint and marker positions
in the ``.osim`` are the mesh coordinates in metres.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from mskpipe.plugins.base import PluginParams, SkeletonPlugin

if TYPE_CHECKING:
    from mskpipe.config.schema import SkeletonConfig
    from mskpipe.core.step import StepContext


def required_bones(side: str) -> tuple[str, ...]:
    """Bone meshes the hip model is built from."""
    return ("pelvis_no_sacrum", f"femur_{side}", f"tibia_{side}")


def build_hip_model(
    bones: Mapping[str, Path],
    out_dir: Path,
    config: SkeletonConfig,
    *,
    log_file: Path | None = None,
) -> Path:
    """Build ``out_dir/<model_name>.osim`` and ``out_dir/Geometry`` from bone meshes (mm).

    ``bones`` maps bone names to ``.stl`` (or pystaple ``.mat``) files and must contain
    :func:`required_bones`; other entries are ignored.
    """
    names = required_bones(config.side)
    missing = [n for n in names if n not in bones]
    if missing:
        raise FileNotFoundError(f"missing bone meshes for the hip model: {', '.join(missing)}")

    from pystaple.io import load_mesh
    from pystaple.workflow import build_hip_model as _build

    handler = _attach_log(log_file)
    try:
        geom_set = {n: load_mesh(Path(bones[n])) for n in names}
        osim, *_ = _build(
            geom_set,
            out_dir,
            body_mass=config.body_mass,
            joint_defs=config.joint_definitions,
            vis_geom_format=config.geometry_format,
            model_file_name=f"{config.model_name}.osim",
            coeff_face_reduc=config.geometry_reduction,
            backend="xml",
        )
    finally:
        _detach_log(handler)
    return Path(osim)


class PyStaple(SkeletonPlugin):
    name = "pystaple"
    description = "Lower-limb skeletal model (STAPLE hip_model) from bone meshes; see `skeleton`."
    requires_modules = ("pystaple", "fast_simplification")

    def build(
        self, ctx: StepContext, bones: Mapping[str, Path], out_dir: Path, params: PluginParams
    ) -> Path:
        return build_hip_model(
            bones, out_dir, ctx.config.skeleton, log_file=ctx.ws.logs_dir / f"{ctx.step_name}.log"
        )


def _attach_log(log_file: Path | None) -> logging.Handler | None:
    if log_file is None:
        return None
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("pystaple")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def _detach_log(handler: logging.Handler | None) -> None:
    if handler is not None:
        logging.getLogger("pystaple").removeHandler(handler)
        handler.close()
