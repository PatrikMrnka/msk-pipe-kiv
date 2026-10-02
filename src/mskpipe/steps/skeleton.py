# SPDX-License-Identifier: Apache-2.0
"""Step ``skeleton``: OpenSim bone model from the bone meshes of step ``mesh``.

Input (``03_mesh/``): ``bones/<name>.stl`` listed in ``meshes.json``.

Output (``04_skeleton/``)::

    <model_name>.osim       OpenSim model written by the backend (default pystaple)
    Geometry/<bone>.obj     visualization geometries referenced by the model
    skeleton.json           index: model, frame, settings, warnings and quality checks

The bodies keep the frame of the meshes (world coordinates of the input image, RAS+):
joint and marker positions in the ``.osim`` are world coordinates in metres. STAPLE is
equivariant to rigid transforms, so the model is the one MATLAB STAPLE builds from the
same bones in any other rigid frame (checked by ``tools/skeleton_parity.py``).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from mskpipe.config import PipelineConfig
from mskpipe.core.manifest import package_info
from mskpipe.core.registry import PluginError, Registry, default_registry
from mskpipe.core.step import Step, StepContext, StepError
from mskpipe.io.osim import OsimError, read_osim

if TYPE_CHECKING:
    from mskpipe.io.osim import OsimModel

SKELETON_FILE = "skeleton.json"
FORMAT = "mskpipe.skeleton"
VERSION = 1
COORDINATES = {"space": "world_ras", "units": "m", "geometry_units": "mm"}

# Distributions whose version changes a backend's outputs (part of the step fingerprint).
BACKEND_PACKAGES: dict[str, tuple[str, ...]] = {
    "pystaple": ("pystaple", "fast-simplification"),
}
FEMUR_LENGTH_MM = (250.0, 600.0)  # plausible adult range (QC warning outside)


class SkeletonStep(Step):
    name: ClassVar[str] = "skeleton"
    version: ClassVar[str] = "1"
    config_sections: ClassVar[tuple[str, ...]] = ("skeleton",)

    def __init__(self, registry: Registry | None = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> Registry:
        return self._registry or default_registry()

    def fingerprint_extra(self, config: PipelineConfig) -> dict[str, Any]:
        backend = config.skeleton.backend
        packages = {}
        for dist in BACKEND_PACKAGES.get(backend, ()):
            info = package_info(dist)
            packages[dist] = info.commit or info.version
        return {"plugin": self.registry.identity("skeleton", backend), "packages": packages}

    def run(self, ctx: StepContext) -> None:
        from mskpipe.steps.mesh import mesh_paths

        cfg = ctx.config.skeleton
        try:
            plugin = self.registry.create("skeleton", cfg.backend)
        except PluginError as exc:
            raise StepError(str(exc)) from exc

        bones = mesh_paths(ctx.step_dir("mesh"), "bone")
        if not bones:
            raise StepError("No bone meshes from step 'mesh'")
        ctx.logger.info("[skeleton] %s from %s", cfg.backend, ", ".join(sorted(bones)))

        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                osim = plugin.build(ctx, bones, ctx.out_dir, plugin.Params())
            except (FileNotFoundError, ValueError, RuntimeError, NotImplementedError) as exc:
                raise StepError(f"{cfg.backend}: {exc}") from exc
        elapsed = time.perf_counter() - start
        backend_warnings = sorted({str(w.message) for w in caught})
        for msg in backend_warnings:
            ctx.logger.warning("[skeleton] %s", msg)

        try:
            model = read_osim(osim)
        except OsimError as exc:
            raise StepError(f"{cfg.backend} wrote an unreadable model: {exc}") from exc
        qc = skeleton_qc(model, cfg.side)
        for msg in qc["problems"]:
            ctx.logger.warning("[skeleton] QC: %s", msg)

        geometry = sorted(p for p in (ctx.out_dir / "Geometry").glob("*") if p.is_file())
        index = {
            "format": FORMAT,
            "version": VERSION,
            "coordinates": COORDINATES,
            "model": osim.relative_to(ctx.out_dir).as_posix(),
            "model_name": model.name,
            "osim_version": model.version,
            "geometry": [p.relative_to(ctx.out_dir).as_posix() for p in geometry],
            "bones": {n: p.name for n, p in sorted(bones.items())},
            "settings": cfg.model_dump(mode="json"),
            "time_s": round(elapsed, 3),
            "warnings": backend_warnings,
            "qc": qc,
        }
        index_path = ctx.out_dir / SKELETON_FILE
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for path in (osim, *geometry, index_path):
            ctx.record.add_output(path)

        ctx.record.metrics.update(
            {
                "backend": cfg.backend,
                "n_bodies": len(model.bodies),
                "n_joints": len(model.joints),
                "n_markers": len(model.markers),
                "n_warnings": len(backend_warnings),
                "skeleton_time_s": round(elapsed, 3),
                **{k: v for k, v in qc.items() if k != "problems"},
                "qc_problems": qc["problems"],
            }
        )
        ctx.logger.info(
            "[skeleton] %s: %d bodies, %d joints, %.1f s",
            osim.name,
            len(model.bodies),
            len(model.joints),
            elapsed,
        )
        if qc.get("side_consistent") is False:
            raise StepError(
                "The model is mirrored (hip joint on the wrong side of the pelvis): "
                "check the orientation (affine) of the input image and skeleton.side"
            )


def read_skeleton_index(skeleton_dir: Path) -> dict[str, Any]:
    """Load ``skeleton.json`` of a finished ``skeleton`` step."""
    path = Path(skeleton_dir) / SKELETON_FILE
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StepError(f"Skeleton index not found: {path}") from None
    if index.get("format") != FORMAT or index.get("version") != VERSION:
        raise StepError(f"Unsupported skeleton index: {path}")
    return index


def skeleton_qc(model: OsimModel, side: str) -> dict[str, Any]:
    """Anatomical plausibility of a hip model (lengths in mm).

    Assumes STAPLE's convention that body frames coincide with the mesh frame, so the
    joint offset translations are positions in one common frame. The pelvis frame is the
    child frame of ``ground_pelvis`` (ISB: x anterior, y superior, z right).
    """
    import numpy as np

    hip, knee = f"hip_{side}", f"knee_{side}"
    missing = [j for j in ("ground_pelvis", hip, knee) if j not in model.joints]
    if missing:
        return {"side": side, "problems": [f"joint '{j}' missing" for j in missing]}

    problems: list[str] = []
    pelvis = model.joints["ground_pelvis"].child
    hip_c = model.joints[hip].parent.translation * 1000.0
    knee_c = model.joints[knee].parent.translation * 1000.0
    hip_in_pelvis = pelvis.rotation.T @ (hip_c - pelvis.translation * 1000.0)
    lateral = hip_in_pelvis[2] if side == "r" else -hip_in_pelvis[2]
    qc: dict[str, Any] = {
        "side": side,
        "side_consistent": bool(lateral > 0),
        "hip_center_in_pelvis_mm": [round(float(v), 3) for v in hip_in_pelvis],
        "femur_length_mm": round(float(np.linalg.norm(hip_c - knee_c)), 3),
    }
    if "RASI" in model.markers and "LASI" in model.markers:
        width = model.markers["RASI"].location - model.markers["LASI"].location
        qc["asis_width_mm"] = round(float(np.linalg.norm(width)) * 1000.0, 3)

    if not qc["side_consistent"]:
        problems.append(f"{hip} on the wrong side of the pelvis (z = {hip_in_pelvis[2]:.1f} mm)")
    lo, hi = FEMUR_LENGTH_MM
    if not lo <= qc["femur_length_mm"] <= hi:
        problems.append(f"femur length {qc['femur_length_mm']:.0f} mm outside {lo:.0f}-{hi:.0f} mm")
    qc["problems"] = problems
    return qc
