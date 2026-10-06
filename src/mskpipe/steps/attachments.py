# SPDX-License-Identifier: Apache-2.0
"""Step ``attachments``: muscle attachment areas on the subject's bones.

Inputs: ``03_mesh/`` bone and muscle meshes (side ``skeleton.side``); the tibia body is
tibia + fibula as in step ``skeleton``, the pelvis body ``pelvis_with_sacrum`` =
``pelvis_no_sacrum`` + ``sacrum`` when the sacrum was segmented (sacral attachments of
gluteus maximus and piriformis).

Output (``05_attachments/``)::

    <muscle>/<Stem>_Ori.vtk, <Stem>_Ins.vtk   closed outlines, mm, world RAS (Muscle Wrapping)
    registration/<bone>.stl                   atlas bones after registration (bone_registration)
    bodies/tibia_<s>.stl                      tibia + fibula given to the method
    bodies/pelvis_with_sacrum.stl             pelvis + sacrum given to the method
    attachments.json                          index: method, atlas, registration, areas
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, ClassVar

from mskpipe.config import PipelineConfig
from mskpipe.core.registry import PluginError, Registry, default_registry
from mskpipe.core.step import Step, StepContext, StepError

ATTACHMENTS_FILE = "attachments.json"
FORMAT = "mskpipe.attachments"
VERSION = 1
COORDINATES = {"space": "world_ras", "units": "mm"}


class AttachmentsStep(Step):
    name: ClassVar[str] = "attachments"
    version: ClassVar[str] = "1"
    config_sections: ClassVar[tuple[str, ...]] = ("attachments", "skeleton")

    def __init__(self, registry: Registry | None = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> Registry:
        return self._registry or default_registry()

    def fingerprint_extra(self, config: PipelineConfig) -> dict[str, Any]:
        method = config.attachments.method
        try:
            extra = dict(self.registry.get("attachments", method).cache_identity(config))
        except PluginError as exc:  # reported by run(); keep the fingerprint computable
            extra = {"error": str(exc)}
        return {"plugin": self.registry.identity("attachments", method), **extra}

    def run(self, ctx: StepContext) -> None:
        from mskpipe.steps.mesh import mesh_paths
        from mskpipe.steps.skeleton import body_meshes

        cfg = ctx.config.attachments
        side = ctx.config.skeleton.side
        try:
            plugin = self.registry.create("attachments", cfg.method)
            params = self.registry.validate_params(
                "attachments", cfg.method, cfg.params, where="attachments"
            )
        except PluginError as exc:
            raise StepError(str(exc)) from exc

        mesh_dir = ctx.step_dir("mesh")
        bones = mesh_paths(mesh_dir, "bone")
        if not bones:
            raise StepError("No bone meshes from step 'mesh'")
        bones, sources = body_meshes(bones, side, True, ctx.out_dir / "bodies")
        bones, sources = pelvis_body(bones, sources, ctx.out_dir / "bodies")
        if "sacrum" not in sources:
            ctx.logger.warning("[attachments] no sacrum mesh: sacral areas cannot be snapped")
        muscles = {
            n: p for n, p in mesh_paths(mesh_dir, "muscle").items() if n.endswith(f"_{side}")
        }
        if not muscles:
            raise StepError(f"No muscle meshes of side '{side}' from step 'mesh'")
        ctx.logger.info("[attachments] %s, %d muscles, side %s", cfg.method, len(muscles), side)

        start = time.perf_counter()
        try:
            outputs = plugin.compute(ctx, bones, muscles, ctx.out_dir, params)
        except (OSError, ValueError, RuntimeError, NotImplementedError) as exc:
            raise StepError(f"{cfg.method}: {exc}") from exc
        elapsed = time.perf_counter() - start

        transfer_path = ctx.out_dir / "transfer.json"
        transfer = (
            json.loads(transfer_path.read_text(encoding="utf-8")) if transfer_path.is_file() else {}
        )
        files = {
            m: sorted(p.relative_to(ctx.out_dir).as_posix() for p in Path(d).glob("*.vtk"))
            for m, d in sorted(outputs.items())
        }
        index = {
            "format": FORMAT,
            "version": VERSION,
            "coordinates": COORDINATES,
            "method": cfg.method,
            "params": params.model_dump(mode="json"),
            "side": side,
            "body_geometry_sources": sources,
            "muscles": files,
            "time_s": round(elapsed, 3),
            **{k: v for k, v in transfer.items() if k != "side"},
        }
        index_path = ctx.out_dir / ATTACHMENTS_FILE
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for paths in files.values():
            for rel in paths:
                ctx.record.add_output(ctx.out_dir / rel)
        ctx.record.add_output(index_path)

        areas = transfer.get("areas", [])
        ok = [a for a in areas if a.get("status") == "ok"]
        failed = [f"{a['muscle']}:{a['kind']}:{a['status']}" for a in areas if a["status"] != "ok"]
        ctx.record.metrics.update(
            {
                "n_muscles": len(files),
                "n_areas": len(ok),
                "areas_failed": failed,
                "not_in_atlas": transfer.get("not_in_atlas", []),
                "not_meshed": transfer.get("not_meshed", []),
                "registration": {
                    b: {k: v for k, v in r.items() if k in ("scale", "rigid", "nonrigid", "status")}
                    for b, r in transfer.get("bones", {}).items()
                },
                "min_snapped": min((a["snapped"] for a in ok), default=None),
                "attachments_time_s": round(elapsed, 3),
            }
        )
        for msg in failed:
            ctx.logger.warning("[attachments] %s", msg)
        if not ok:
            raise StepError("No attachment area was produced")


def pelvis_body(
    bones: dict[str, Path], sources: dict[str, list[str]], work_dir: Path
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Add ``pelvis_with_sacrum`` (both hip bones + sacrum) when both meshes exist."""
    import numpy as np

    from mskpipe.io.mesh_io import TriMesh, read_mesh, write_mesh

    parts = ("pelvis_no_sacrum", "sacrum")
    if not all(p in bones for p in parts):
        return bones, sources
    meshes = [read_mesh(bones[p]) for p in parts]
    offset = meshes[0].n_vertices
    merged = TriMesh(
        np.vstack([meshes[0].vertices, meshes[1].vertices]),
        np.vstack([meshes[0].faces, meshes[1].faces + offset]),
    )
    bones = {**bones, "pelvis_with_sacrum": write_mesh(work_dir / "pelvis_with_sacrum.stl", merged)}
    return bones, {**sources, "pelvis_with_sacrum": list(parts)}


def read_attachments_index(folder: Path) -> dict[str, Any]:
    path = Path(folder) / ATTACHMENTS_FILE
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StepError(f"Attachments index not found: {path}") from None
    if index.get("format") != FORMAT or index.get("version") != VERSION:
        raise StepError(f"Unsupported attachments index: {path}")
    return index
