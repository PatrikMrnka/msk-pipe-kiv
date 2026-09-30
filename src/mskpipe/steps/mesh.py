# SPDX-License-Identifier: Apache-2.0
"""Step ``mesh``: label map -> one surface mesh per structure.

Input (``02_labelmap/``): ``labelmap.nii.gz`` and ``labels.json``
(see :mod:`mskpipe.io.labels`).

Output (``03_mesh/``)::

    bones/<name>.stl        pystaple input
    muscles/<name>.obj      Muscle Wrapping 2.x input
    meshes.json             index: file, parameters and metrics of every structure

Coordinates are world coordinates of the input image (NIfTI RAS+, mm).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, ClassVar

from mskpipe.config import PipelineConfig
from mskpipe.core.step import Step, StepContext, StepError

MESHES_FILE = "meshes.json"
FORMAT = "mskpipe.meshes"
VERSION = 1
KIND_DIRS = {"bone": "bones", "muscle": "muscles"}
KIND_SUFFIX = {"bone": ".stl", "muscle": ".obj"}
COORDINATES = {"space": "world_ras", "units": "mm"}


class MeshStep(Step):
    name: ClassVar[str] = "mesh"
    version: ClassVar[str] = "1"
    config_sections: ClassVar[tuple[str, ...]] = ("mesh",)

    def fingerprint_extra(self, config: PipelineConfig) -> dict[str, Any]:
        return {"vtk": _vtk_version()}  # VTK 9.3 and 9.4 give (slightly) different meshes

    def run(self, ctx: StepContext) -> None:
        import numpy as np
        from scipy import ndimage

        from mskpipe.geometry.metrics import mesh_stats
        from mskpipe.geometry.surface import (
            SurfaceError,
            mask_to_surface,
            set_threads,
            vtk_version,
        )
        from mskpipe.io.labels import LABELMAP_FILE, LABELS_FILE, LabelTable, LabelTableError
        from mskpipe.io.mesh_io import write_mesh
        from mskpipe.io.nifti import NiftiError, load_labelmap

        src = ctx.step_dir("labelmap")
        try:
            table = LabelTable.load(src / LABELS_FILE)
            volume = load_labelmap(src / LABELMAP_FILE)
        except (LabelTableError, NiftiError) as exc:
            raise StepError(str(exc)) from exc
        if len(table) == 0:
            raise StepError(f"No structures in {LABELS_FILE}")

        set_threads(ctx.config.runtime.threads)
        data = volume.data
        boxes = ndimage.find_objects(data, max_label=max(label.value for label in table))
        entries: list[dict[str, Any]] = []

        for label in table:
            params = getattr(ctx.config.mesh, KIND_DIRS[label.kind])
            entry: dict[str, Any] = {"name": label.name, "kind": str(label.kind)}
            box = boxes[label.value - 1]
            if box is None:
                ctx.logger.warning("[mesh] %s: no voxels in the label map, skipped", label.name)
                entries.append({**entry, "status": "missing"})
                continue

            start = time.perf_counter()
            try:
                result = mask_to_surface(
                    data[box] == label.value,
                    volume.affine,
                    params,
                    offset=[s.start for s in box],
                    full_shape=data.shape,
                )
            except SurfaceError as exc:
                raise StepError(f"{label.name}: {exc}") from exc
            rel = Path(KIND_DIRS[label.kind]) / f"{label.name}{KIND_SUFFIX[label.kind]}"
            path = write_mesh(ctx.out_dir / rel, result.mesh)
            elapsed = time.perf_counter() - start
            ctx.record.add_output(path)

            stats = mesh_stats(result.mesh)
            entry.update(
                status="ok",
                file=rel.as_posix(),
                time_s=round(elapsed, 3),
                **result.info,
                **stats,
                volume_ratio=stats["volume"] / result.info["mask_volume"],
            )
            entries.append(entry)
            if not stats["closed"]:
                ctx.logger.warning(
                    "[mesh] %s: surface not closed (%d boundary, %d non-manifold edges)",
                    label.name,
                    stats["boundary_edges"],
                    stats["non_manifold_edges"],
                )
            if result.info["touches_border"]:
                ctx.logger.warning("[mesh] %s: cut by the image border", label.name)
            ctx.logger.info("[mesh] %s: %d faces, %.1f s", label.name, stats["n_faces"], elapsed)

        ok = [e for e in entries if e["status"] == "ok"]
        if not ok:
            raise StepError("No structure produced a mesh (all labels empty)")

        index = {
            "format": FORMAT,
            "version": VERSION,
            "coordinates": COORDINATES,
            "vtk": vtk_version(),
            "params": ctx.config.mesh.model_dump(mode="json"),
            "structures": entries,
        }
        index_path = ctx.out_dir / MESHES_FILE
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        ctx.record.add_output(index_path)

        times = np.array([e["time_s"] for e in ok])
        ctx.record.metrics.update(
            {
                "vtk": vtk_version(),
                "n_structures": len(entries),
                "n_meshed": len(ok),
                "missing": [e["name"] for e in entries if e["status"] == "missing"],
                "not_closed": [e["name"] for e in ok if not e["closed"]],
                "touching_border": [e["name"] for e in ok if e["touches_border"]],
                "total_faces": int(sum(e["n_faces"] for e in ok)),
                "mesh_time_s": round(float(times.sum()), 3),
                "structures": {e["name"]: _summary(e) for e in ok},
            }
        )


def read_mesh_index(mesh_dir: Path) -> dict[str, Any]:
    """Load ``meshes.json`` of a finished ``mesh`` step."""
    path = Path(mesh_dir) / MESHES_FILE
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StepError(f"Mesh index not found: {path}") from None
    if index.get("format") != FORMAT or index.get("version") != VERSION:
        raise StepError(f"Unsupported mesh index: {path}")
    return index


def mesh_paths(mesh_dir: Path, kind: str) -> dict[str, Path]:
    """Name -> mesh file of all meshed structures of ``kind`` (``bone`` or ``muscle``)."""
    index = read_mesh_index(mesh_dir)
    return {
        s["name"]: Path(mesh_dir) / s["file"]
        for s in index["structures"]
        if s["kind"] == kind and s["status"] == "ok"
    }


def _summary(entry: dict[str, Any]) -> dict[str, Any]:
    keys = ("kind", "n_faces", "area", "volume", "volume_ratio", "closed", "touches_border")
    out = {k: entry[k] for k in keys}
    out["time_s"] = entry["time_s"]
    for k in ("area", "volume", "volume_ratio"):
        out[k] = round(out[k], 4)
    return out


def _vtk_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("vtk")
    except PackageNotFoundError:
        return "missing"
