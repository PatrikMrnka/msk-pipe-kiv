# SPDX-License-Identifier: Apache-2.0
"""Step ``mesh``: label map -> one checked surface mesh per structure.

Input (``02_labelmap/``): ``labelmap.nii.gz`` and ``labels.json``
(see :mod:`mskpipe.io.labels`).

Output (``03_mesh/``)::

    bones/<name>.stl        pystaple input
    muscles/<name>.obj      Muscle Wrapping 2.x input
    meshes.json             index: file, parameters, metrics, check and repair of every
                            structure (``valid``, ``problems``, ``repair``)
    defects.json            bone surfaces that were repaired, are invalid, or whose genus
                            differs from the anatomy (if any)

Coordinates are world coordinates of the input image (NIfTI RAS+, mm).

This step is where surfaces are guaranteed: every structure is checked as the steps after
it need it and repaired here if it fails (:mod:`mskpipe.geometry.checked`; bones closed
and manifold for pystaple, muscles in addition one part with genus 0 for Muscle Wrapping).
The steps after only read ``valid`` and never repair. A bone that stays invalid stops the
run here (``mesh.on_invalid``); an invalid muscle is excluded or kept by ``export_mw2``.
Surfaces that pass the check are the BP meshes (quadric decimation), unchanged.
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
    # 2: genus check; 3: every surface checked and repaired here; 4: local patch of bones
    version: ClassVar[str] = "4"
    config_sections: ClassVar[tuple[str, ...]] = ("mesh",)

    def fingerprint_extra(self, config: PipelineConfig) -> dict[str, Any]:
        return {"vtk": _vtk_version()}  # VTK 9.3 and 9.4 give (slightly) different meshes

    def run(self, ctx: StepContext) -> None:
        import numpy as np
        from scipy import ndimage

        from mskpipe.geometry.checked import brief, build_surface
        from mskpipe.geometry.metrics import mesh_stats
        from mskpipe.geometry.surface import SurfaceError, set_threads, vtk_version
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
        defects: list[Any] = []

        for label in table:
            ctx.check_cancel()
            kind = str(label.kind)
            params = getattr(ctx.config.mesh, KIND_DIRS[kind])
            entry: dict[str, Any] = {"name": label.name, "kind": kind}
            box = boxes[label.value - 1]
            if box is None:
                ctx.logger.warning("[mesh] %s: no voxels in the label map, skipped", label.name)
                entries.append({**entry, "status": "missing"})
                continue

            start = time.perf_counter()
            try:
                build = build_surface(
                    data[box] == label.value,
                    volume.affine,
                    kind,
                    params,
                    ctx.config.mesh.repair,
                    offset=[s.start for s in box],
                    full_shape=data.shape,
                    check_cancel=ctx.check_cancel,
                )
            except SurfaceError as exc:
                raise StepError(f"{label.name}: {exc}") from exc
            rel = Path(KIND_DIRS[kind]) / f"{label.name}{KIND_SUFFIX[kind]}"
            path = write_mesh(ctx.out_dir / rel, build.mesh)
            elapsed = time.perf_counter() - start
            ctx.record.add_output(path)

            stats = mesh_stats(build.mesh)
            entry.update(
                status="ok",
                file=rel.as_posix(),
                time_s=round(elapsed, 3),
                decimation=build.decimation,
                **build.info,
                **stats,
                volume_ratio=stats["volume"] / build.info["mask_volume"],
                valid=build.valid,
                problems=build.problems,
                check=brief(build.check),
            )
            if build.repair is not None:
                entry["repaired"] = build.repaired
                entry["repair"] = build.repair
            _log_build(ctx, label.name, build)
            if kind == "bone":
                entry["genus"], found = _bone_defects(ctx, label.name, build)
                defects += found
            entries.append(entry)
            if build.info["touches_border"]:
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
        from mskpipe.qc.defects import DEFECTS_FILE, DefectReport

        (ctx.out_dir / DEFECTS_FILE).unlink(missing_ok=True)
        if defects:
            ctx.record.add_output(DefectReport(defects, {"step": "mesh"}).write(ctx.out_dir))
        index_path = ctx.out_dir / MESHES_FILE
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        ctx.record.add_output(index_path)

        times = np.array([e["time_s"] for e in ok])
        invalid = [e["name"] for e in ok if not e["valid"]]
        ctx.record.metrics.update(
            {
                "vtk": vtk_version(),
                "n_structures": len(entries),
                "n_meshed": len(ok),
                "missing": [e["name"] for e in entries if e["status"] == "missing"],
                "not_closed": [e["name"] for e in ok if not e["closed"]],
                "repaired": [e["name"] for e in ok if e.get("repaired")],
                "invalid": invalid,
                "touching_border": [e["name"] for e in ok if e["touches_border"]],
                "total_faces": int(sum(e["n_faces"] for e in ok)),
                "mesh_time_s": round(float(times.sum()), 3),
                "structures": {e["name"]: _summary(e) for e in ok},
                "genus_defects": [d.structure for d in defects if d.method == "check only"],
            }
        )
        bad_bones = [e for e in ok if e["kind"] == "bone" and not e["valid"]]
        if bad_bones and ctx.config.mesh.on_invalid == "fail":
            raise StepError(
                "bone surface unusable after repair: "
                + "; ".join(f"{e['name']} ({', '.join(e['problems'])})" for e in bad_bones)
                + ". Inspect with `mskpipe qc <run> --show`; mesh.on_invalid=warn continues."
            )


def _log_build(ctx: StepContext, name: str, build: Any) -> None:
    if build.repair is None:
        return
    info = build.repair
    if build.repaired:
        ctx.logger.warning(
            "[mesh] %s repaired (%s): volume x%.3f, mean change %.2f mm, Hausdorff %.1f mm",
            name,
            info["used"],
            info.get("volume_ratio") or 1.0,
            info["distance_to_original_mm"]["mean"],
            info["distance_to_original_mm"]["hausdorff"],
        )
    else:
        ctx.logger.warning(
            "[mesh] %s could not be repaired (%d attempts; %s)",
            name,
            len(info["attempts"]),
            info.get("best_summary", "no attempt"),
        )


def _bone_defects(ctx: StepContext, name: str, build: Any) -> tuple[list[int | None], list]:
    """Genus per component and the defects of a bone surface: repaired or still invalid
    (located at its open / non-manifold edges), genus above the anatomy."""
    from mskpipe.qc.defects import Defect

    out = []
    if build.repair is not None:
        info = build.repair
        detail = "surface failed the check: " + ", ".join(info["original_problems"])
        if not build.valid:
            detail += "; left after the repair: " + ", ".join(build.problems)
        out.append(
            Defect(
                structure=name,
                kind="surface",
                center_ijk=None,
                center_mm=info["problem_center_mm"],
                size_mm=None,
                fixed=build.valid,
                method=info.get("used", "none") if build.valid else "repair failed",
                detail=detail,
                step="mesh",
            )
        )
    genera, defect = _genus_check(name, build.mesh)
    if defect is not None:
        out.append(defect)
    for d in out:
        ctx.logger.warning("[mesh] %s | %s", d.message(), d.detail)
    return genera, out


# Anatomical genus of bone surfaces (sum over components); others are not checked (sacrum
# foramina vary). pelvis_no_sacrum = two hip bones with one obturator foramen each.
EXPECTED_SURFACE_GENUS = {"pelvis_no_sacrum": 2, "pelvis_with_sacrum": None}


def _expected_genus(name: str) -> int | None:
    if name in EXPECTED_SURFACE_GENUS:
        return EXPECTED_SURFACE_GENUS[name]
    return 0 if name.split("_")[0] in ("femur", "tibia", "fibula", "patella") else None


def _genus_check(name: str, mesh: Any) -> tuple[list[int | None], Any]:
    """Genus per component and a ``surface`` defect if a component is not closed or the
    total exceeds the anatomy (centred on the worst component)."""

    import numpy as np

    from mskpipe.geometry.topology import component_genus, components
    from mskpipe.qc.defects import Defect

    genera = component_genus(mesh)
    expected = _expected_genus(name)
    open_parts = [i for i, g in enumerate(genera) if g is None]
    known = [g for g in genera if g is not None]
    if open_parts:
        worst = open_parts[0]
        problem = f"surface not closed ({len(open_parts)} of {len(genera)} components)"
    elif expected is not None and known and sum(known) > expected:
        worst = int(np.argmax(genera))
        problem = (
            f"surface genus {sum(known)} > anatomical {expected} (handles or holes left "
            "after meshing)"
        )
    else:
        return genera, None
    labels = components(mesh)
    part = mesh.vertices[np.unique(mesh.faces[labels == np.unique(labels)[worst]])]
    centre = part.mean(axis=0)
    return genera, Defect(
        structure=name,
        kind="surface",
        center_ijk=None,
        center_mm=[round(float(c), 2) for c in centre],
        size_mm=None,
        fixed=False,
        method="check only",
        detail=f"{problem}; genus per component: {genera}; component {worst} centred at "
        f"({centre[0]:.0f}, {centre[1]:.0f}, {centre[2]:.0f}) mm",
        step="mesh",
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
