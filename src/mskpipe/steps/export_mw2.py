# SPDX-License-Identifier: Apache-2.0
"""Step ``export_mw2``: input folder for Muscle Wrapping 2.x (``OsimMuscleGeneratorTool``).

Inputs: ``03_mesh/`` (bone and muscle meshes), ``02_labelmap/`` (only to remesh muscles
that need repair), ``04_skeleton/`` (``.osim``), ``05_attachments/`` (outlines).

Output (``06_mw2_input/``)::

    setup_MuscleGeneratorTool.xml      run: opensim-cmd -L OsimMuscleGeneratorTool.dll
                                            run-tool setup_MuscleGeneratorTool.xml
    <model_name>.osim                  skeleton model, rest pose = pose of the image
    <coordinate>.mot                   first row = rest pose, then the swept coordinate
    Geometry/<body>.obj                bone geometry per body (mm)
    muscles/<muscle>/<Stem>.obj        muscle surface (mm), checked / repaired
    muscles/<muscle>/<Stem>_Ori.vtk    origin outline (mm), inflated when degenerate
    muscles/<muscle>/<Stem>_Ins.vtk    insertion outline (mm), inflated when degenerate
    analysis/, fibres/                 output folders of Muscle Wrapping (when enabled)
    export.json                        index: pose, checks, repairs, exclusions

Coordinates: the OpenSim frame (Y up), i.e. the world frame of the input image (NIfTI
RAS+) rotated by ``RAS_TO_OPENSIM`` (:mod:`mskpipe.geometry.frames`: X anterior,
Y superior, Z right; ``coordinates.rotation_from_ras`` in ``export.json``). The steps
before stay in RAS: the skeleton model, the bone and muscle meshes and the attachment
outlines are rotated here, and the rest pose, the child-frame alignment and the outline
checks run on the rotated data. Files in mm with
``scale_factors 0.001``, model in metres. Muscle Wrapping places a point at
``X_G,body(t) S p``: the meshes are valid only if every body frame equals the mesh frame
in the first motion frame. pystaple bodies use the mesh frame, so the rest pose is the
*image pose* (all bodies on the ground frame, :func:`~mskpipe.geometry.kinematics.image_pose`),
written as coordinate defaults and as the first row of the motion. Joints that cannot reach
it (the 1-DOF STAPLE knee) get their child frame rotated (``export.align_child_frames``).

Every muscle mesh is checked as Muscle Wrapping needs it (closed 2-manifold, one
component, genus 0) and repaired if needed; muscles whose mesh fails are excluded or kept
(``export.on_invalid``), with the reasons in ``export.json``. The attachment outlines are
measured after projection onto the muscle (gap = missing tendon, collapse, crossing,
:mod:`mskpipe.geometry.outline`) and reported as warnings only: they never exclude a muscle.
An outline whose projection is degenerate (fewer than 3 distinct points or a span below two
mean edge lengths; Muscle Wrapping would cut an empty patch and crash) is replaced by the
outline of a small patch of the muscle surface around it
(:mod:`mskpipe.geometry.attachment_guard`, ``export.inflate_radius_mm``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, ClassVar

from mskpipe.config import PipelineConfig
from mskpipe.core.step import Step, StepContext, StepError

if TYPE_CHECKING:
    import numpy as np

    from mskpipe.io.mesh_io import TriMesh
    from mskpipe.io.osim import OsimModel

EXPORT_FILE = "export.json"
SETUP_FILE = "setup_MuscleGeneratorTool.xml"
FORMAT = "mskpipe.mw2_input"
VERSION = 1
COORDINATES = {"space": "opensim_yup", "units": "mm", "model_units": "m", "scale_factors": 0.001}
GEOMETRY_DIR = "Geometry"
MUSCLES_DIR = "muscles"
ANALYSIS_DIR = "analysis/"
FIBRES_DIR = "fibres/"
POSE_TOL_DEG = 1e-6
KINDS = {"Ori": "origin", "Ins": "insertion"}
MW2_COMMAND = "opensim-cmd -L OsimMuscleGeneratorTool.dll run-tool " + SETUP_FILE

# Muscles excluded by default (export.exclude), with the reason written to export.json.
# All four reach their bony insertion through a tendon or fascia that the segmentation
# does not capture and that runs around a bony pulley (or, for TFL, down to the tibia),
# so the insertion outline cannot be placed on the segmented belly: Muscle Wrapping 2.x
# projects it onto the closest part of the belly, where it collapses, and Luca2018 then
# moves the fibre ends with the wrong bone. On LHDL, MW2 crashed on three of them and gave
# an anatomically wrong decomposition for the fourth.
EXCLUSION_REASONS = {
    "piriformis": "the tendon to the tip of the greater trochanter is not segmented and "
    "the muscle wraps over the greater sciatic notch; its insertion projects onto the belly "
    "next to the pelvis (collapses, fibre ends move with the pelvis, not the femur)",
    "obturator_internus": "the belly lies inside the pelvis; the tendon turns about 90 deg "
    "around the lesser sciatic notch to the trochanteric fossa and is not segmented; the "
    "insertion projects onto the belly inside the pelvis (collapses, fibre ends move with "
    "the pelvis, not the femur)",
    "obturator_externus": "the tendon wraps under the femoral neck to the trochanteric fossa "
    "and is not segmented; the insertion projects onto a small region of the belly "
    "(collapses)",
    "tensor_fasciae_latae": "inserts on the lateral tibia (Gerdy's tubercle) through the "
    "iliotibial tract, a fascial band that is not segmented; the insertion lies hundreds of "
    "mm below the belly (collapses, fibre ends move with the femur, not the tibia)",
}


class ExportMw2Step(Step):
    name: ClassVar[str] = "export_mw2"
    version: ClassVar[str] = "3"  # 3: OpenSim frame (Y up), inflated degenerate outlines
    config_sections: ClassVar[tuple[str, ...]] = ("export",)

    def fingerprint_extra(self, config: PipelineConfig) -> dict[str, Any]:
        from mskpipe.steps.mesh import _vtk_version

        return {"vtk": _vtk_version()}  # remeshing of muscles

    def run(self, ctx: StepContext) -> None:
        from mskpipe.geometry.frames import RAS_TO_OPENSIM, FrameError, rotate_osim_file
        from mskpipe.io.osim import OsimError, read_osim
        from mskpipe.steps.attachments import read_attachments_index
        from mskpipe.steps.mesh import mesh_paths
        from mskpipe.steps.skeleton import read_skeleton_index

        start = time.perf_counter()
        cfg = ctx.config.export
        side = ctx.config.skeleton.side
        out = ctx.out_dir

        skel_dir = ctx.step_dir("skeleton")
        skel = read_skeleton_index(skel_dir)
        model_src = skel_dir / skel["model"]
        model_dst = out / model_src.name
        try:  # everything below works on the model in the OpenSim frame
            rotated = rotate_osim_file(model_src, model_dst, RAS_TO_OPENSIM)
            model = read_osim(model_dst)
        except (OsimError, FrameError) as exc:
            raise StepError(f"skeleton model: {exc}") from exc
        ctx.logger.info(
            "[export_mw2] model rotated to the OpenSim frame (%d frames, %d on ground kept)",
            rotated["frames"],
            rotated["frames_on_ground"],
        )
        times = {"model_rotation": time.perf_counter() - start}
        attachments = read_attachments_index(ctx.step_dir("attachments"))
        mesh_dir = ctx.step_dir("mesh")
        bones = mesh_paths(mesh_dir, "bone")
        muscle_meshes = {
            n: p for n, p in mesh_paths(mesh_dir, "muscle").items() if n.endswith(f"_{side}")
        }

        mark = time.perf_counter()
        geometry, bodies = self._bodies(ctx, model, bones, skel_dir)
        times["bodies"] = time.perf_counter() - mark
        mark = time.perf_counter()
        pose_info, rest, rotational = self._model(ctx, model, model_dst, geometry)
        motion_path, motion_info = self._motion(ctx, rest, rotational, side)
        times["rest_pose"] = time.perf_counter() - mark
        ctx.check_cancel()

        selected = sorted(muscle_meshes) if cfg.muscles == "all" else list(cfg.muscles)
        areas = attachment_areas(attachments, ctx.step_dir("attachments"))
        records, specs = [], []
        repairer = _Repairer(ctx)
        locators = _BodyLocators(geometry)
        mark = time.perf_counter()
        for name in selected:
            ctx.check_cancel()
            reason = excluded_reason(name, cfg.exclude)
            rec, spec = self._muscle(ctx, name, muscle_meshes, areas, repairer, locators, reason)
            records.append(rec)
            if spec is not None:
                specs.append(spec)
        times["muscles"] = time.perf_counter() - mark
        times["inflate"] = sum(r.pop("inflate_time_s", 0.0) for r in records)

        from mskpipe.io.mw2 import SetupSpec, write_setup

        exported = [r["name"] for r in records if r["status"] != "excluded"]
        setup_path = out / SETUP_FILE
        if specs:
            if cfg.muscle_analysis:
                (out / ANALYSIS_DIR).mkdir(exist_ok=True)
            if cfg.fibre_export:
                (out / FIBRES_DIR).mkdir(exist_ok=True)
            write_setup(
                setup_path,
                SetupSpec(
                    name=f"mskpipe_{ctx.input.subject_id}",
                    model_file=model_dst.name,
                    motion_file=motion_path.name,
                    output_model_file=f"{model_dst.stem}_mw2.osim",
                    muscles=specs,
                    coordinate=motion_info["coordinate"],
                    num_of_lines=cfg.num_of_lines,
                    line_res=cfg.line_res,
                    decomposition_method=cfg.decomposition_method,
                    bone_weights=cfg.bone_weights,
                    analysis_prefix=ANALYSIS_DIR if cfg.muscle_analysis else None,
                    export_folder=FIBRES_DIR if cfg.fibre_export else None,
                ),
            )
        elapsed = time.perf_counter() - start
        times_s = {k: round(v, 3) for k, v in times.items()} | {"total": round(elapsed, 3)}

        index = {
            "format": FORMAT,
            "version": VERSION,
            "coordinates": {**COORDINATES, "rotation_from_ras": RAS_TO_OPENSIM.tolist()},
            "setup": SETUP_FILE if specs else None,
            "command": MW2_COMMAND,
            "model": model_dst.name,
            "model_source": model_src.relative_to(ctx.ws.root).as_posix(),
            "motion": motion_info,
            "rest_pose": pose_info,
            "bodies": bodies,
            "muscles": records,
            "settings": cfg.model_dump(mode="json"),
            "time_s": round(elapsed, 3),
            "times_s": times_s,
        }
        index_path = out / EXPORT_FILE
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for path in sorted(p for p in out.rglob("*") if p.is_file()):
            ctx.record.add_output(path)

        excluded = {r["name"]: r["problems"] for r in records if r["status"] == "excluded"}
        kept = [r["name"] for r in records if r["status"] == "kept_invalid"]
        ctx.record.metrics.update(
            {
                "n_selected": len(records),
                "n_exported": len(exported),
                "excluded": excluded,
                "kept_invalid": kept,
                "excluded_by_config": sorted(
                    r["name"] for r in records if r.get("excluded_by_config")
                ),
                "outline_warnings": {
                    r["name"]: r["warnings"] for r in records if r.get("warnings")
                },
                "repaired": [r["name"] for r in records if r["mesh"].get("repaired")],
                "rest_pose_deg": pose_info["values"],
                "joint_residual_deg": {
                    j: round(v["rotation_deg"], 4) for j, v in pose_info["joints"].items()
                },
                "aligned_child_frames": pose_info["aligned_child_frames"],
                "rest_pose_max_displacement_mm": {
                    b: v.get("max_displacement_mm") for b, v in pose_info["errors"].items()
                },
                "motion_frames": motion_info["n_frames"],
                "tendon_gap_mean_mm": {
                    r["name"]: {k: a.get("gap_mean_mm") for k, a in r["areas"].items()}
                    for r in records
                    if r["areas"]
                },
                "luca_body_mismatch": [
                    f"{r['name']}:{k}"
                    for r in records
                    if r["status"] != "excluded"
                    for k, a in r["areas"].items()
                    if a.get("nearest_body") not in (None, a.get("body"))
                ],
                "inflated": {
                    r["name"]: inflated
                    for r in records
                    if (
                        inflated := {
                            k: a["inflated"] for k, a in r["areas"].items() if "inflated" in a
                        }
                    )
                },
                "coordinates_space": COORDINATES["space"],
                "times_s": times_s,
                "export_time_s": round(elapsed, 3),
            }
        )
        for name, problems in excluded.items():
            ctx.logger.warning("[export_mw2] %s excluded: %s", name, "; ".join(problems))
        for name in kept:
            ctx.logger.warning("[export_mw2] %s kept despite failed checks", name)
        ctx.logger.info(
            "[export_mw2] %d/%d muscles exported, %.1f s", len(exported), len(records), elapsed
        )
        if not specs:
            raise StepError(f"No muscle passed the Muscle Wrapping checks (see {EXPORT_FILE})")

    # ------------------------------------------------------------------ bodies

    def _bodies(
        self, ctx: StepContext, model: OsimModel, bones: dict[str, Path], skel_dir: Path
    ) -> tuple[dict[str, TriMesh], dict[str, Any]]:
        """Bone geometry per model body: full-resolution meshes, merged as configured and
        rotated to the OpenSim frame."""
        import numpy as np

        from mskpipe.geometry.frames import RAS_TO_OPENSIM, homogeneous
        from mskpipe.io.mesh_io import TriMesh, read_mesh, write_mesh

        cfg, side = ctx.config.export, ctx.config.skeleton.side
        geometry: dict[str, TriMesh] = {}
        info: dict[str, Any] = {}
        for body in model.bodies.values():
            if not body.mesh_files:
                continue
            stem = Path(PureWindowsPath(body.mesh_files[0]).name).stem
            parts = [stem]
            if stem == "pelvis_no_sacrum" and cfg.pelvis_with_sacrum and "sacrum" in bones:
                parts.append("sacrum")
            if (
                stem == f"tibia_{side}"
                and ctx.config.skeleton.include_fibula
                and f"fibula_{side}" in bones
            ):
                parts.append(f"fibula_{side}")
            if all(p in bones for p in parts):
                meshes = [read_mesh(bones[p]) for p in parts]
                source = [f"mesh:{p}" for p in parts]
            else:  # not a pipeline bone: take the model's own geometry
                fallback = skel_dir.joinpath(*PureWindowsPath(body.mesh_files[0]).parts)
                if not fallback.is_file():
                    raise StepError(f"No geometry for body '{body.name}' ({body.mesh_files[0]})")
                meshes, source = [read_mesh(fallback)], [f"skeleton:{body.mesh_files[0]}"]
            offsets = np.cumsum([0] + [m.n_vertices for m in meshes[:-1]])
            merged = TriMesh(
                np.vstack([m.vertices for m in meshes]),
                np.vstack([m.faces + o for m, o in zip(meshes, offsets, strict=True)]),
            ).transformed(homogeneous(RAS_TO_OPENSIM))
            rel = f"{GEOMETRY_DIR}/{body.name}.obj"
            write_mesh(ctx.out_dir / rel, merged)
            geometry[body.name] = merged
            info[body.name] = {"file": rel, "sources": source, "n_faces": merged.n_faces}
            if len(parts) > 1:
                ctx.logger.info("[export_mw2] %s geometry = %s", body.name, " + ".join(parts))
        if not geometry:
            raise StepError("The skeleton model has no body geometry")
        return geometry, info

    # ------------------------------------------------------------------ model and pose

    def _model(
        self,
        ctx: StepContext,
        model: OsimModel,
        path: Path,
        geometry: dict[str, TriMesh],
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool]]:
        """Put the (rotated) model ``path`` into the image pose in place; returns its info,
        the rest values of the joint coordinates below the pelvis (motion columns, deg/m)
        and their types."""
        from mskpipe.geometry.kinematics import (
            KinematicsError,
            frame_matrix,
            image_pose,
            joint_order,
            joint_transform,
            pose_errors,
        )
        from mskpipe.io.mw2 import orientation_angles, write_model
        from mskpipe.io.osim import OsimError, read_osim

        try:
            pose = image_pose(model)
            order = joint_order(model)
        except (KinematicsError, OsimError) as exc:
            raise StepError(f"rest pose: {exc}") from exc

        points = {b: m.vertices / 1000.0 for b, m in geometry.items()}
        before = pose_errors(model, pose.values, points)
        frames, aligned = {}, []
        if ctx.config.export.align_child_frames:
            for j in order:
                if pose.joints[j.name]["rotation_deg"] > POSE_TOL_DEG:
                    x = frame_matrix(j.parent) @ joint_transform(j, pose.values)
                    frames[(j.name, j.child_frame)] = orientation_angles(x[:3, :3])
                    aligned.append(j.name)
                    ctx.logger.info(
                        "[export_mw2] %s cannot reach the image pose (%.2f deg): "
                        "child frame rotated",
                        j.name,
                        pose.joints[j.name]["rotation_deg"],
                    )
        ranges = {}
        for j in order:
            for c, (lo, hi) in j.coordinates.items():
                v = pose.values.get(c, 0.0)
                if not lo <= v <= hi:
                    ranges[c] = (min(lo, v), max(hi, v))
        try:
            write_model(
                path,
                path,
                mesh_files={b: f"{GEOMETRY_DIR}/{b}.obj" for b in geometry},
                defaults=pose.values,
                ranges=ranges,
                frame_orientations=frames,
            )
            edited = read_osim(path)
            after = pose_errors(edited, pose.values, points)
        except (ValueError, KinematicsError) as exc:
            raise StepError(f"model for Muscle Wrapping: {exc}") from exc

        worst = max((v.get("max_displacement_mm", 0.0) for v in after.values()), default=0.0)
        if worst > 1e-3:
            ctx.logger.warning(
                "[export_mw2] bones off the image pose in the first frame by up to %.1f mm",
                worst,
            )
        columns = [c for j in order if j.parent.parent_name != "ground" for c in j.coordinates]
        info = {
            "values": {k: round(v, 6) for k, v in pose.degrees().items()},
            "units": "deg (rotations), m (translations)",
            "joints": pose.joints,
            "aligned_child_frames": aligned,
            "ranges_extended": sorted(ranges),
            "errors_before_alignment": _round(before),
            "errors": _round(after),
        }
        degrees = pose.degrees()
        return info, {c: degrees[c] for c in columns}, pose.rotational

    def _motion(
        self, ctx: StepContext, rest: dict[str, float], rotational: dict[str, bool], side: str
    ) -> tuple[Path, dict[str, Any]]:
        import numpy as np

        from mskpipe.io.mw2 import write_motion

        mcfg = ctx.config.export.motion
        coord = mcfg.coordinate or f"hip_flexion_{side}"
        columns = list(rest)
        if coord not in columns or not rotational.get(coord, False):
            raise StepError(
                f"export.motion.coordinate '{coord}' is not a rotational joint coordinate "
                f"of the model (available: {', '.join(columns)})"
            )
        start = rest[coord]
        step = mcfg.step_deg
        first = (np.floor(start / step) + 1) * step
        if first - start < 0.25 * step:
            first += step
        sweep = np.arange(first, mcfg.end_deg + 1e-9, step)
        if len(sweep) == 0:
            raise StepError(
                f"export.motion.end_deg ({mcfg.end_deg}) must lie above the rest pose of "
                f"{coord} ({start:.2f} deg); the first motion row is the rest pose"
            )
        rows = np.tile([rest[c] for c in columns], (len(sweep) + 1, 1))
        rows[1:, columns.index(coord)] = sweep
        if any(not rotational.get(c, True) for c in columns):
            ctx.logger.warning("[export_mw2] translational coordinates written in metres")
        path = write_motion(ctx.out_dir / f"{coord}.mot", columns, rows, in_degrees=True)
        info = {
            "file": path.name,
            "coordinate": coord,
            "columns": columns,
            "rest_deg": round(float(start), 6),
            "start_deg": round(float(sweep[0]), 6),
            "end_deg": round(float(sweep[-1]), 6),
            "step_deg": step,
            "n_frames": len(rows),
        }
        return path, info

    # ------------------------------------------------------------------ muscles

    def _muscle(
        self,
        ctx: StepContext,
        name: str,
        meshes: dict[str, Path],
        areas: dict[str, dict[str, dict[str, Any]]],
        repairer: _Repairer,
        locators: _BodyLocators,
        excluded: str | None = None,
    ) -> tuple[dict[str, Any], Any]:
        """Check, repair and write one muscle. ``excluded``: reason from ``export.exclude``
        (the muscle is still measured for the record, but not exported)."""
        from mskpipe.geometry import attachment_guard as guard
        from mskpipe.geometry.compare import SurfaceLocator
        from mskpipe.geometry.frames import RAS_TO_OPENSIM, homogeneous, rotate_points
        from mskpipe.geometry.metrics import surface_area
        from mskpipe.geometry.outline import contour_distance, outline_report
        from mskpipe.geometry.topology import check_surface, mean_edge_length
        from mskpipe.io.attachment_vtk import AttachmentFileError, read_points, write_points
        from mskpipe.io.mesh_io import read_mesh, write_mesh
        from mskpipe.io.mw2 import AreaSpec, MuscleSpec

        cfg = ctx.config.export
        rec: dict[str, Any] = {"name": name, "status": "excluded", "mesh": {}, "areas": {}}
        problems: list[str] = []
        warnings: list[str] = []
        if name not in meshes:
            rec["problems"] = ["not meshed"]
            return rec, None
        found = areas.get(name, {})
        missing = [k for k in KINDS if k not in found]
        if missing:
            rec["problems"] = [f"no {KINDS[k]} area" for k in missing]
            if excluded is not None:
                rec["excluded_by_config"] = excluded
                rec["problems"].insert(0, f"export.exclude: {excluded}")
            return rec, None

        mesh = read_mesh(meshes[name])
        check = check_surface(mesh)
        rec["mesh"] = {"source": meshes[name].name, "check": _brief(check), "repaired": False}
        if not check["ok"] and cfg.repair:
            fixed, repair = repairer.repair(name, mesh)
            rec["mesh"]["repair"] = repair
            if fixed is not None:  # accepted repair, or the best failed attempt
                mesh, check = fixed, check_surface(fixed)
                rec["mesh"]["repaired"] = repair["ok"]
                rec["mesh"]["check"] = _brief(check)
            if not repair["ok"]:
                problems.append(f"mesh repair failed ({repair.get('best_summary', 'no attempt')})")
        problems += [f"mesh {p}" for p in check["problems"]]

        # checked and repaired in RAS (the label map is); the outlines are measured rotated
        mesh = mesh.transformed(homogeneous(RAS_TO_OPENSIM))
        locator = SurfaceLocator(mesh)
        edge = mean_edge_length(mesh)
        area = surface_area(mesh)
        projected, outlines = {}, {}
        for kind, a in found.items():
            try:
                pts = rotate_points(read_points(a["path"]), RAS_TO_OPENSIM)
            except (AttachmentFileError, OSError) as exc:
                problems.append(f"{KINDS[kind]}: {exc}")
                continue
            outlines[kind] = pts
            proj, gaps = locator.closest(pts)
            report = outline_report(pts, proj, gaps, edge, area)
            body = a.get("body") or locators.nearest(pts)
            report["body"] = body
            report["nearest_body"] = locators.nearest(proj)
            report["source"] = a["rel"]
            rec["areas"][kind] = report
            projected[kind] = proj
            warnings += [f"{KINDS[kind]} {p}" for p in report.pop("problems")]
            if report["nearest_body"] != body:
                ctx.logger.warning(
                    "[export_mw2] %s %s: projected outline nearer to %s than to %s "
                    "(Luca2018 moves these fibre ends with %s)",
                    name,
                    KINDS[kind],
                    report["nearest_body"],
                    body,
                    report["nearest_body"],
                )
            # Muscle Wrapping would cut an empty patch along a degenerate projection
            mark = time.perf_counter()
            stats = guard.outline_stats(proj)
            if guard.is_degenerate(
                stats, min_distinct=3, min_span_mm=2 * guard.mean_edge_length(mesh)
            ):
                try:
                    inflated = guard.inflate_outline(mesh, proj, radius_mm=cfg.inflate_radius_mm)
                except guard.GuardError as exc:  # the outline on the bone is kept
                    warnings.append(f"{name} {KINDS[kind]}: area not inflated ({exc})")
                else:
                    outlines[kind] = projected[kind] = inflated.points
                    report["inflated"] = inflated.as_dict()
                    warnings.append(f"{name} {KINDS[kind]}: area inflated")
                ctx.logger.warning("[export_mw2] %s", warnings[-1])
            rec["inflate_time_s"] = rec.get("inflate_time_s", 0.0) + time.perf_counter() - mark
        if len(projected) == 2:
            dist = contour_distance(projected["Ori"], projected["Ins"])
            rec["ori_ins_distance_mm"] = round(dist, 3)
            if dist < edge:
                warnings.append("origin and insertion overlap")
        rec["mean_edge_mm"] = round(edge, 3)
        rec["problems"] = problems
        rec["warnings"] = warnings  # outline checks: reported only, never exclude
        if excluded is not None:
            rec["excluded_by_config"] = excluded
            rec["problems"] = [f"export.exclude: {excluded}", *problems]
            return rec, None
        if problems and cfg.on_invalid == "exclude":
            return rec, None
        rec["status"] = "kept_invalid" if problems else "ok"

        stem = Path(found["Ori"]["rel"]).name.rsplit("_", 1)[0]
        folder = ctx.out_dir / MUSCLES_DIR / name
        mesh_rel = f"{MUSCLES_DIR}/{name}/{stem}.obj"
        write_mesh(ctx.out_dir / mesh_rel, mesh)
        files = {}
        for kind in found:
            target = write_points(folder / f"{stem}_{kind}.vtk", outlines[kind])
            files[kind] = target.relative_to(ctx.out_dir).as_posix()
        rec["files"] = {"mesh": mesh_rel, **files}
        spec = MuscleSpec(
            name=name,
            mesh_file=mesh_rel,
            origin=AreaSpec("origin", rec["areas"]["Ori"]["body"], files["Ori"]),
            insertion=AreaSpec("insertion", rec["areas"]["Ins"]["body"], files["Ins"]),
        )
        return rec, spec


# ---------------------------------------------------------------------------- helpers


def excluded_reason(name: str, exclude: list[str]) -> str | None:
    """Reason why muscle ``name`` (with side suffix) is excluded by ``export.exclude``."""
    base = name.rsplit("_", 1)[0]
    if base not in exclude:
        return None
    return EXCLUSION_REASONS.get(base, "listed in export.exclude")


def attachment_areas(index: dict[str, Any], folder: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """muscle -> {Ori|Ins: {path, rel, body}} from ``attachments.json``.

    The body comes from the area records of the method (``areas``, when it writes them),
    otherwise it is left to the nearest bone.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for muscle, files in index.get("muscles", {}).items():
        for rel in files:
            stem = Path(rel).name
            for kind in KINDS:
                if stem.endswith(f"_{kind}.vtk"):
                    out.setdefault(muscle, {})[kind] = {"path": folder / rel, "rel": rel}
    for area in index.get("areas", []):
        entry = out.get(area.get("muscle"), {}).get(area.get("kind"))
        if entry is not None and area.get("status") == "ok":
            entry["body"] = area.get("body")
    return out


class _BodyLocators:
    def __init__(self, geometry: dict[str, TriMesh]) -> None:
        self._geometry = geometry
        self._locators: dict[str, Any] = {}

    def nearest(self, points: np.ndarray) -> str:
        """Body nearest to the points (by the median distance)."""
        import numpy as np

        from mskpipe.geometry.compare import SurfaceLocator

        best, best_d = "", np.inf
        for body, mesh in self._geometry.items():
            if body not in self._locators:
                self._locators[body] = SurfaceLocator(mesh)
            d = float(np.median(self._locators[body].closest(points)[1]))
            if d < best_d:
                best, best_d = body, d
        return best


class _Repairer:
    """Mesh-level repair, then remeshing of the label map mask (openings, closings,
    both; :func:`~mskpipe.geometry.topology.repair_mask`)."""

    def __init__(self, ctx: StepContext) -> None:
        self.ctx = ctx
        self._volume: Any = None
        self._table: Any = None
        self._boxes: Any = None

    def repair(self, name: str, mesh: TriMesh) -> tuple[TriMesh | None, dict[str, Any]]:
        """Accepted mesh, or the best failed attempt (``info["ok"]`` False), or None."""
        from mskpipe.geometry.compare import surface_distance
        from mskpipe.geometry.metrics import signed_volume
        from mskpipe.geometry.topology import check_surface, repair_mask, repair_surface

        cfg = self.ctx.config.export
        original = abs(signed_volume(mesh))
        fixed, fixes = repair_surface(mesh)
        check = check_surface(fixed)
        ratio = abs(check["volume"]) / original if original else 1.0
        volume_ok = abs(1.0 - ratio) <= cfg.repair_max_volume_change
        attempts: list[dict[str, Any]] = [
            {
                "method": "mesh",
                "fixes": fixes,
                "genus": check.get("genus"),
                "components": check.get("components"),
                "volume_ratio": round(ratio, 4),
                "problems": check["problems"],
                "accepted": check["ok"] and volume_ok,
            }
        ]
        result, ok, best = (fixed, True, 0) if attempts[0]["accepted"] else (None, False, None)
        if not ok:
            loaded = self._mask(name)
            if loaded is not None:
                mask, affine, offset, shape = loaded
                rep = repair_mask(
                    mask,
                    affine,
                    self.ctx.config.mesh.muscles,
                    offset=offset,
                    full_shape=shape,
                    reference_volume=original,
                    max_closing_mm=cfg.repair_max_closing_mm,
                    max_opening_mm=cfg.repair_max_opening_mm,
                    max_volume_change=cfg.repair_max_volume_change,
                    check_cancel=self.ctx.check_cancel,
                )
                attempts += rep.attempts
                result, ok = rep.mesh, rep.ok
                best = None if rep.best is None else rep.best + 1
        info: dict[str, Any] = {"attempts": attempts, "ok": ok, "best": best}
        if result is not None and best is not None:
            used = attempts[best]
            label = (
                "mesh"
                if used["method"] == "mesh"
                else f"closing {used['closing_mm']} mm, opening {used['opening_mm']} mm"
            )
            dist = surface_distance(result, mesh)
            info["used"] = label
            info["distance_to_original_mm"] = {
                "mean": round(dist.mean, 3),
                "hausdorff": round(dist.hausdorff, 3),
            }
            info["volume_ratio"] = used.get("volume_ratio")
            if ok:
                self.ctx.logger.info(
                    "[export_mw2] %s repaired (%s): volume x%.3f, mean change %.2f mm, "
                    "Hausdorff %.1f mm",
                    name,
                    label,
                    used.get("volume_ratio") or 1.0,
                    dist.mean,
                    dist.hausdorff,
                )
            else:
                info["best_summary"] = f"best {label}: " + "; ".join(used["problems"])
        if not ok:
            self.ctx.logger.warning(
                "[export_mw2] %s could not be repaired (%d attempts; %s)",
                name,
                len(attempts),
                info.get("best_summary", "no remesh possible"),
            )
        return result, info

    def _mask(self, name: str):
        """Cropped, zero-padded mask of the label, its affine, offset and full shape."""
        import numpy as np
        from scipy import ndimage

        from mskpipe.io.labels import LABELMAP_FILE, LABELS_FILE, LabelTable, LabelTableError
        from mskpipe.io.nifti import NiftiError, load_labelmap
        from mskpipe.labelmap.clean import element_radius, padded_box

        cfg = self.ctx.config.export
        if self._volume is None:
            src = self.ctx.step_dir("labelmap")
            try:
                self._table = LabelTable.load(src / LABELS_FILE)
                self._volume = load_labelmap(src / LABELMAP_FILE)
            except (LabelTableError, NiftiError, OSError) as exc:
                self.ctx.logger.warning("[export_mw2] cannot remesh: %s", exc)
                return None
            self._boxes = ndimage.find_objects(self._volume.data)
        label = next((lab for lab in self._table if lab.name == name), None)
        if label is None or label.value > len(self._boxes) or self._boxes[label.value - 1] is None:
            return None
        data, affine = self._volume.data, self._volume.affine
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        radius = max(cfg.repair_max_closing_mm, cfg.repair_max_opening_mm)
        pad = element_radius(radius, spacing) + 2
        grown, padding = padded_box(data.shape, self._boxes[label.value - 1], pad)
        mask = np.pad(data[grown] == label.value, padding)
        offset = [g.start - p[0] for g, p in zip(grown, padding, strict=True)]
        return mask, affine, offset, data.shape


def _brief(check: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "ok",
        "n_vertices",
        "n_faces",
        "components",
        "genus",
        "boundary_edges",
        "non_manifold_edges",
        "misoriented_edges",
        "degenerate_faces",
        "problems",
    )
    return {k: check.get(k) for k in keys}


def _round(errors: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {b: {k: round(v, 4) for k, v in e.items()} for b, e in errors.items()}


def read_export_index(folder: Path) -> dict[str, Any]:
    """Load ``export.json`` of a finished ``export_mw2`` step."""
    path = Path(folder) / EXPORT_FILE
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StepError(f"Export index not found: {path}") from None
    if index.get("format") != FORMAT or index.get("version") != VERSION:
        raise StepError(f"Unsupported export index: {path}")
    return index
