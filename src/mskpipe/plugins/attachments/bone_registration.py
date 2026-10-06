# SPDX-License-Identifier: Apache-2.0
"""Attachment areas from an atlas by registering its bones (plugin ``bone_registration``).

The atlas is a folder (not distributed with mskpipe, e.g. built from the LHDL dataset)::

    atlas.json                      muscles -> {stem, Ori/Ins: {file, bone, body}}
    bones/pelvis_with_sacrum.obj    and/or pelvis_no_sacrum.obj (the one matching the
                                    subject's pelvis is preferred), femur_<s>.obj,
                                    tibia_<s>.obj (tibia + fibula)
    muscles/<muscle>/<Stem>_<Ori|Ins>.vtk   closed outlines in mm (Muscle Wrapping 2.x)

Per bone: symmetric trimmed similarity ICP (identity or centroid start, the better one;
point-to-point, then point-to-plane), then one non-rigid stage (``nonrigid``):

* ``coherent_icp`` - non-rigid ICP with a Gaussian motion-coherence field, hard
  point-to-plane correspondences (default; :func:`~mskpipe.geometry.register.fit_coherent_icp`),
* ``cpd`` - Coherent Point Drift (Myronenko & Song 2010), soft GMM correspondences,
* ``fast_rnrr`` - Fast-RNRR (Yao et al. 2020), external executable supplied by the user,
* ``none`` - similarity only.

The outlines are mapped by the same transforms, snapped onto the subject's bone surface
and written with the point order kept.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from mskpipe.plugins.base import AttachmentsPlugin, PluginParams

if TYPE_CHECKING:
    from mskpipe.config import PipelineConfig
    from mskpipe.core.step import StepContext

ATLAS_ENV = "MSKPIPE_ATLAS_DIR"
ATLAS_INDEX = "atlas.json"
TRANSFER_FILE = "transfer.json"


class BoneRegistrationParams(PluginParams):
    atlas_dir: str | None = Field(
        None, description=f"Atlas folder (atlas.json, bones/, muscles/); default ${ATLAS_ENV}."
    )
    n_points: int = Field(
        3000, ge=200, le=20000, description="Atlas surface samples per bone (ICP source)."
    )
    target_points: int = Field(
        50000,
        ge=1000,
        le=500000,
        description="Subject surface samples per bone (ICP target); denser = less bias.",
    )
    scale: bool = Field(True, description="Allow a uniform scale in the rigid stage.")
    icp_metric: Literal["point_to_plane", "point_to_point"] = Field(
        "point_to_plane",
        description="Final ICP metric; point_to_plane refines a point_to_point start.",
    )
    icp_iterations: int = Field(100, ge=1, le=1000, description="Maximum ICP iterations.")
    trim: float = Field(
        0.9, gt=0.5, le=1.0, description="Fraction of closest pairs kept (bones of other extent)."
    )
    nonrigid: Literal["coherent_icp", "cpd", "fast_rnrr", "none"] = Field(
        "coherent_icp", description="Non-rigid stage after the similarity ICP."
    )
    nonrigid_points: int = Field(
        1500, ge=100, le=5000, description="Atlas points driving coherent_icp / cpd per bone."
    )
    coherent_beta_mm: float = Field(
        40.0, gt=1.0, le=200.0, description="coherent_icp: width of the Gaussian kernel [mm]."
    )
    coherent_lambda: float = Field(
        2.0, gt=0.0, le=100.0, description="coherent_icp: regularisation (larger = stiffer)."
    )
    coherent_iterations: int = Field(30, ge=1, le=500, description="coherent_icp: iterations.")
    cpd_beta: float = Field(
        2.0, gt=0.0, le=20.0, description="cpd: kernel width in normalised units (paper: 2)."
    )
    cpd_lambda: float = Field(2.0, gt=0.0, le=100.0, description="cpd: regularisation (paper: 2).")
    cpd_w: float = Field(
        0.1, ge=0.0, lt=1.0, description="cpd: outlier weight (0 = none, probreg default)."
    )
    cpd_iterations: int = Field(150, ge=1, le=2000, description="cpd: maximum EM iterations.")
    cpd_target_points: int = Field(
        5000, ge=500, le=20000, description="cpd: subject points (memory ~ points x nonrigid)."
    )
    cpd_sigma_init_mm: float | None = Field(
        None,
        gt=0.0,
        description="cpd: initial GMM width [mm]; None = standard CPD start (all pairs).",
    )
    fast_rnrr_exe: str | None = Field(
        None,
        description="fast_rnrr: executable (not distributed); default $MSKPIPE_FAST_RNRR or PATH.",
    )
    fast_rnrr_max_faces: int = Field(
        20000, ge=1000, le=500000, description="fast_rnrr: atlas bone decimated to this size."
    )
    max_projection_mm: float = Field(
        10.0, ge=0.0, le=100.0, description="Snap points closer than this onto the bone; 0 = off."
    )
    seed: int = Field(0, description="Seed of the surface sampling.")


def atlas_dir(params: Mapping[str, Any] | BoneRegistrationParams) -> Path | None:
    value = (
        params.atlas_dir if isinstance(params, BoneRegistrationParams) else params.get("atlas_dir")
    )
    value = value or os.environ.get(ATLAS_ENV)
    return Path(value) if value else None


def atlas_fingerprint(folder: Path | None) -> str:
    """sha256 over the atlas files (index, bones, outlines); 'missing' if not there."""
    if folder is None or not (folder / ATLAS_INDEX).is_file():
        return "missing"
    h = hashlib.sha256()
    files = [folder / ATLAS_INDEX, *sorted((folder / "bones").glob("*.obj"))]
    files += sorted((folder / "muscles").rglob("*.vtk"))
    for f in files:
        h.update(f.relative_to(folder).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def atlas_bone_group(bone: str) -> str:
    """Atlas bone name -> group: pelvis_with_sacrum -> pelvis, femur_r -> femur, ..."""
    return "pelvis" if bone.startswith("pelvis") else bone.rsplit("_", 1)[0]


class BoneRegistration(AttachmentsPlugin):
    name = "bone_registration"
    description = (
        "Attachment outlines transferred from an atlas by bone registration "
        "(similarity ICP + coherent_icp | cpd | fast_rnrr)."
    )
    Params = BoneRegistrationParams
    requires_modules = ("vtk", "scipy")

    @classmethod
    def cache_identity(cls, config: PipelineConfig) -> Mapping[str, Any]:
        params = config.attachments.params
        out: dict[str, Any] = {"atlas_sha256": atlas_fingerprint(atlas_dir(params))}
        if params.get("nonrigid") == "fast_rnrr":
            out["fast_rnrr_sha256"] = _executable_fingerprint(params.get("fast_rnrr_exe"))
        return out

    def compute(
        self,
        ctx: StepContext,
        bones: Mapping[str, Path],
        muscles: Mapping[str, Path],
        out_dir: Path,
        params: PluginParams,
    ) -> Mapping[str, Path]:
        """``bones``: subject meshes ``pelvis_with_sacrum`` (else ``pelvis_no_sacrum``),
        ``femur_<s>``, ``tibia_<s>`` (tibia + fibula); ``muscles``: meshed muscles to
        transfer (``<name>_<s>``).
        Writes ``<muscle>/<Stem>_<Ori|Ins>.vtk``, ``registration/<bone>.stl`` and
        ``transfer.json``; returns the output folder per muscle."""
        import time

        import numpy as np

        from mskpipe.geometry.compare import closest_on_surface
        from mskpipe.geometry.register import apply, sample_surface, similarity_icp
        from mskpipe.io.attachment_vtk import read_points, write_points
        from mskpipe.io.mesh_io import TriMesh, read_mesh, write_mesh

        assert isinstance(params, BoneRegistrationParams)
        side = ctx.config.skeleton.side
        folder = atlas_dir(params)
        if folder is None:
            raise ValueError(
                "No atlas: set attachments.params.atlas_dir "
                f"or the {ATLAS_ENV} environment variable"
            )
        index_path = folder / ATLAS_INDEX
        if not index_path.is_file():
            raise FileNotFoundError(f"Atlas index not found: {index_path}")
        atlas = json.loads(index_path.read_text(encoding="utf-8"))
        if atlas.get("units", "mm") != "mm" or atlas.get("frame", "ras") != "ras":
            raise ValueError(
                f"Atlas must be in RAS mm, got {atlas.get('frame')} {atlas.get('units')}"
            )

        with_sacrum = "pelvis_with_sacrum" in bones
        subject = {
            "pelvis": bones.get("pelvis_with_sacrum") or bones.get("pelvis_no_sacrum"),
            "femur": bones.get(f"femur_{side}"),
            "tibia": bones.get(f"tibia_{side}"),
        }
        pelvis_order = ("pelvis_with_sacrum", "pelvis_no_sacrum")
        if not with_sacrum:
            pelvis_order = pelvis_order[::-1]
        atlas_bones = {
            "pelvis": next(
                (
                    folder / f"bones/{n}.obj"
                    for n in pelvis_order
                    if (folder / f"bones/{n}.obj").is_file()
                ),
                None,
            ),
            "femur": folder / f"bones/femur_{side}.obj",
            "tibia": folder / f"bones/tibia_{side}.obj",
        }
        if atlas_bones["pelvis"] is not None and atlas_bones["pelvis"].stem != pelvis_order[0]:
            ctx.logger.warning(
                "[attachments] atlas has no %s.obj, registering %s to the subject's %s",
                pelvis_order[0],
                atlas_bones["pelvis"].name,
                pelvis_order[0],
            )

        # areas to transfer: atlas muscles of this side that were meshed
        entries = {k: v for k, v in atlas["muscles"].items() if k.endswith(f"_{side}")}
        report: dict[str, Any] = {
            "atlas": {"dir": str(folder), "sha256": atlas_fingerprint(folder)},
            "side": side,
            "not_in_atlas": sorted(set(muscles) - set(entries)),
            "not_meshed": sorted(set(entries) - set(muscles)),
            "bones": {},
            "areas": [],
        }
        areas = []
        for muscle in sorted(set(entries) & set(muscles)):
            for kind in ("Ori", "Ins"):
                area = entries[muscle].get(kind) or {}
                if "file" not in area:
                    report["areas"].append(
                        {"muscle": muscle, "kind": kind, "status": "not_in_atlas"}
                    )
                    continue
                areas.append((muscle, kind, entries[muscle]["stem"], area))

        # register the bones that carry areas
        rng = np.random.default_rng(params.seed)
        transforms: dict[str, Any] = {}
        meshes: dict[str, TriMesh] = {}
        for group in sorted({atlas_bone_group(a[3]["bone"]) for a in areas}):
            ctx.check_cancel()
            src_path, dst_path = atlas_bones.get(group), subject.get(group)
            if src_path is None or not Path(src_path).is_file() or dst_path is None:
                report["bones"][group] = {
                    "status": "missing",
                    "atlas": str(src_path),
                    "subject": str(dst_path),
                }
                continue
            t0 = time.perf_counter()
            src_mesh, dst_mesh = read_mesh(src_path), read_mesh(dst_path)
            src, src_n = sample_surface(src_mesh, params.n_points, rng, normals=True)
            dst, dst_n = sample_surface(dst_mesh, params.target_points, rng, normals=True)
            starts = {"identity": np.eye(4), "centroid": np.eye(4)}
            starts["centroid"][:3, 3] = dst.mean(0) - src.mean(0)
            plane = params.icp_metric == "point_to_plane"
            # point-to-plane only needs a rough start: coarse stage on a subsample
            coarse = dst[: 4 * params.n_points] if plane else dst
            fits = {}
            for key, init in starts.items():
                fit = similarity_icp(
                    src,
                    coarse,
                    init=init,
                    scale=params.scale,
                    iterations=params.icp_iterations,
                    trim=params.trim,
                    tol=1e-4 if plane else 1e-6,
                )
                if plane:
                    fit = similarity_icp(
                        src,
                        dst,
                        init=fit.matrix,
                        scale=params.scale,
                        iterations=params.icp_iterations,
                        trim=params.trim,
                        src_normals=src_n,
                        dst_normals=dst_n,
                    )
                fits[key] = fit
            start = min(fits, key=lambda k: fits[k].rms)
            rigid = fits[start]
            moved = apply(rigid.matrix, src)
            info: dict[str, Any] = {
                "status": "ok",
                "atlas": Path(src_path).name,
                "subject": Path(dst_path).name,
                "start": start,
                "scale": round(rigid.scale, 5),
                "icp_iterations": rigid.iterations,
                "rotation_deg": round(_rotation_deg(rigid.matrix), 3),
                "translation_mm": np.round(rigid.matrix[:3, 3], 3).tolist(),
                "rigid": _residual(moved, dst, params.trim),
            }
            warp, extra = _nonrigid(
                ctx,
                params,
                rng,
                moved,
                dst,
                dst_n,
                TriMesh(apply(rigid.matrix, src_mesh.vertices), src_mesh.faces),
                dst_mesh,
                out_dir / "registration" / "fast_rnrr" / group,
            )
            info["nonrigid_method"] = params.nonrigid
            if warp is not None:
                info["nonrigid"] = _residual(warp(moved), dst, params.trim)
                info["warp_max_mm"] = round(
                    float(np.linalg.norm(warp(moved) - moved, axis=1).max()), 3
                )
                info["nonrigid_info"] = extra
            info["time_s"] = round(time.perf_counter() - t0, 3)
            report["bones"][group] = info
            transforms[group] = (rigid.matrix, warp)
            meshes[group] = dst_mesh
            reg = apply(rigid.matrix, src_mesh.vertices)
            write_mesh(
                out_dir / "registration" / f"{Path(dst_path).stem}.stl",
                TriMesh(reg if warp is None else warp(reg), src_mesh.faces),
            )
            ctx.logger.info(
                "[attachments] %s: scale %.3f, rms %.2f -> %.2f mm (%s start)",
                group,
                rigid.scale,
                info["rigid"]["mean_mm"],
                info.get("nonrigid", info["rigid"])["mean_mm"],
                start,
            )

        # transfer the outlines
        outputs: dict[str, Path] = {}
        for muscle, kind, stem, area in areas:
            group = atlas_bone_group(area["bone"])
            rec: dict[str, Any] = {
                "muscle": muscle,
                "kind": kind,
                "bone": group,
                "body": area.get("body"),
            }
            if group not in transforms:
                report["areas"].append({**rec, "status": "bone_missing"})
                continue
            matrix, warp = transforms[group]
            pts = read_points(folder / area["file"])
            moved = apply(matrix, pts)
            if warp is not None:
                moved = warp(moved)
            closest, dist = closest_on_surface(moved, meshes[group])
            snap = dist <= params.max_projection_mm
            final = np.where(snap[:, None], closest, moved)
            path = write_points(out_dir / muscle / f"{stem}_{kind}.vtk", final)
            outputs[muscle] = path.parent
            report["areas"].append(
                {
                    **rec,
                    "status": "ok",
                    "file": path.relative_to(out_dir).as_posix(),
                    "n_points": len(final),
                    "snapped": round(float(snap.mean()), 3),
                    "to_bone_mean_mm": round(float(dist.mean()), 3),
                    "to_bone_max_mm": round(float(dist.max()), 3),
                    "shift_from_atlas_mean_mm": round(
                        float(np.linalg.norm(final - pts, axis=1).mean()), 3
                    ),
                }
            )
        (out_dir / TRANSFER_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return outputs


def _nonrigid(ctx, params, rng, moved, dst, dst_n, src_moved_mesh, dst_mesh, work):
    """Non-rigid stage on top of the similarity; returns (field or None, info)."""
    import time

    import numpy as np

    from mskpipe.geometry import register

    t0 = time.perf_counter()
    method = params.nonrigid
    if method == "none":
        return None, {}
    if method == "fast_rnrr":
        from mskpipe.geometry.fast_rnrr import (
            MeshWarp,
            decimate,
            find_executable,
            run_fast_rnrr,
        )

        exe = find_executable(params.fast_rnrr_exe)
        source = decimate(src_moved_mesh, params.fast_rnrr_max_faces)
        deformed = run_fast_rnrr(exe, source, dst_mesh, work, ctx.run_command)
        info = {"executable": str(exe), "vertices": len(source.vertices)}
        return MeshWarp(source, deformed.vertices), {
            **info,
            "time_s": round(time.perf_counter() - t0, 3),
        }
    sub = moved[rng.choice(len(moved), min(params.nonrigid_points, len(moved)), replace=False)]
    if method == "coherent_icp":
        warp, rms = register.fit_coherent_icp(
            sub,
            dst,
            dst_normals=dst_n,
            beta=params.coherent_beta_mm,
            lam=params.coherent_lambda,
            iterations=params.coherent_iterations,
            trim=params.trim,
        )
        info = {"rms_mm": round(rms, 3)}
    else:  # cpd
        target = dst[: params.cpd_target_points]  # samples are i.i.d.: a uniform subset
        res = register.fit_cpd(
            sub,
            target,
            beta=params.cpd_beta,
            lam=params.cpd_lambda,
            w=params.cpd_w,
            iterations=params.cpd_iterations,
            sigma_init=params.cpd_sigma_init_mm,
        )
        warp = res.warp
        info = {
            "iterations": res.iterations,
            "sigma_mm": round(float(np.sqrt(res.sigma2)), 3),
            "beta_mm": round(res.beta, 3),
        }
    info["time_s"] = round(time.perf_counter() - t0, 3)
    return warp, info


def _executable_fingerprint(value: str | None) -> str:
    from mskpipe.geometry.fast_rnrr import FastRnrrError, find_executable

    try:
        exe = find_executable(value)
    except FastRnrrError:
        return "missing"
    return hashlib.sha256(exe.read_bytes()).hexdigest()


def _rotation_deg(m) -> float:
    import numpy as np

    r = m[:3, :3] / np.cbrt(np.linalg.det(m[:3, :3]))
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def _residual(moved, dst, trim: float) -> dict[str, float]:
    """Distances atlas samples -> subject samples (all and trimmed)."""
    import numpy as np
    from scipy.spatial import cKDTree

    d = cKDTree(dst).query(moved)[0]
    kept = d[d <= np.quantile(d, trim)]
    return {
        "mean_mm": round(float(d.mean()), 3),
        "p95_mm": round(float(np.percentile(d, 95)), 3),
        "trimmed_mean_mm": round(float(kept.mean()), 3),
    }
