# SPDX-License-Identifier: Apache-2.0
"""Attachment areas supplied by the user (plugin ``manual``).

The outlines are drawn elsewhere - in NMSBuilder on the bone meshes of a run, or in the
GUI editor (Attachments…) - and saved in a folder (``areas_dir``); see
:mod:`mskpipe.io.manual_areas` for how files are assigned to muscles and areas. Each
outline is

* converted to RAS (``frame``; the meshes of a run are RAS mm, so outlines drawn on them
  need nothing),
* assigned to the nearest bone body (pelvis with sacrum, femur, tibia with fibula),
* optionally put in order around its centroid (``order: angular``, for unordered
  landmark clouds), clockwise seen from outside as the atlas outlines,
* snapped onto that bone (``max_projection_mm``) and written as
  ``<muscle>/<Stem>_<Ori|Ins>.vtk``.

Outlines far from every bone are reported; when most of them are, the folder is
probably in another coordinate frame and the step fails.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from mskpipe.config.schema import BASIC, recommended
from mskpipe.plugins.base import AttachmentsPlugin, PluginParams

if TYPE_CHECKING:
    import numpy as np

    from mskpipe.config import PipelineConfig
    from mskpipe.core.step import StepContext
    from mskpipe.io.mesh_io import TriMesh

TRANSFER_FILE = "transfer.json"
NO_FOLDER = (
    "attachments.params.areas_dir is not set: choose the folder with the attachment "
    "outlines (method manual)"
)


class ManualParams(PluginParams):
    areas_dir: str | None = Field(
        None,
        description="Folder with the outlines: <muscle>/<Stem>_Ori.vtk, _Ins.vtk "
        "(or names / areas.csv, see the help).",
        json_schema_extra={**BASIC, "empty": "not set (required)"},
    )
    frame: Literal["ras", "lps", "voxel"] = Field(
        "ras",
        description="Coordinate frame of the files: ras = the run's meshes; lps = DICOM; "
        "voxel = voxel index x voxel size of the input image, without its origin.",
    )
    order: Literal["keep", "angular"] = Field(
        "keep",
        description="keep = points in file order; angular = order them around the "
        "centroid (unordered landmark clouds).",
        json_schema_extra=BASIC,
    )
    max_projection_mm: float = Field(
        10.0,
        ge=0.0,
        le=100.0,
        description="Snap points closer than this onto the bone; 0 = off.",
        json_schema_extra=recommended(1, 20, "above 20 mm points can snap to another bone"),
    )
    max_distance_mm: float = Field(
        20.0,
        gt=0.0,
        le=500.0,
        description="Outlines farther from every bone are reported; the step fails when "
        "the median outline is (another coordinate frame).",
        json_schema_extra=recommended(5, 50, "bones are a few mm off between segmentations"),
    )


class FrameMismatchError(ValueError):
    """The outlines lie far from the bones: another coordinate frame."""

    @classmethod
    def of(cls, outlines, bodies: Mapping[str, Any], median_mm: float, where: str = ""):
        import numpy as np

        def box(points) -> str:
            lo, hi = np.min(points, axis=0), np.max(points, axis=0)
            return ", ".join(f"{a} {lo[i]:.0f} to {hi[i]:.0f}" for i, a in enumerate("xyz"))

        pts = np.vstack([np.asarray(p).reshape(-1, 3) for p in outlines])
        bones = np.vstack([m.vertices for m in bodies.values()])
        return cls(
            f"The outlines{where} lie {median_mm:.0f} mm from the bones (median): they are "
            "in another coordinate frame than the meshes of the run (RAS, mm). Outlines: "
            f"{box(pts)}; bones: {box(bones)} mm. Draw or export them on the meshes of the "
            "run (03_mesh), or set attachments.params.frame (lps: DICOM coordinates; voxel: "
            "voxel index x voxel size of the input image, e.g. meshes of a raw volume in "
            "NMSBuilder)."
        )


def areas_dir(params: Mapping[str, Any] | ManualParams) -> Path | None:
    value = params.areas_dir if isinstance(params, ManualParams) else params.get("areas_dir")
    return Path(value).expanduser() if value else None


# --------------------------------------------------------------------------- bodies


def body_sources(bones: Mapping[str, Path], side: str, merge: bool) -> dict[str, list[str]]:
    """OpenSim body -> bone meshes it consists of.

    ``merge=False``: ``bones`` are the meshes given to the attachment method by the step
    (``pelvis_with_sacrum`` and the tibia with the fibula already merged); ``merge=True``:
    the meshes of step ``mesh``, merged here the same way.
    """
    out: dict[str, list[str]] = {}
    if "pelvis_with_sacrum" in bones:
        out["pelvis"] = ["pelvis_with_sacrum"]
    elif "pelvis_no_sacrum" in bones:
        both = merge and "sacrum" in bones
        out["pelvis"] = ["pelvis_no_sacrum", "sacrum"] if both else ["pelvis_no_sacrum"]
    if f"femur_{side}" in bones:
        out[f"femur_{side}"] = [f"femur_{side}"]
    if f"tibia_{side}" in bones:
        with_fibula = merge and f"fibula_{side}" in bones
        out[f"tibia_{side}"] = (
            [f"tibia_{side}", f"fibula_{side}"] if with_fibula else [f"tibia_{side}"]
        )
    return out


def load_bodies(bones: Mapping[str, Path], side: str, merge: bool) -> dict[str, TriMesh]:
    """OpenSim body -> its (merged) mesh."""
    import numpy as np

    from mskpipe.io.mesh_io import TriMesh, read_mesh

    out = {}
    for body, parts in body_sources(bones, side, merge).items():
        meshes = [read_mesh(bones[p]) for p in parts]
        if len(meshes) == 1:
            out[body] = meshes[0]
            continue
        offsets = np.cumsum([0] + [m.n_vertices for m in meshes[:-1]])
        out[body] = TriMesh(
            np.vstack([m.vertices for m in meshes]),
            np.vstack([m.faces + o for m, o in zip(meshes, offsets, strict=True)]),
        )
    return out


class BodySurface:
    """Closest points and outer normals on one body mesh."""

    def __init__(self, mesh: TriMesh) -> None:
        import numpy as np
        import vtk

        from mskpipe.geometry.surface import to_polydata

        self.mesh = mesh
        self._locator = vtk.vtkStaticCellLocator()
        self._locator.SetDataSet(to_polydata(mesh))
        self._locator.BuildLocator()
        v, f = mesh.vertices, mesh.faces
        normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        # signed volume < 0: faces wound inwards
        volume = float(np.einsum("ij,ij->i", v[f[:, 0]], normals).sum()) / 6.0
        self._normals = normals * (-1.0 if volume < 0 else 1.0)

    def closest(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(closest points, distances, cell ids)."""
        import numpy as np
        import vtk

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.empty_like(pts)
        dist = np.empty(len(pts))
        cells = np.empty(len(pts), dtype=np.int64)
        closest = [0.0, 0.0, 0.0]
        cell, sub, d2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        for i, p in enumerate(pts):
            self._locator.FindClosestPoint(p, closest, cell, sub, d2)
            out[i], dist[i], cells[i] = closest, d2.get() ** 0.5, cell.get()
        return out, dist, cells

    def outward(self, cells: np.ndarray) -> np.ndarray:
        """Mean outer normal of the given cells (unit vector)."""
        import numpy as np

        n = self._normals[np.asarray(cells, dtype=np.int64)].sum(axis=0)
        norm = float(np.linalg.norm(n))
        return n / norm if norm > 0 else n


def place_outline(
    points: np.ndarray,
    surfaces: Mapping[str, BodySurface],
    *,
    order: str = "keep",
    max_projection_mm: float = 10.0,
    body: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Outline on the nearest (or given) body: ordered, snapped; and its record."""
    import numpy as np

    from mskpipe.geometry.outline import angular_order

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    found = {name: s.closest(pts) for name, s in surfaces.items()}
    if body is None or body not in found:
        body = min(found, key=lambda name: float(np.median(found[name][1])))
    closest, dist, cells = found[body]
    rec: dict[str, Any] = {"body": body, "n_points": len(pts)}
    if order == "angular":
        idx = angular_order(pts, surfaces[body].outward(cells))
        rec["reordered"] = bool((idx != np.arange(len(idx))).any())
        pts, closest, dist = pts[idx], closest[idx], dist[idx]
    snap = dist <= max_projection_mm
    final = np.where(snap[:, None], closest, pts)
    rec.update(
        snapped=round(float(snap.mean()), 3),
        to_bone_mean_mm=round(float(dist.mean()), 3),
        to_bone_median_mm=round(float(np.median(dist)), 3),
        to_bone_max_mm=round(float(dist.max()), 3),
    )
    return final, rec


# --------------------------------------------------------------------------- plugin


class Manual(AttachmentsPlugin):
    name = "manual"
    description = (
        "Attachment outlines supplied by the user (e.g. drawn in NMSBuilder or edited in "
        "the GUI), assigned to the nearest bone and snapped onto it."
    )
    Params = ManualParams
    requires_modules = ("vtk",)

    @classmethod
    def cache_identity(cls, config: PipelineConfig) -> Mapping[str, Any]:
        from mskpipe.io.manual_areas import folder_fingerprint

        return {"areas_sha256": folder_fingerprint(areas_dir(config.attachments.params))}

    @classmethod
    def _matching(cls, config: PipelineConfig):
        from mskpipe.io.manual_areas import match_folder

        folder = areas_dir(config.attachments.params)
        return None if folder is None else match_folder(folder, config.skeleton.side)

    @classmethod
    def preflight(cls, config: PipelineConfig, modality: str) -> list[str]:
        from mskpipe.io.manual_areas import ManualAreasError, read_outline

        match = cls._matching(config)
        if match is None:
            return [NO_FOLDER]
        problems = match.problems()
        if problems:
            return problems
        side = config.skeleton.side
        areas = match.of_side(side)
        if not areas:
            return [
                f"No attachment outlines of side '{side}' recognised in {match.folder} "
                "(names: <muscle>/<Stem>_Ori.vtk, or list them in areas.csv)"
            ]
        for area in areas.values():
            try:
                read_outline(area.path)
            except (ManualAreasError, OSError) as exc:
                problems.append(f"{match.rel(area.path)}: {exc}")
        return problems

    @classmethod
    def preflight_warnings(cls, config: PipelineConfig, modality: str) -> list[str]:
        match = cls._matching(config)
        if match is None or not match.folder.is_dir():
            return []
        notes = []
        if match.unmatched:
            names = ", ".join(match.rel(p) for p in match.unmatched[:5])
            more = f" (+{len(match.unmatched) - 5} more)" if len(match.unmatched) > 5 else ""
            notes.append(
                f"{len(match.unmatched)} file(s) in {match.folder} not recognised as an "
                f"attachment area, ignored: {names}{more} (name them <muscle>_Ori/_Ins or "
                "list them in areas.csv)"
            )
        side = config.skeleton.side
        areas = match.of_side(side)
        origins = {m for m, k in areas if k == "Ori"}
        insertions = {m for m, k in areas if k == "Ins"}
        incomplete = sorted(origins ^ insertions)
        if incomplete:
            notes.append(
                "Only one of origin / insertion given, these muscles will not be exported: "
                + ", ".join(incomplete)
            )
        return notes

    def compute(
        self,
        ctx: StepContext,
        bones: Mapping[str, Path],
        muscles: Mapping[str, Path],
        out_dir: Path,
        params: PluginParams,
    ) -> Mapping[str, Path]:
        """``bones`` as for ``atlas_registration``; ``muscles``: meshed muscles of the
        side. Writes ``<muscle>/<Stem>_<Ori|Ins>.vtk`` and ``transfer.json``."""
        import numpy as np

        from mskpipe.io.attachment_vtk import write_points
        from mskpipe.io.manual_areas import (
            KINDS,
            folder_fingerprint,
            match_folder,
            read_outline,
            to_ras,
        )

        assert isinstance(params, ManualParams)
        side = ctx.config.skeleton.side
        folder = areas_dir(params)
        if folder is None:
            raise ValueError(NO_FOLDER)
        match = match_folder(folder, side)
        if match.problems():
            raise ValueError("; ".join(match.problems()))
        areas = match.of_side(side)
        if not areas:
            raise ValueError(f"No attachment outlines of side '{side}' in {folder}")

        affine = None
        if params.frame == "voxel":
            from mskpipe.io.manual_areas import image_affine

            affine = image_affine(ctx.ws.input_image)
        bodies = load_bodies(bones, side, merge=False)
        if not bodies:
            raise ValueError("No pelvis, femur or tibia mesh to put the outlines on")
        surfaces = {name: BodySurface(mesh) for name, mesh in bodies.items()}
        provided = {m for m, _ in areas}
        report: dict[str, Any] = {
            "source": {"dir": str(folder), "sha256": folder_fingerprint(folder)},
            "side": side,
            "frame": params.frame,
            "order": params.order,
            "bodies": {b: body_sources(bones, side, merge=False)[b] for b in bodies},
            "not_provided": sorted(set(muscles) - provided),
            "not_meshed": sorted(provided - set(muscles)),
            "unmatched": [match.rel(p) for p in match.unmatched],
            "areas": [],
        }
        outputs: dict[str, Path] = {}
        far: list[str] = []
        medians: list[float] = []
        given: list[np.ndarray] = []
        for muscle in sorted(muscles):
            for kind in KINDS:
                rec: dict[str, Any] = {"muscle": muscle, "kind": kind}
                area = areas.get((muscle, kind))
                if area is None:
                    report["areas"].append({**rec, "status": "not_provided"})
                    continue
                ctx.check_cancel()
                pts = to_ras(read_outline(area.path), params.frame, affine)
                final, placed = place_outline(
                    pts,
                    surfaces,
                    order=params.order,
                    max_projection_mm=params.max_projection_mm,
                )
                path = write_points(out_dir / muscle / f"{area.stem}_{kind}.vtk", final)
                outputs[muscle] = path.parent
                bone = body_sources(bones, side, merge=False)[placed["body"]][0]
                rec.update(
                    status="ok",
                    bone=bone,
                    file=path.relative_to(out_dir).as_posix(),
                    source_file=match.rel(area.path),
                    match=area.rule,
                    **placed,
                )
                medians.append(placed["to_bone_median_mm"])
                given.append(pts)
                if placed["to_bone_median_mm"] > params.max_distance_mm:
                    far.append(f"{muscle} {kind} ({placed['to_bone_median_mm']:.0f} mm)")
                    rec["warning"] = "far from the bone"
                report["areas"].append(rec)
        if medians and float(np.median(medians)) > params.max_distance_mm:
            raise FrameMismatchError.of(
                given, bodies, float(np.median(medians)), where=f" in {folder}"
            )
        for item in far:
            ctx.logger.warning("[attachments] %s: outline far from the bone", item)
        for muscle in report["not_provided"]:
            ctx.logger.warning("[attachments] %s: no outline given (not exported)", muscle)
        (out_dir / TRANSFER_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return outputs


__all__ = [
    "BodySurface",
    "FrameMismatchError",
    "Manual",
    "ManualParams",
    "body_sources",
    "load_bodies",
    "place_outline",
]
