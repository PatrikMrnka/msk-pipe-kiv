# SPDX-License-Identifier: Apache-2.0
"""Mask -> surface that the next steps can use: build, check, repair (step ``mesh``).

Requirements by kind (:func:`blocking_problems`):

* **bone** (pystaple: inertia, convex hulls) - closed 2-manifold, consistently and outward
  oriented, no degenerate or duplicate triangles. Several parts are allowed (both hip bones
  of ``pelvis_no_sacrum``); a genus above the anatomy is a warning of step ``mesh``, the
  holes themselves are repaired on the mask in step ``labelmap``.
* **muscle** (Muscle Wrapping 2.x: harmonic field between the attachment outlines) - all
  of :func:`~mskpipe.geometry.topology.check_surface`: in addition one part and genus 0.

The surface is first built with the BP filter chain (quadric decimation; meshes that pass
stay identical to BP). If it fails, the repair tries, in order (first accepted wins):

1. mesh-level fixes (:func:`~mskpipe.geometry.topology.repair_surface`),
   bones also a local patch of the defects
   (:func:`~mskpipe.geometry.topology.patch_defects`),
2. a new surface from the mask with topology-preserving decimation (cannot open holes or
   create non-manifold edges); muscles also with openings and closings
   (:func:`~mskpipe.geometry.topology.repair_mask`).

A repair is accepted only if the volume changes by at most ``max_volume_change``. If none
is, the best attempt is kept and reported (``valid = False``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mskpipe.io.mesh_io import TriMesh

# check_surface problems that make a bone unusable (prefix of the problem text)
BONE_BLOCKING = (
    "boundary edges",
    "non manifold edges",
    "misoriented edges",
    "degenerate faces",
    "duplicate faces",
    "inward orientation",
)
BRIEF_KEYS = (
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


def blocking_problems(kind: str, check: dict[str, Any]) -> list[str]:
    """Problems of :func:`check_surface` that the steps after ``mesh`` cannot accept."""
    problems = list(check.get("problems", []))
    if kind == "bone":
        return [p for p in problems if p.startswith(BONE_BLOCKING)]
    return problems


def brief(check: dict[str, Any]) -> dict[str, Any]:
    """The part of a :func:`check_surface` result kept in the indexes."""
    return {k: check.get(k) for k in BRIEF_KEYS}


@dataclass
class SurfaceBuild:
    mesh: TriMesh
    info: dict[str, Any]  # mask_to_surface info (voxels, components, border)
    decimation: str  # "quadric" (BP) or "topology"
    check: dict[str, Any]  # check_surface of ``mesh``
    problems: list[str]  # blocking problems left (empty = valid)
    repair: dict[str, Any] | None = None  # attempts, when a repair was needed

    @property
    def valid(self) -> bool:
        return not self.problems

    @property
    def repaired(self) -> bool:
        return self.repair is not None and bool(self.repair["ok"])


def build_surface(
    mask: np.ndarray,
    affine: np.ndarray,
    kind: str,
    params: Any,
    repair: Any,
    *,
    offset: Sequence[int],
    full_shape: Sequence[int],
    check_cancel: Any = None,
) -> SurfaceBuild:
    """Surface of ``mask`` (crop of the label map at ``offset``) checked for ``kind``
    (``bone`` or ``muscle``) and repaired if needed (``repair``: ``mesh.repair``)."""
    from mskpipe.geometry.surface import mask_to_surface
    from mskpipe.geometry.topology import check_surface

    first = mask_to_surface(mask, affine, params, offset=offset, full_shape=full_shape)
    check = check_surface(first.mesh)
    problems = blocking_problems(kind, check)
    build = SurfaceBuild(first.mesh, first.info, "quadric", check, problems)
    if not problems or not repair.enabled:
        return build
    return _repair(build, mask, affine, kind, params, repair, offset, full_shape, check_cancel)


def _repair(build, mask, affine, kind, params, cfg, offset, full_shape, check_cancel):
    from mskpipe.geometry.compare import surface_distance
    from mskpipe.geometry.metrics import signed_volume
    from mskpipe.geometry.surface import SurfaceError, mask_to_surface
    from mskpipe.geometry.topology import (
        check_surface,
        patch_defects,
        repair_mask,
        repair_surface,
    )

    original = build.mesh
    reference = abs(signed_volume(original))
    keep_parts = kind == "bone"
    attempts: list[dict[str, Any]] = []
    candidates: list[tuple[TriMesh, dict[str, Any], str]] = []  # mesh, attempt, decimation

    def judge(mesh: TriMesh, rec: dict[str, Any], decimation: str) -> bool:
        check = check_surface(mesh)
        ratio = abs(check["volume"]) / reference if reference > 0 else 1.0
        volume_ok = abs(1.0 - ratio) <= cfg.max_volume_change
        problems = blocking_problems(kind, check)
        if not volume_ok:
            problems.append(
                f"volume change {100 * (ratio - 1):+.1f} % "
                f"(limit {100 * cfg.max_volume_change:.0f} %)"
            )
        rec.update(
            genus=check.get("genus"),
            components=check.get("components"),
            volume_ratio=round(ratio, 4),
            problems=problems,
            accepted=not problems,
        )
        attempts.append(rec)
        candidates.append((mesh, rec, decimation))
        return rec["accepted"]

    fixed, fixes = repair_surface(original, keep_components=keep_parts)
    done = judge(fixed, {"method": "mesh", "fixes": fixes}, build.decimation)
    if not done and kind == "bone":
        # local patch of the BP surface: only the faces around the defect change
        patched, fixes = patch_defects(original)
        if fixes:
            done = judge(patched, {"method": "patch", "fixes": fixes}, build.decimation)
    if not done and kind == "bone":
        rec: dict[str, Any] = {"method": "remesh", "closing_mm": 0.0, "opening_mm": 0.0}
        try:
            topo = mask_to_surface(
                mask, affine, params, offset=offset, full_shape=full_shape, decimation="topology"
            ).mesh
        except SurfaceError as exc:
            attempts.append({**rec, "fixes": [], "problems": [f"remesh failed: {exc}"]})
        else:
            topo, fixes = repair_surface(topo, keep_components=True)
            done = judge(topo, {**rec, "fixes": fixes}, "topology")
            if not done:
                patched, fixes = patch_defects(topo)
                if fixes:
                    rec = {**rec, "method": "remesh+patch"}
                    done = judge(patched, {**rec, "fixes": fixes}, "topology")
    elif not done:
        from mskpipe.labelmap.clean import element_radius

        spacing = np.linalg.norm(np.asarray(affine, dtype=float)[:3, :3], axis=0)
        pad = element_radius(max(cfg.max_closing_mm, cfg.max_opening_mm), spacing) + 2
        rep = repair_mask(
            np.pad(mask, [(int(p), int(p)) for p in pad]),
            affine,
            params,
            offset=[int(o) - int(p) for o, p in zip(offset, pad, strict=True)],
            full_shape=full_shape,
            reference_volume=reference,
            max_closing_mm=cfg.max_closing_mm,
            max_opening_mm=cfg.max_opening_mm,
            max_volume_change=cfg.max_volume_change,
            check_cancel=check_cancel,
        )
        for rec in rep.attempts:  # same acceptance as above (muscle = all of check_surface)
            attempts.append(rec)
        if rep.mesh is not None and rep.best is not None:
            candidates.append((rep.mesh, rep.attempts[rep.best], "topology"))
        done = rep.ok

    centre = problem_center(original)
    info: dict[str, Any] = {
        "original_problems": build.problems,
        "problem_center_mm": None if centre is None else [round(float(c), 2) for c in centre],
        "attempts": attempts,
        "ok": bool(done),
        "best": None,
    }
    if not candidates:
        info["best_summary"] = "no attempt"
        return SurfaceBuild(original, build.info, build.decimation, build.check, build.problems,
                            info)  # fmt: skip
    if done:
        mesh, used, decimation = candidates[-1]
    else:  # the attempt with the fewest blocking problems (first on ties)
        mesh, used, decimation = min(candidates, key=lambda c: len(c[1]["problems"]))
    info["best"] = next(i for i, a in enumerate(attempts) if a is used)
    label = _label(used)
    dist = surface_distance(mesh, original)
    info["used"] = label
    info["distance_to_original_mm"] = {
        "mean": round(dist.mean, 3),
        "hausdorff": round(dist.hausdorff, 3),
    }
    info["volume_ratio"] = used.get("volume_ratio")
    if not done:
        info["best_summary"] = f"best {label}: " + "; ".join(used["problems"])
    check = check_surface(mesh)
    return SurfaceBuild(mesh, build.info, decimation, check, blocking_problems(kind, check), info)


def _label(attempt: dict[str, Any]) -> str:
    if attempt["method"] in ("mesh", "patch"):
        return {"mesh": "mesh", "patch": "local patch"}[attempt["method"]]
    if attempt["method"] == "remesh+patch":
        return "topology-preserving decimation + local patch"
    closing, opening = attempt.get("closing_mm", 0.0), attempt.get("opening_mm", 0.0)
    if not closing and not opening:
        return "topology-preserving decimation"
    return f"closing {closing} mm, opening {opening} mm"


def problem_center(mesh: TriMesh) -> np.ndarray | None:
    """Mean position of the vertices of open or non-manifold edges (None if there are none)."""
    f = mesh.faces
    if len(f) == 0:
        return None
    edges = np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    bad = uniq[counts != 2]
    if len(bad) == 0:
        return None
    return mesh.vertices[np.unique(bad)].mean(axis=0)


__all__ = [
    "SurfaceBuild",
    "blocking_problems",
    "brief",
    "build_surface",
    "problem_center",
]
