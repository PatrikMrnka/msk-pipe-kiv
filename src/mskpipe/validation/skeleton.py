# SPDX-License-Identifier: Apache-2.0
"""Skeleton backend vs. MATLAB STAPLE.

1. parity: same bodies, joints, coordinates and markers as the MATLAB model, numbers
   within tolerance (``TOLERANCE``) - only for models built from the *same* bone meshes;
2. rigid frame: bones moved by a rigid transform (the BP "voxel index x spacing" frame vs.
   NIfTI world: LPS <-> RAS and a shift) give the same model moved by that transform;
3. anatomical comparison (:func:`compare_anatomical`): models from *different* bone meshes
   (other segmentation, other frame), compared in the pelvis anatomical frame in mm/deg.

Needs pystaple (environments ``cpu``, ``gpu``, ``dev-cpu``).
"""

from __future__ import annotations

import time
import urllib.request
import warnings
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from mskpipe.config.schema import SkeletonConfig
from mskpipe.core.manifest import package_info
from mskpipe.io.osim import read_osim
from mskpipe.plugins.skeleton.pystaple_backend import build_hip_model, required_bones
from mskpipe.steps.skeleton import skeleton_qc

PYSTAPLE_RAW = "https://raw.githubusercontent.com/PatrikMrnka/pystaple"
DATASET = "MSKPIPE_CT"  # LHDL CT bones meshed by the BP pipeline + MATLAB STAPLE model
TOLERANCE = {"m": 1e-9, "rad": 1e-8, "kg": 1e-9, "other": 1e-9}
RIGID_R = np.diag([-1.0, -1.0, 1.0])  # LPS <-> RAS
RIGID_T = np.array([0.0, 0.0, -1105.0])  # mm, z shift of the LHDL CT
RIGID_TOL_M = 1e-8
RIGID_TOL_QC_MM = 1e-3


class ParityError(RuntimeError):
    """Reference data cannot be found or downloaded."""


def fetch_reference(cache: Path) -> tuple[Path, Path]:
    """Download pystaple's ``MSKPIPE_CT`` dataset for the installed pystaple commit.

    Returns (bones folder, MATLAB model). Files are cached per commit.
    """
    commit = package_info("pystaple").commit
    if not commit:
        raise ParityError("pystaple is not installed from git; pass the bones and model")
    folder = Path(cache) / commit[:12] / DATASET
    files = (*(f"mesh_{b}.mat" for b in required_bones("r")), "osim/bone_model.osim")
    for rel in files:
        target = folder / rel
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{PYSTAPLE_RAW}/{commit}/reference/{DATASET}/{rel}"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
        except OSError as exc:
            raise ParityError(f"cannot download {url}: {exc}") from exc
        target.write_bytes(data)
    return folder, folder / "osim" / "bone_model.osim"


def find_bones(folder: Path, side: str = "r") -> dict[str, Path]:
    """``<bone>.stl``, ``<bone>.mat`` or ``mesh_<bone>.mat`` of the hip-model bones."""
    bones = {}
    for name in required_bones(side):
        for cand in (f"{name}.stl", f"{name}.mat", f"mesh_{name}.mat"):
            if (Path(folder) / cand).is_file():
                bones[name] = Path(folder) / cand
                break
    missing = sorted(set(required_bones(side)) - set(bones))
    if missing:
        raise ParityError(f"missing bones in {folder}: {', '.join(missing)}")
    return bones


def _unit(quantity: str) -> str:
    return next((u for u in ("m", "rad", "kg") if f"[{u}" in quantity), "other")


def compare_parity(ours: Path, ref: Path) -> dict[str, Any]:
    """Property-by-property comparison of two models (pystaple.osim.compare)."""
    from pystaple.osim.compare import compare_models

    structural, ignored, worst = [], [], {}
    for d in compare_models(ours, ref):
        if d.max_abs is not None:
            unit = _unit(d.quantity)
            worst[unit] = max(worst.get(unit, 0.0), d.max_abs)
            continue
        if d.equal:
            continue
        if d.quantity == "mesh files":  # the BP pipeline appended a sacrum display mesh
            a = [PureWindowsPath(m).as_posix() for m in read_osim(ours).bodies[d.item].mesh_files]
            b = [PureWindowsPath(m).as_posix() for m in read_osim(ref).bodies[d.item].mesh_files]
            if b[: len(a)] == a:
                ignored.append(f"{d.item}: extra display meshes in the reference {b[len(a) :]}")
                continue
        structural.append(f"{d.section} {d.item} {d.quantity}: {d.note}")
    over = [
        f"max |diff| {v:.3g} {u} > {TOLERANCE[u]:g}" for u, v in worst.items() if v > TOLERANCE[u]
    ]
    return {
        "max_abs_diff": worst,
        "ignored": ignored,
        "failures": structural + over,
        "passed": not structural and not over,
    }


def compare_rigid(bones: dict[str, Path], ours: Path, config: SkeletonConfig, tmp: Path):
    """Build the model from rigidly moved bones and compare with ``ours`` moved likewise."""
    from pystaple.io import load_mesh
    from pystaple.mesh import TriMesh
    from pystaple.workflow import build_hip_model as build

    moved = {}
    for name, path in bones.items():
        mesh = load_mesh(path)
        moved[name] = TriMesh(mesh.points @ RIGID_R.T + RIGID_T, mesh.faces)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other, *_ = build(moved, tmp, body_mass=config.body_mass, coeff_face_reduc=1.0)

    a, b = read_osim(ours), read_osim(other)
    t_m = RIGID_T / 1000.0
    d_pos = d_rot = 0.0
    for name, joint in a.joints.items():
        for fname, fa in joint.frames.items():
            if fa.parent_name == "ground":
                continue
            fb = b.joints[name].frames[fname]
            d_pos = max(d_pos, float(np.abs(RIGID_R @ fa.translation + t_m - fb.translation).max()))
            d_rot = max(d_rot, float(np.abs(RIGID_R @ fa.rotation - fb.rotation).max()))
    for name, ma in a.markers.items():
        loc = RIGID_R @ ma.location + t_m
        d_pos = max(d_pos, float(np.abs(loc - b.markers[name].location).max()))
    qa, qb = skeleton_qc(a, config.side), skeleton_qc(b, config.side)
    hip_diff = np.subtract(qa["hip_center_in_pelvis_mm"], qb["hip_center_in_pelvis_mm"])
    d_qc = max(abs(qa["femur_length_mm"] - qb["femur_length_mm"]), float(np.abs(hip_diff).max()))
    return {
        "transform": {"R": RIGID_R.tolist(), "t_mm": RIGID_T.tolist()},
        "max_position_diff_m": d_pos,
        "max_rotation_diff": d_rot,
        "max_qc_diff_mm": d_qc,
        "passed": d_pos < RIGID_TOL_M and d_rot < RIGID_TOL_M and d_qc < RIGID_TOL_QC_MM,
    }


def run_parity(bones: dict[str, Path], reference: Path, out_dir: Path) -> dict[str, Any]:
    """Build with the pipeline backend into ``out_dir`` and run both checks."""
    config = SkeletonConfig()
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = build_hip_model(bones, Path(out_dir) / "model", config)
    elapsed = time.perf_counter() - start
    parity = compare_parity(ours, reference)
    rigid = compare_rigid(bones, ours, config, Path(out_dir) / "rigid")
    pkg = package_info("pystaple")
    return {
        "bones": {n: str(p) for n, p in bones.items()},
        "reference": str(reference),
        "pystaple": {"version": pkg.version, "commit": pkg.commit},
        "time_s": round(elapsed, 3),
        "parity": parity,
        "rigid_frame": rigid,
        "qc": skeleton_qc(read_osim(ours), config.side),
        "passed": parity["passed"] and rigid["passed"],
    }


# ---------------------------------------------------------------------- anatomical comparison


def _pelvis_frame(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and origin (m) of the pelvis ACS: child frame of ``ground_pelvis``."""
    joint = model.joints.get("ground_pelvis")
    if joint is None:
        raise ParityError(f"model '{model.name}' has no ground_pelvis joint")
    return joint.child.rotation, joint.child.translation


def anatomical_frames(model: Any) -> dict[str, Any]:
    """Joint centres (mm), joint frame axes and markers (mm) in the pelvis ACS.

    STAPLE bodies share the frame of the bone meshes, so all offset frames and markers are
    in one common frame; expressing them in the pelvis ACS removes any rigid difference
    between the frames of two models (BP voxel frame vs. NIfTI world).
    """
    rot, origin = _pelvis_frame(model)
    joints: dict[str, Any] = {}
    for name, joint in model.joints.items():
        if joint.parent.parent_name == "ground":
            continue
        joints[name] = {
            "centre_mm": rot.T @ (joint.parent.translation - origin) * 1000.0,
            "parent_axes": rot.T @ joint.parent.rotation,
            "child_axes": rot.T @ joint.child.rotation,
        }
    markers = {n: rot.T @ (m.location - origin) * 1000.0 for n, m in model.markers.items()}
    return {"joints": joints, "markers": markers}


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _axis_angles_deg(a: np.ndarray, b: np.ndarray) -> list[float]:
    cos = np.clip(np.sum(a * b, axis=0), -1.0, 1.0)
    return [round(float(v), 3) for v in np.degrees(np.arccos(cos))]


def compare_anatomical(ours: Path, reference: Path, side: str = "r") -> dict[str, Any]:
    """Compare two models built from different bones (e.g. msk-pipe vs. BP MATLAB STAPLE).

    Per joint: distance of the joint centres (mm) and rotation between the parent and the
    child frames (deg, total and per axis), all in the pelvis ACS of each model. Markers:
    distances (mm). QC: hip centre in the pelvis, femur length, ASIS width.
    """
    a, b = read_osim(ours), read_osim(reference)
    fa, fb = anatomical_frames(a), anatomical_frames(b)
    joints: dict[str, Any] = {}
    for name in sorted(set(fa["joints"]) & set(fb["joints"])):
        ja, jb = fa["joints"][name], fb["joints"][name]
        diff = ja["centre_mm"] - jb["centre_mm"]
        joints[name] = {
            "centre_ours_mm": [round(float(v), 3) for v in ja["centre_mm"]],
            "centre_ref_mm": [round(float(v), 3) for v in jb["centre_mm"]],
            "centre_diff_mm": [round(float(v), 3) for v in diff],
            "centre_distance_mm": round(float(np.linalg.norm(diff)), 3),
            "parent_rotation_deg": round(_angle_deg(ja["parent_axes"], jb["parent_axes"]), 3),
            "child_rotation_deg": round(_angle_deg(ja["child_axes"], jb["child_axes"]), 3),
            "child_axis_angles_deg": _axis_angles_deg(ja["child_axes"], jb["child_axes"]),
        }
    markers = {
        n: round(float(np.linalg.norm(fa["markers"][n] - fb["markers"][n])), 3)
        for n in sorted(set(fa["markers"]) & set(fb["markers"]))
    }
    qa, qb = skeleton_qc(a, side), skeleton_qc(b, side)
    qc = {
        key: {"ours": qa.get(key), "reference": qb.get(key)}
        for key in ("hip_center_in_pelvis_mm", "femur_length_mm", "asis_width_mm")
    }
    return {
        "ours": str(ours),
        "reference": str(reference),
        "frame": "pelvis ACS (ground_pelvis child frame; ISB x anterior, y superior, z right)",
        "joints": joints,
        "markers_distance_mm": markers,
        "only_in_ours": sorted(set(fa["joints"]) - set(fb["joints"])),
        "only_in_reference": sorted(set(fb["joints"]) - set(fa["joints"])),
        "qc": qc,
    }
