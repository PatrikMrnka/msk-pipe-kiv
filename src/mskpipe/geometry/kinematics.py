# SPDX-License-Identifier: Apache-2.0
"""Forward kinematics of ``CustomJoint`` models and the pose of the input image.

OpenSim/Simbody conventions (checked against OpenSim 4.6, see ``tests/unit/test_kinematics``):

* a ``CustomJoint`` moves its child frame ``M`` in its parent frame ``F`` by
  ``X_FM = (R1 R2 R3, sum_i f_i(q) a_i)``: body-fixed rotations about ``rotation1..3``,
  translations along ``translation1..3`` expressed in ``F``;
* ``X_G,child = X_G,parent  X_PF  X_FM(q)  X_CM^-1`` with the offset frames ``X_PF``
  (in the parent body) and ``X_CM`` (in the child body).

STAPLE models (pystaple) keep every body frame equal to the frame of the bone meshes (the
image world frame). The *image pose* is the set of coordinate values in which every body
frame coincides with the ground frame, i.e. the model stands as the subject lay in the
scanner and the meshes need no transformation. A joint with fewer degrees of freedom than
needed (e.g. the 1-DOF knee) reaches it only approximately; the residual is reported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mskpipe.io.osim import Joint, OffsetFrame, OsimModel, orientation_matrix

GROUND = "ground"
TRANSLATION_WEIGHT = 10.0  # residual weight of 1 m vs 1 rad in the pose fit


class KinematicsError(ValueError):
    """The model cannot be posed (unsupported joint, disconnected bodies)."""


@dataclass(frozen=True)
class ImagePose:
    values: dict[str, float]  # coordinate -> value (rad or m)
    rotational: dict[str, bool]  # coordinate -> rotational?
    joints: dict[str, dict[str, float]] = field(default_factory=dict)  # residual per joint

    def degrees(self) -> dict[str, float]:
        """Values with rotational coordinates in degrees (translations stay in m)."""
        return {
            k: float(np.degrees(v)) if self.rotational[k] else float(v)
            for k, v in self.values.items()
        }


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix of ``angle`` (rad) about ``axis`` (Rodrigues)."""
    a = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm == 0.0:
        raise KinematicsError("zero rotation axis")
    x, y, z = a / norm
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def frame_matrix(frame: OffsetFrame) -> np.ndarray:
    """4x4 transform of an offset frame in its parent (m)."""
    m = np.eye(4)
    m[:3, :3] = orientation_matrix(frame.orientation)
    m[:3, 3] = frame.translation
    return m


def joint_transform(joint: Joint, values: Mapping[str, float]) -> np.ndarray:
    """``X_FM`` of a ``CustomJoint`` for the coordinate ``values`` (missing = default)."""
    if not joint.axes:
        if joint.type == "WeldJoint":
            return np.eye(4)
        raise KinematicsError(f"joint '{joint.name}' ({joint.type}) has no SpatialTransform")
    rot, trans = np.eye(3), np.zeros(3)
    for ax in joint.axes:
        q = values.get(ax.coordinate, joint.default_values.get(ax.coordinate, 0.0))
        v = ax.value(q) if ax.coordinate else ax.value()
        if ax.rotational:
            rot = rot @ axis_rotation(ax.axis, v)
        elif v != 0.0:
            trans = trans + v * np.asarray(ax.axis) / np.linalg.norm(ax.axis)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = trans
    return m


def joint_order(model: OsimModel) -> list[Joint]:
    """Joints from ground outwards (parents before children)."""
    placed, order = {GROUND}, []
    pending = list(model.joints.values())
    while pending:
        ready = [j for j in pending if j.parent.parent_name in placed]
        if not ready:
            names = ", ".join(j.name for j in pending)
            raise KinematicsError(f"joints not connected to ground: {names}")
        for j in ready:
            order.append(j)
            placed.add(j.child.parent_name)
            pending.remove(j)
    return order


def body_transforms(model: OsimModel, values: Mapping[str, float]) -> dict[str, np.ndarray]:
    """``X_G,B`` (4x4, m) of every body for the coordinate ``values``."""
    out = {GROUND: np.eye(4)}
    for j in joint_order(model):
        x_cm = frame_matrix(j.child)
        out[j.child.parent_name] = (
            out[j.parent.parent_name]
            @ frame_matrix(j.parent)
            @ joint_transform(j, values)
            @ np.linalg.inv(x_cm)
        )
    del out[GROUND]
    return out


def coordinate_types(model: OsimModel) -> dict[str, bool]:
    """Coordinate -> True for rotational, False for translational."""
    out: dict[str, bool] = {}
    for j in model.joints.values():
        for ax in j.axes:
            if ax.coordinate:
                out.setdefault(ax.coordinate, ax.rotational)
        for name in j.coordinates:
            out.setdefault(name, True)
    return out


def image_pose(model: OsimModel) -> ImagePose:
    """Coordinate values that put every body frame onto the ground frame (least squares)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    types = coordinate_types(model)
    values: dict[str, float] = {}
    residuals: dict[str, dict[str, float]] = {}
    for j in joint_order(model):
        target = np.linalg.inv(frame_matrix(j.parent)) @ frame_matrix(j.child)
        names = list(dict.fromkeys(ax.coordinate for ax in j.axes if ax.coordinate))

        def residual(q: np.ndarray, j: Joint = j, names: list[str] = names, t=target):
            m = joint_transform(j, dict(zip(names, q, strict=True)))
            rot = Rotation.from_matrix(t[:3, :3].T @ m[:3, :3]).as_rotvec()
            return np.concatenate([rot, TRANSLATION_WEIGHT * (m[:3, 3] - t[:3, 3])])

        if names:
            best = None
            for start in _starts(names, types):
                fit = least_squares(residual, start, xtol=1e-14, ftol=1e-14, gtol=1e-14)
                q = np.array(
                    [_wrap(v) if types[n] else v for n, v in zip(names, fit.x, strict=True)]
                )
                key = (round(float(np.sum(residual(q) ** 2)), 12), float(np.abs(q).sum()))
                if best is None or key < best[0]:
                    best = (key, q)
            assert best is not None
            values.update(zip(names, (float(v) for v in best[1]), strict=True))
        r = residual(np.array([values[n] for n in names]))
        residuals[j.name] = {
            "rotation_deg": float(np.degrees(np.linalg.norm(r[:3]))),
            "translation_mm": float(np.linalg.norm(r[3:]) / TRANSLATION_WEIGHT * 1000.0),
        }
    return ImagePose(values, {k: types[k] for k in values}, residuals)


def pose_errors(
    model: OsimModel, values: Mapping[str, float], points: Mapping[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """How far each body (and its geometry ``points``, m) is from the ground frame."""
    from scipy.spatial.transform import Rotation

    out: dict[str, dict[str, Any]] = {}
    for body, x in body_transforms(model, values).items():
        rec: dict[str, Any] = {
            "rotation_deg": float(np.degrees(Rotation.from_matrix(x[:3, :3]).magnitude())),
            "translation_mm": float(np.linalg.norm(x[:3, 3]) * 1000.0),
        }
        pts = points.get(body)
        if pts is not None and len(pts):
            moved = pts @ x[:3, :3].T + x[:3, 3]
            rec["max_displacement_mm"] = float(np.linalg.norm(moved - pts, axis=1).max() * 1000)
        out[body] = rec
    return out


def _wrap(angle: float) -> float:
    """Angle in (-pi, pi]."""
    a = (angle + np.pi) % (2 * np.pi) - np.pi
    return float(np.pi if a == -np.pi else a)


def _starts(names: list[str], types: Mapping[str, bool]) -> list[np.ndarray]:
    """Zero start plus starts rotated by +-90 deg on each rotational coordinate."""
    starts = [np.zeros(len(names))]
    for i, n in enumerate(names):
        if types[n]:
            for s in (np.pi / 2, -np.pi / 2):
                q = np.zeros(len(names))
                q[i] = s
                starts.append(q)
    return starts


__all__ = [
    "ImagePose",
    "KinematicsError",
    "axis_rotation",
    "body_transforms",
    "coordinate_types",
    "frame_matrix",
    "image_pose",
    "joint_order",
    "joint_transform",
    "pose_errors",
]
