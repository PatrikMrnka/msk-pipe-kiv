# SPDX-License-Identifier: Apache-2.0
"""Synthetic OpenSim hip models for tests (the structure pystaple writes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Pelvis frame of the models: rotated 90 deg about the world z axis, so that the QC must
# actually express positions in the pelvis frame.
PELVIS_ORIGIN = np.array([0.20, 0.18, 1.02])  # m
PELVIS_ORIENTATION = np.array([0.0, 0.0, np.pi / 2])


def _vec(v) -> str:
    return " ".join(repr(float(x)) for x in v)


def _frame(name: str, parent: str, t, o=(0.0, 0.0, 0.0)) -> str:
    return (
        f'<PhysicalOffsetFrame name="{name}"><socket_parent>{parent}</socket_parent>'
        f"<translation>{_vec(t)}</translation><orientation>{_vec(o)}</orientation>"
        "</PhysicalOffsetFrame>"
    )


def _joint(name: str, parent: tuple, child: tuple, coords=()) -> str:
    c = "".join(f'<Coordinate name="{n}"><range>-1.5 1.5</range></Coordinate>' for n in coords)
    return (
        f'<CustomJoint name="{name}">'
        f"<socket_parent_frame>{parent[0]}</socket_parent_frame>"
        f"<socket_child_frame>{child[0]}</socket_child_frame>"
        f"<coordinates>{c}</coordinates>"
        f"<frames>{_frame(*parent)}{_frame(*child)}</frames></CustomJoint>"
    )


def hip_model_text(
    side: str = "r",
    hip_in_pelvis_mm=(-55.0, -82.0, 95.0),
    femur_length_mm: float = 400.0,
    version: int = 40500,
) -> str:
    """Model text; pass a negative lateral hip coordinate for a mirrored model."""
    from mskpipe.io.osim import orientation_matrix

    s = side
    rot = orientation_matrix(PELVIS_ORIENTATION)
    hip = PELVIS_ORIGIN + rot @ (np.asarray(hip_in_pelvis_mm) / 1000.0)
    knee = hip - np.array([0.0, 0.0, femur_length_mm / 1000.0])
    side_z = 0.12 if s == "r" else -0.12
    bodies = "".join(
        f'<Body name="{b}"><attached_geometry><Mesh name="{b}_geom_1">'
        f"<mesh_file>Geometry/{g}.obj</mesh_file></Mesh></attached_geometry>"
        f"<mass>{m}</mass></Body>"
        for b, g, m in (
            ("pelvis", "pelvis_no_sacrum", 9.7),
            (f"femur_{s}", f"femur_{s}", 7.7),
            (f"tibia_{s}", f"tibia_{s}", 3.1),
        )
    )
    joints = "".join(
        (
            _joint(
                "ground_pelvis",
                ("ground_offset", "/ground", (0, 0, 0)),
                ("pelvis_offset", "/bodyset/pelvis", PELVIS_ORIGIN, PELVIS_ORIENTATION),
                ("pelvis_tilt", "pelvis_list"),
            ),
            _joint(
                f"hip_{s}",
                ("pelvis_offset", "/bodyset/pelvis", hip),
                (f"femur_{s}_offset", f"/bodyset/femur_{s}", hip),
                (f"hip_flexion_{s}",),
            ),
            _joint(
                f"knee_{s}",
                (f"femur_{s}_offset", f"/bodyset/femur_{s}", knee),
                (f"tibia_{s}_offset", f"/bodyset/tibia_{s}", knee),
                (f"knee_angle_{s}",),
            ),
        )
    )
    markers = "".join(
        f'<Marker name="{n}"><socket_parent_frame>/bodyset/pelvis</socket_parent_frame>'
        f"<location>{_vec(PELVIS_ORIGIN + np.array([0.0, 0.0, z]))}</location></Marker>"
        for n, z in (("RASI", side_z), ("LASI", -side_z))
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        f'<OpenSimDocument Version="{version}"><Model name="auto2020_hip_{s.upper()}">'
        f"<BodySet><objects>{bodies}</objects></BodySet>"
        f"<JointSet><objects>{joints}</objects></JointSet>"
        f"<MarkerSet><objects>{markers}</objects></MarkerSet>"
        "</Model></OpenSimDocument>\n"
    )


def write_hip_model(path: Path, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(hip_model_text(**kwargs), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- full STAPLE model
# Same structure as a pystaple model: CustomJoints with SpatialTransform (body-fixed
# rotations z-x-y at the hip, 1-DOF knee), body frames = mesh frame (world, m).

HIP_CENTER = np.array([0.0, 0.03, -0.08])  # m, world
KNEE_CENTER = np.array([0.0, 0.0, -0.48])
IMAGE_HIP_DEG = (10.0, -4.0, 3.0)  # flexion, adduction, rotation of the image pose
IMAGE_KNEE_DEG = 7.0  # knee flexion of the image pose
KNEE_OFF_AXIS_DEG = 5.0  # rotation the 1-DOF knee cannot reproduce


def _rot(axis: str, deg: float) -> np.ndarray:
    from mskpipe.geometry.kinematics import axis_rotation

    return axis_rotation({"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis], np.radians(deg))


def _axes(rot: list[tuple[str, str]], trans: list[str]) -> str:
    xyz = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}

    def axis(name, coord, ax):
        func = (
            '<LinearFunction name="function"><coefficients> 1 0</coefficients></LinearFunction>'
            if coord
            else '<Constant name="function"><value>0</value></Constant>'
        )
        return (
            f'<TransformAxis name="{name}"><coordinates>{coord}</coordinates>'
            f"<axis>{xyz[ax]}</axis>{func}</TransformAxis>"
        )

    rots = [axis(f"rotation{i + 1}", c, a) for i, (c, a) in enumerate(rot)]
    trs = [
        axis(f"translation{i + 1}", c, a) for i, (c, a) in enumerate(zip(trans, "xyz", strict=True))
    ]
    return f"<SpatialTransform>{''.join(rots + trs)}</SpatialTransform>"


def staple_model_text(side: str = "r", pelvis_rotation: np.ndarray | None = None) -> str:
    """``pelvis_rotation``: pelvis frame in the world (default: 90 deg about the world z)."""
    from mskpipe.io.mw2 import orientation_angles

    s = side
    r_pelvis = _rot("z", 90.0) if pelvis_rotation is None else np.asarray(pelvis_rotation)
    r_hip_p = r_pelvis
    r_hip_f = r_hip_p @ _rot("z", IMAGE_HIP_DEG[0]) @ _rot("x", IMAGE_HIP_DEG[1])
    r_hip_f = r_hip_f @ _rot("y", IMAGE_HIP_DEG[2])
    r_knee_f = r_hip_f
    r_knee_t = r_knee_f @ _rot("z", IMAGE_KNEE_DEG) @ _rot("x", KNEE_OFF_AXIS_DEG)

    def frame(name, parent, t, r):
        return _frame(name, parent, t, orientation_angles(r))

    def joint(name, pf, cf, coords, axes):
        c = "".join(
            f'<Coordinate name="{n}"><range>{lo} {hi}</range></Coordinate>' for n, lo, hi in coords
        )
        return (
            f'<CustomJoint name="{name}"><socket_parent_frame>{pf[0]}</socket_parent_frame>'
            f"<socket_child_frame>{cf[0]}</socket_child_frame><coordinates>{c}</coordinates>"
            f"<frames>{frame(*pf)}{frame(*cf)}</frames>{axes}</CustomJoint>"
        )

    rot90, rot120 = np.pi / 2, 2 * np.pi / 3
    joints = (
        joint(
            "ground_pelvis",
            ("ground_offset", "/ground", (0, 0, 0), np.eye(3)),
            ("pelvis_offset", "/bodyset/pelvis", PELVIS_ORIGIN, r_pelvis),
            [(n, -rot90, rot90) for n in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")]
            + [(n, -10, 10) for n in ("pelvis_tx", "pelvis_ty", "pelvis_tz")],
            _axes(
                [("pelvis_tilt", "z"), ("pelvis_list", "x"), ("pelvis_rotation", "y")],
                ["pelvis_tx", "pelvis_ty", "pelvis_tz"],
            ),
        )
        + joint(
            f"hip_{s}",
            ("pelvis_offset", "/bodyset/pelvis", HIP_CENTER, r_hip_p),
            (f"femur_{s}_offset", f"/bodyset/femur_{s}", HIP_CENTER, r_hip_f),
            [(f"hip_{n}_{s}", -rot120, rot120) for n in ("flexion", "adduction", "rotation")],
            _axes(
                [
                    (f"hip_flexion_{s}", "z"),
                    (f"hip_adduction_{s}", "x"),
                    (f"hip_rotation_{s}", "y"),
                ],
                ["", "", ""],
            ),
        )
        + joint(
            f"knee_{s}",
            (f"femur_{s}_offset", f"/bodyset/femur_{s}", KNEE_CENTER, r_knee_f),
            (f"tibia_{s}_offset", f"/bodyset/tibia_{s}", KNEE_CENTER, r_knee_t),
            [(f"knee_angle_{s}", -rot120, np.radians(10))],
            _axes([(f"knee_angle_{s}", "z"), ("", "x"), ("", "y")], ["", "", ""]),
        )
    )
    bodies = "".join(
        f'<Body name="{b}"><!--geometry--><attached_geometry><Mesh name="{b}_geom_1">'
        f"<scale_factors>0.001 0.001 0.001</scale_factors>"
        f"<mesh_file>Geometry\\{g}.obj</mesh_file></Mesh></attached_geometry>"
        f"<mass>1</mass></Body>"
        for b, g in (
            ("pelvis", "pelvis_no_sacrum"),
            (f"femur_{s}", f"femur_{s}"),
            (f"tibia_{s}", f"tibia_{s}"),
        )
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        f'<OpenSimDocument Version="40500"><Model name="auto2020_hip_{s.upper()}">'
        f"<BodySet><objects>{bodies}</objects></BodySet>"
        f"<JointSet><objects>{joints}</objects></JointSet>"
        "</Model></OpenSimDocument>\n"
    )


def write_staple_model(path: Path, side: str = "r", **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(staple_model_text(side, **kwargs), encoding="utf-8")
    return path
