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
