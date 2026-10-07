# SPDX-License-Identifier: Apache-2.0
"""Rigid change of the common frame of an exported model (RAS+ -> OpenSim, Y up).

The pipeline works in world coordinates of the input image (RAS+, z superior). OpenSim
shows the y axis as up, so a model kept in RAS lies on its back in the GUI. The export
re-expresses everything in a frame rotated by a fixed axis permutation::

    X_osim = y_RAS (anterior),  Y_osim = z_RAS (superior),  Z_osim = x_RAS (right)

It is a proper rotation (det = +1, no mirroring) and nothing is fitted. Bone geometry,
offset frames on bodies, markers, mass centres and inertias of the ``.osim``, muscle
meshes and attachment outlines must all be rotated with the same matrix; frames whose
parent is the ground are left alone, so the rest pose solved afterwards absorbs the
rotation in the coordinates of the ground joint (pelvis_* then end up close to zero).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from mskpipe.io.osim import OsimError, orientation_matrix

# Rows: new axes expressed in RAS (p_osim = RAS_TO_OPENSIM @ p_ras).
RAS_TO_OPENSIM = np.array(
    [
        [0.0, 1.0, 0.0],  # X = anterior
        [0.0, 0.0, 1.0],  # Y = superior
        [1.0, 0.0, 0.0],  # Z = right
    ]
)

# Elements that carry positions in a body frame and that this module does not rotate.
_UNSUPPORTED = ("WrapObjectSet/objects/*", "ContactGeometrySet/objects/*", "ForceSet/objects/*")


class FrameError(ValueError):
    """The model cannot be re-expressed in the new frame."""


def check_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return ``rotation`` as a float (3, 3) array; reject anything but a proper rotation."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise FrameError(f"rotation must be 3x3, got {r.shape}")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(r), 1.0):
        raise FrameError("rotation must be orthonormal with det = +1 (no mirroring)")
    return r


def rotate_points(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate points (N, 3) about the origin."""
    r = check_rotation(rotation)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ r.T


def homogeneous(rotation: np.ndarray) -> np.ndarray:
    """4x4 matrix of ``rotation`` (for ``TriMesh.transformed``)."""
    m = np.eye(4)
    m[:3, :3] = check_rotation(rotation)
    return m


def xyz_angles(matrix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`mskpipe.io.osim.orientation_matrix` (frame-fixed x-y-z, rad)."""
    r = np.asarray(matrix, dtype=np.float64)
    sb = float(np.clip(r[0, 2], -1.0, 1.0))
    b = np.arcsin(sb)
    if abs(sb) < 1.0 - 1e-12:
        a = np.arctan2(-r[1, 2], r[2, 2])
        c = np.arctan2(-r[0, 1], r[0, 0])
    else:  # gimbal lock: only a + c (or a - c) is defined; put it all into a
        c = 0.0
        a = np.arctan2(r[2, 1], r[1, 1])
    return np.array([a, b, c])


def rotate_osim_tree(root: ET.Element, rotation: np.ndarray) -> dict[str, int]:
    """Rotate an ``OpenSimDocument`` in place; returns counts of changed elements.

    Body frames are assumed to coincide with the common frame (the pystaple/STAPLE
    convention), so every position stored in a body frame is rotated with ``rotation``.
    """
    r = check_rotation(rotation)
    model = root.find("Model")
    if root.tag != "OpenSimDocument" or model is None:
        raise OsimError("not an OpenSim model")
    for path in _UNSUPPORTED:
        found = model.findall(path)
        if found:
            raise FrameError(
                f"{path.split('/')[0]} is not empty ({found[0].tag} '{found[0].get('name')}'):"
                " rotating it is not implemented"
            )

    counts = {"frames": 0, "frames_on_ground": 0, "markers": 0, "bodies": 0}
    for frame in model.iter("PhysicalOffsetFrame"):
        if _on_ground(frame.findtext("socket_parent", "")):
            counts["frames_on_ground"] += 1
            continue
        t = _vec(frame, "translation")
        o = _vec(frame, "orientation")
        _set(frame, "translation", r @ t)
        _set(frame, "orientation", xyz_angles(r @ orientation_matrix(o)))
        counts["frames"] += 1

    for marker in model.iterfind("./MarkerSet/objects/Marker"):
        if _on_ground(marker.findtext("socket_parent_frame", "")):
            continue
        _set(marker, "location", r @ _vec(marker, "location"))
        counts["markers"] += 1

    for body in model.iterfind("./BodySet/objects/Body"):
        if body.find("mass_center") is not None:
            _set(body, "mass_center", r @ _vec(body, "mass_center"))
        if body.find("inertia") is not None:
            xx, yy, zz, xy, xz, yz = _vec(body, "inertia", size=6)
            inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            i = r @ inertia @ r.T
            _set(body, "inertia", [i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]])
        counts["bodies"] += 1
    return counts


def rotate_osim_file(src: str | Path, dst: str | Path, rotation: np.ndarray) -> dict[str, int]:
    """Read ``src``, rotate it (:func:`rotate_osim_tree`) and write ``dst``."""
    src, dst = Path(src), Path(dst)
    try:
        tree = ET.ElementTree(ET.fromstring(src.read_bytes()))
    except ET.ParseError as exc:
        raise OsimError(f"{src.name}: not valid XML ({exc})") from exc
    counts = rotate_osim_tree(tree.getroot(), rotation)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="\t")
    tree.write(dst, encoding="UTF-8", xml_declaration=True)
    return counts


def _on_ground(socket: str) -> bool:
    return socket.strip().rstrip("/").rsplit("/", 1)[-1] == "ground"


def _vec(elem: ET.Element, tag: str, size: int = 3) -> np.ndarray:
    text = elem.findtext(tag)
    values = np.array([float(v) for v in text.split()]) if text else np.array([])
    if values.shape != (size,):
        raise OsimError(f"<{elem.tag} name='{elem.get('name')}'>: bad <{tag}>: {text!r}")
    return values


def _set(elem: ET.Element, tag: str, values) -> None:
    child = elem.find(tag)
    assert child is not None
    child.text = " ".join(f"{float(v):.17g}" for v in values)


__all__ = [
    "RAS_TO_OPENSIM",
    "FrameError",
    "check_rotation",
    "homogeneous",
    "rotate_osim_file",
    "rotate_osim_tree",
    "rotate_points",
    "xyz_angles",
]
