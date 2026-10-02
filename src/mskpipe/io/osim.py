# SPDX-License-Identifier: Apache-2.0
"""Minimal reader of OpenSim 4 model files (``.osim``), enough for quality checks.

Reads bodies (mass, mesh files), joints (offset frames, coordinates) and markers. Units are
those of the file (metres, radians). No OpenSim installation is needed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class OsimError(ValueError):
    """The file is not a readable OpenSim model."""


@dataclass(frozen=True)
class OffsetFrame:
    name: str
    parent: str  # socket path, e.g. "/bodyset/pelvis" or "/ground"
    translation: np.ndarray  # (3,) m, in the parent frame
    orientation: np.ndarray  # (3,) rad, frame-fixed x-y-z sequence

    @property
    def parent_name(self) -> str:
        return self.parent.rstrip("/").rsplit("/", 1)[-1]

    @property
    def rotation(self) -> np.ndarray:
        """Rotation matrix of the frame in its parent (columns = frame axes)."""
        return orientation_matrix(self.orientation)


@dataclass(frozen=True)
class Joint:
    name: str
    type: str
    parent_frame: str
    child_frame: str
    frames: dict[str, OffsetFrame]
    coordinates: dict[str, tuple[float, float]]

    @property
    def parent(self) -> OffsetFrame:
        return self.frames[self.parent_frame]

    @property
    def child(self) -> OffsetFrame:
        return self.frames[self.child_frame]


@dataclass(frozen=True)
class Body:
    name: str
    mass: float
    mesh_files: tuple[str, ...]


@dataclass(frozen=True)
class Marker:
    name: str
    body: str
    location: np.ndarray  # (3,) m, in the body frame


@dataclass(frozen=True)
class OsimModel:
    name: str
    version: int | None
    bodies: dict[str, Body] = field(default_factory=dict)
    joints: dict[str, Joint] = field(default_factory=dict)
    markers: dict[str, Marker] = field(default_factory=dict)


def orientation_matrix(xyz: np.ndarray) -> np.ndarray:
    """OpenSim ``orientation`` (frame-fixed x-y-z angles) as a rotation matrix."""
    a, b, c = (float(v) for v in xyz)
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rx @ ry @ rz


def read_osim(path: str | Path) -> OsimModel:
    """Parse an ``.osim`` file."""
    path = Path(path)
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError as exc:
        raise OsimError(f"{path.name}: not valid XML ({exc})") from exc
    model = root.find("Model")
    if root.tag != "OpenSimDocument" or model is None:
        raise OsimError(f"{path.name}: not an OpenSim model")
    version = root.get("Version")

    bodies = {}
    for b in model.iterfind("./BodySet/objects/Body"):
        meshes = tuple(m.findtext("mesh_file", "").strip() for m in b.iter("Mesh"))
        bodies[b.get("name", "")] = Body(b.get("name", ""), _float(b, "mass"), meshes)

    joints = {}
    for j in model.iterfind("./JointSet/objects/*"):
        frames = {}
        for f in j.iterfind("./frames/PhysicalOffsetFrame"):
            frames[f.get("name", "")] = OffsetFrame(
                f.get("name", ""),
                f.findtext("socket_parent", "").strip(),
                _vec(f, "translation"),
                _vec(f, "orientation"),
            )
        coords = {}
        for c in j.iterfind("./coordinates/Coordinate"):
            lo, hi = _vec(c, "range", size=2)
            coords[c.get("name", "")] = (float(lo), float(hi))
        joints[j.get("name", "")] = Joint(
            j.get("name", ""),
            j.tag,
            _frame_name(j.findtext("socket_parent_frame", "")),
            _frame_name(j.findtext("socket_child_frame", "")),
            frames,
            coords,
        )

    markers = {}
    for m in model.iterfind("./MarkerSet/objects/Marker"):
        body = m.findtext("socket_parent_frame", "").strip().rstrip("/").rsplit("/", 1)[-1]
        markers[m.get("name", "")] = Marker(m.get("name", ""), body, _vec(m, "location"))

    return OsimModel(
        model.get("name", ""),
        int(version) if version and version.isdigit() else None,
        bodies,
        joints,
        markers,
    )


def _frame_name(socket: str) -> str:
    return socket.strip().rstrip("/").rsplit("/", 1)[-1]


def _float(elem: ET.Element, tag: str) -> float:
    text = elem.findtext(tag)
    return float(text) if text is not None and text.strip() else float("nan")


def _vec(elem: ET.Element, tag: str, size: int = 3) -> np.ndarray:
    text = elem.findtext(tag)
    values = np.array([float(v) for v in text.split()]) if text else np.array([])
    if values.shape != (size,):
        raise OsimError(f"<{elem.tag} name='{elem.get('name')}'>: bad <{tag}>: {text!r}")
    return values
