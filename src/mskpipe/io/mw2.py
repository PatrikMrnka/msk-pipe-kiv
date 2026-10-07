# SPDX-License-Identifier: Apache-2.0
"""Files read by Muscle Wrapping 2.x (``OsimMuscleGeneratorTool``).

* :func:`write_setup` - ``setup_MuscleGeneratorTool.xml`` (``OpenSimDocument`` 20302),
* :func:`write_motion` - OpenSim ``Storage`` ``.mot`` (first row = rest pose),
* :func:`write_model` - copy of an ``.osim`` with new mesh files, coordinate defaults,
  ranges and child-frame orientations (comments and everything else kept).

All paths inside the files are relative to the setup XML (Muscle Wrapping changes the
working directory there). Geometry files are in mm with ``scale_factors 0.001``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCALE = "0.001 0.001 0.001"  # mm files, metre model
SETUP_VERSION = "20302"


@dataclass(frozen=True)
class AreaSpec:
    kind: str  # "origin" | "insertion"
    body: str
    file: str  # relative to the setup XML


@dataclass(frozen=True)
class MuscleSpec:
    name: str
    mesh_file: str
    origin: AreaSpec
    insertion: AreaSpec


@dataclass(frozen=True)
class SetupSpec:
    name: str
    model_file: str
    motion_file: str
    output_model_file: str
    muscles: Sequence[MuscleSpec]
    coordinate: str
    num_of_lines: int
    line_res: int
    decomposition_method: str
    bone_weights: str
    analysis_prefix: str | None = None  # folder must exist
    export_folder: str | None = None  # must end with '/'


def fmt(x: float) -> str:
    """Shortest round-trip representation (as OpenSim reads doubles)."""
    return repr(float(x))


def _el(parent: ET.Element, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
    e = ET.SubElement(parent, tag, attrs)
    if text is not None:
        e.text = text
    return e


def setup_xml(spec: SetupSpec) -> ET.Element:
    root = ET.Element("OpenSimDocument", Version=SETUP_VERSION)
    tool = _el(root, "MuscleGeneratorTool", name=spec.name)
    _el(tool, "model_file", spec.model_file)
    _el(tool, "motion_file", spec.motion_file)
    _el(tool, "output_model_file", spec.output_model_file)
    if spec.export_folder:
        if not spec.export_folder.endswith("/"):
            raise ValueError("export_folder must end with '/'")
        _el(tool, "export_folder", spec.export_folder)
    if spec.analysis_prefix:
        _el(tool, "output_muscle_analysis_file_prefix", spec.analysis_prefix)
    objects = _el(_el(tool, "MuscleGeneratorSet"), "objects")
    for m in spec.muscles:
        gen = _el(objects, "MuscleGenerator", name=m.name)
        geom = _el(gen, "MuscleGeometry")
        _el(geom, "body", m.origin.body)
        mesh = _el(geom, "Mesh", name=m.name)
        _el(mesh, "scale_factors", SCALE)
        _el(mesh, "mesh_file", m.mesh_file)
        areas = _el(geom, "attachment_areas")
        for area in (m.origin, m.insertion):
            a = _el(areas, "AttachmentArea", name=f"{area.kind} {area.body}")
            _el(a, "type", area.kind)
            _el(a, "body", area.body)
            _el(a, "point_file", area.file)
            _el(a, "scale_factors", SCALE)
        _el(gen, "decomposition_method", spec.decomposition_method)
        _el(gen, "coordinate", spec.coordinate)
        _el(gen, "num_of_lines", str(spec.num_of_lines))
        _el(gen, "line_res", str(spec.line_res))
    algo = _el(_el(tool, "kinematics_fibre_algorithm"), "Luca2018viaPointsAlgorithm")
    _el(algo, "bone_weights_algorithm", spec.bone_weights)
    return root


def write_setup(path: Path, spec: SetupSpec) -> Path:
    root = setup_xml(spec)
    ET.indent(root, space="\t")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def write_motion(
    path: Path,
    columns: Sequence[str],
    rows: np.ndarray,
    *,
    in_degrees: bool = True,
    duration_s: float = 1.0,
) -> Path:
    """OpenSim ``Storage`` file; ``rows`` (n, len(columns)), time from 0 to ``duration_s``."""
    rows = np.asarray(rows, dtype=float).reshape(-1, len(columns))
    n = len(rows)
    time = np.linspace(0.0, duration_s, n) if n > 1 else np.zeros(1)
    lines = [
        path.stem,
        "version=1",
        f"nRows={n}",
        f"nColumns={len(columns) + 1}",
        f"inDegrees={'yes' if in_degrees else 'no'}",
        "endheader",
        "\t".join(["time", *columns]),
    ]
    for t, row in zip(time, rows, strict=True):
        lines.append("\t".join(f"{v:.10f}" for v in (t, *row)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def orientation_angles(rot: np.ndarray) -> np.ndarray:
    """Inverse of :func:`mskpipe.io.osim.orientation_matrix` (frame-fixed x-y-z)."""
    r = np.asarray(rot, dtype=float)
    b = math.asin(max(-1.0, min(1.0, r[0, 2])))
    if abs(r[0, 2]) < 1.0 - 1e-12:
        a = math.atan2(-r[1, 2], r[2, 2])
        c = math.atan2(-r[0, 1], r[0, 0])
    else:  # gimbal lock: put everything into a
        a = math.atan2(r[2, 1], r[1, 1])
        c = 0.0
    return np.array([a, b, c])


def write_model(
    src: Path,
    dst: Path,
    *,
    mesh_files: Mapping[str, str] | None = None,
    defaults: Mapping[str, float] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
    frame_orientations: Mapping[tuple[str, str], np.ndarray] | None = None,
) -> Path:
    """Copy ``src`` to ``dst`` with edits.

    ``mesh_files``: body -> new ``mesh_file`` of its (single) Mesh; ``defaults``:
    coordinate -> ``default_value``; ``ranges``: coordinate -> new ``range``;
    ``frame_orientations``: (joint, PhysicalOffsetFrame) -> new ``orientation`` (rad);
    frame names repeat across joints in STAPLE models, so the joint is part of the key.
    """
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(Path(src).read_bytes(), parser=parser)
    model = root.find("Model")
    if model is None:
        raise ValueError(f"{src}: not an OpenSim model")
    todo = {
        "body": dict(mesh_files or {}),
        "default": dict(defaults or {}),
        "range": dict(ranges or {}),
        "frame": dict(frame_orientations or {}),
    }
    for body in model.iterfind("./BodySet/objects/Body"):
        new = todo["body"].pop(body.get("name", ""), None)
        if new is None:
            continue
        meshes = list(body.iter("Mesh"))
        if len(meshes) != 1:
            raise ValueError(f"body '{body.get('name')}' has {len(meshes)} meshes, expected 1")
        _set(meshes[0], "mesh_file", new)
    for coord in model.iter("Coordinate"):
        name = coord.get("name", "")
        if name in todo["default"]:
            _set(coord, "default_value", fmt(todo["default"].pop(name)), first=True)
        if name in todo["range"]:
            lo, hi = todo["range"].pop(name)
            _set(coord, "range", f"{fmt(lo)} {fmt(hi)}")
    for joint in model.iterfind("./JointSet/objects/*"):
        for frame in joint.iterfind("./frames/PhysicalOffsetFrame"):
            key = (joint.get("name", ""), frame.get("name", ""))
            if key in todo["frame"]:
                angles = todo["frame"].pop(key)
                _set(frame, "orientation", " ".join(fmt(v) for v in angles))
    missing = {k: sorted(v) for k, v in todo.items() if v}
    if missing:
        raise ValueError(f"{src}: not found in the model: {missing}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def _set(parent: ET.Element, tag: str, text: str, *, first: bool = False) -> None:
    """Set the text of child ``tag``, creating it (indented like its siblings)."""
    child = parent.find(tag)
    if child is None:
        child = ET.Element(tag)
        indent = parent.text if parent.text and not parent.text.strip() else "\n"
        child.tail = indent
        if first or len(parent) == 0:
            parent.insert(0, child)
        else:
            last = parent[-1]
            child.tail, last.tail = last.tail, indent
            parent.append(child)
    child.text = text


__all__ = [
    "AreaSpec",
    "MuscleSpec",
    "SetupSpec",
    "orientation_angles",
    "setup_xml",
    "write_model",
    "write_motion",
    "write_setup",
]
