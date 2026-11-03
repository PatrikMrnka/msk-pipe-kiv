# SPDX-License-Identifier: Apache-2.0
"""Attachment outlines of a run, edited in the GUI (no Qt, no OpenGL: testable).

:class:`AttachmentSession` loads the bone and muscle meshes of a run (step ``mesh``) and
the outlines the run computed (step ``attachments``, usually the atlas registration) as
the starting point. Outlines are then edited point by point (every change can be undone),
imported from files (e.g. NMSBuilder) and saved as a complete folder in the layout read
by the method ``manual``::

    <folder>/<muscle>/<Stem>_Ori.vtk, <Stem>_Ins.vtk
    <folder>/manual.json        run, side and source of every area (auto/edited/imported)

Points are always kept on the bone surface (picked on it, snapped when imported).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mskpipe.io.manual_areas import INDEX_FILE, KINDS, AreaFile, default_stem, layout_path

Key = tuple[str, str]  # (muscle with side, Ori|Ins)

FORMAT = "mskpipe.manual_attachments"
VERSION = 1
SOURCES = ("auto", "edited", "imported", "new")
MIN_POINTS = 3


class SessionError(ValueError):
    """The run cannot be edited (no meshes); the message is for end users."""


@dataclass
class EditableArea:
    muscle: str
    kind: str
    stem: str
    points: np.ndarray  # (N, 3) mm, RAS, in outline order
    auto: np.ndarray | None = None  # outline computed by the run (reset target)
    source: str = "auto"
    body: str | None = None

    @property
    def key(self) -> Key:
        return self.muscle, self.kind

    @property
    def usable(self) -> bool:
        return len(self.points) >= MIN_POINTS


@dataclass
class _Change:
    key: Key
    points: np.ndarray
    source: str
    existed: bool


@dataclass
class AttachmentSession:
    run_dir: Path
    side: str
    bones: dict[str, Path]
    muscles: dict[str, Path]  # meshed muscles of the side
    areas: dict[Key, EditableArea] = field(default_factory=dict)
    method: str | None = None  # method of the attachments step the outlines come from
    source_run: Path | None = None  # run the starting outlines come from
    dirty: bool = False
    _undo: list[_Change] = field(default_factory=list)
    _surfaces: dict[str, Any] = field(default_factory=dict)
    _muscle_cache: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, run_dir: str | Path) -> AttachmentSession:
        from mskpipe.core.workspace import Workspace
        from mskpipe.steps.mesh import mesh_paths

        run_dir = Path(run_dir)
        try:
            ws = Workspace.open(run_dir)
            side = ws.load_config().skeleton.side
            mesh_dir = ws.step_dir("mesh")
            bones = mesh_paths(mesh_dir, "bone")
            muscles = {
                n: p for n, p in mesh_paths(mesh_dir, "muscle").items() if n.endswith(f"_{side}")
            }
        except Exception as exc:  # any unreadable run: one message for the user
            raise SessionError(f"{run_dir.name}: no meshes to edit on ({exc})") from exc
        if not bones or not muscles:
            raise SessionError(f"{run_dir.name}: the run has no bone or muscle meshes yet")
        session = cls(run_dir, side, bones, muscles)
        session._load_run_outlines(ws.step_dir("attachments"))
        if session.method is None:  # e.g. the step failed: the same meshes in another run
            other = _run_with_outlines(ws, side)
            if other is not None:
                session._load_run_outlines(other / "05_attachments")
                session.source_run = other if session.method else None
        else:
            session.source_run = run_dir
        return session

    def _load_run_outlines(self, folder: Path) -> None:
        """Outlines of an attachments step whose meshes are those of this session."""
        from mskpipe.io.attachment_vtk import AttachmentFileError, read_points

        index_path = folder / "attachments.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.method = index.get("method")
        bodies = {
            (a.get("muscle"), a.get("kind")): a.get("body")
            for a in index.get("areas", [])
            if a.get("status") == "ok"
        }
        for muscle, files in (index.get("muscles") or {}).items():
            if muscle not in self.muscles:
                continue
            for rel in files:
                name = Path(rel).stem  # <Stem>_<kind>
                stem, _, kind = name.rpartition("_")
                if kind not in KINDS:
                    continue
                try:
                    pts = read_points(folder / rel)
                except (AttachmentFileError, OSError):
                    continue
                self.areas[(muscle, kind)] = EditableArea(
                    muscle,
                    kind,
                    stem or default_stem(muscle),
                    pts,
                    pts.copy(),
                    "auto",
                    bodies.get((muscle, kind)),
                )

    # ------------------------------------------------------------------ geometry

    @property
    def surfaces(self) -> dict[str, Any]:
        """Body name -> :class:`~mskpipe.plugins.attachments.manual.BodySurface` (lazy)."""
        if not self._surfaces:
            from mskpipe.plugins.attachments.manual import BodySurface, load_bodies

            for body, mesh in load_bodies(self.bones, self.side, merge=True).items():
                self._surfaces[body] = BodySurface(mesh)
        return self._surfaces

    def body_meshes(self) -> dict[str, Any]:
        return {name: s.mesh for name, s in self.surfaces.items()}

    def nearest_body(self, points: np.ndarray) -> str | None:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(pts):
            return None
        dist = {b: float(np.median(s.closest(pts)[1])) for b, s in self.surfaces.items()}
        return min(dist, key=dist.__getitem__) if dist else None

    def snap(self, point: np.ndarray, body: str | None = None) -> np.ndarray:
        """Closest point on ``body`` (or the nearest body)."""
        p = np.asarray(point, dtype=float).reshape(1, 3)
        body = body if body in self.surfaces else self.nearest_body(p)
        return self.surfaces[body].closest(p)[0][0]

    # ------------------------------------------------------------------ queries

    def area_keys(self) -> list[Key]:
        """All areas of the meshed muscles of the side, given or not."""
        return [(m, k) for m in sorted(self.muscles) for k in KINDS]

    def area(self, key: Key) -> EditableArea | None:
        return self.areas.get(key)

    def points(self, key: Key) -> np.ndarray:
        area = self.areas.get(key)
        return np.empty((0, 3)) if area is None else area.points

    def state(self, key: Key) -> str:
        """auto | edited | imported | new | missing (fewer than 3 points)."""
        area = self.areas.get(key)
        if area is None or not area.usable:
            return "missing"
        return area.source

    # ------------------------------------------------------------------ editing

    def _record(self, key: Key) -> EditableArea:
        area = self.areas.get(key)
        existed = area is not None
        if area is None:
            muscle, kind = key
            area = EditableArea(muscle, kind, default_stem(muscle), np.empty((0, 3)), None, "new")
            self.areas[key] = area
        self._undo.append(_Change(key, area.points.copy(), area.source, existed))
        self.dirty = True
        return area

    def _edited(self, area: EditableArea) -> None:
        if area.source != "new":  # a new area stays new while it is drawn
            area.source = "edited"
        if area.body is None and len(area.points):
            area.body = self.nearest_body(area.points)

    def add_point(self, key: Key, point: np.ndarray, index: int | None = None) -> int:
        """Insert ``point`` before ``index`` (default: at the end); returns its index."""
        area = self._record(key)
        n = len(area.points)
        i = n if index is None else int(np.clip(index, 0, n))
        area.points = np.insert(area.points, i, np.asarray(point, dtype=float), axis=0)
        self._edited(area)
        return i

    def move_point(self, key: Key, index: int, point: np.ndarray, record: bool = True) -> None:
        """Move a point; ``record=False`` for the intermediate positions of a drag (the
        drag is one undo step: record its first move only)."""
        area = self._record(key) if record else self.areas[key]
        area.points = area.points.copy()
        area.points[index] = np.asarray(point, dtype=float)
        self.dirty = True
        self._edited(area)

    def remove_point(self, key: Key, index: int) -> None:
        area = self._record(key)
        area.points = np.delete(area.points, index, axis=0)
        self._edited(area)

    def clear(self, key: Key) -> None:
        area = self._record(key)
        area.points = np.empty((0, 3))
        area.source = "new"

    def reset(self, key: Key) -> bool:
        """Back to the outline computed by the run; False if there is none."""
        area = self.areas.get(key)
        if area is None or area.auto is None:
            return False
        area = self._record(key)
        area.points = area.auto.copy()
        area.source = "auto"
        return True

    def reorder(self, key: Key) -> bool:
        """Order the points around their centre, clockwise seen from outside the bone."""
        from mskpipe.geometry.outline import angular_order

        area = self.areas.get(key)
        if area is None or len(area.points) < 3:
            return False
        body = area.body or self.nearest_body(area.points)
        cells = self.surfaces[body].closest(area.points)[2]
        order = angular_order(area.points, self.surfaces[body].outward(cells))
        if (order == np.arange(len(order))).all():
            return False
        area = self._record(key)
        area.points = area.points[order]
        self._edited(area)
        return True

    def undo(self) -> Key | None:
        """Undo the last change; returns the area it changed."""
        if not self._undo:
            return None
        change = self._undo.pop()
        if not change.existed:
            self.areas.pop(change.key, None)
        else:
            area = self.areas[change.key]
            area.points = change.points
            area.source = change.source
        self.dirty = True
        return change.key

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    # ------------------------------------------------------------------ import

    def import_files(
        self,
        files: list[AreaFile],
        frame: str = "ras",
        order: str = "keep",
        max_projection_mm: float = 10.0,
        max_distance_mm: float = 20.0,
    ) -> list[Key]:
        """Replace areas by files (assigned muscle/kind), snapped onto the nearest bone as
        method ``manual`` does. Areas of other sides or muscles that are not meshed are
        skipped. Nothing is imported when the outlines lie far from the bones (median of
        the outlines above ``max_distance_mm``): they are in another coordinate frame,
        and snapping them would only crush them onto the nearest bone."""
        from mskpipe.io.manual_areas import read_outline, to_ras
        from mskpipe.plugins.attachments.manual import FrameMismatchError, place_outline

        affine = self.input_affine() if frame == "voxel" else None
        placed = []
        for f in files:
            if f.muscle not in self.muscles:
                continue
            pts = to_ras(read_outline(f.path), frame, affine)
            final, rec = place_outline(
                pts, self.surfaces, order=order, max_projection_mm=max_projection_mm
            )
            placed.append((f, pts, final, rec))
        medians = [rec["to_bone_median_mm"] for *_, rec in placed]
        if medians and float(np.median(medians)) > max_distance_mm:
            raise FrameMismatchError.of(
                [p for _, p, _, _ in placed], self.body_meshes(), float(np.median(medians))
            )
        done = []
        for f, _, final, rec in placed:
            key = (f.muscle, f.kind)
            area = self._record(key)
            area.points = final
            area.source = "imported"
            area.stem = f.stem
            area.body = rec["body"]
            done.append(key)
        return done

    def input_affine(self) -> np.ndarray:
        """Voxel -> RAS matrix of the run's input image."""
        from mskpipe.core.workspace import Workspace
        from mskpipe.io.manual_areas import image_affine

        return image_affine(Workspace.open(self.run_dir).input_image)

    def guess_frame(self, paths: list[Path], max_distance_mm: float = 20.0) -> Any:
        """:class:`~mskpipe.io.manual_areas.FrameGuess` of outline files by the bones of
        this run (None when no file is readable)."""
        from mskpipe.io.manual_areas import ManualAreasError, guess_frame, read_outline

        outlines = []
        for path in paths:
            try:
                outlines.append(read_outline(path))
            except (ManualAreasError, OSError, ValueError):
                continue
        try:
            affine = self.input_affine()
        except Exception:  # no input copy in the run: voxel is not tried
            affine = None
        bones = [s.mesh.vertices for s in self.surfaces.values()]
        return guess_frame(outlines, affine=affine, bones=bones, max_distance_mm=max_distance_mm)

    # ------------------------------------------------------------------ checks

    def _muscle(self, muscle: str) -> tuple[Any, float, float]:
        if muscle not in self._muscle_cache:
            from mskpipe.geometry.attachment_guard import mean_edge_length
            from mskpipe.geometry.compare import SurfaceLocator
            from mskpipe.geometry.metrics import surface_area
            from mskpipe.io.mesh_io import read_mesh

            mesh = read_mesh(self.muscles[muscle])
            self._muscle_cache[muscle] = (
                SurfaceLocator(mesh),
                mean_edge_length(mesh),
                surface_area(mesh),
            )
        return self._muscle_cache[muscle]

    def check(self, key: Key) -> dict[str, Any]:
        """What Muscle Wrapping will see: the outline projected onto the muscle (the
        export checks: collapsed, crossing, patch too large, touching the other area)."""
        from mskpipe.geometry.outline import contour_distance, outline_report

        muscle, kind = key
        area = self.areas.get(key)
        if area is None or not area.usable:
            n = 0 if area is None else len(area.points)
            return {"problems": [f"{n} point(s), at least {MIN_POINTS} needed"], "n_points": n}
        locator, edge, surface = self._muscle(muscle)
        proj, gaps = locator.closest(area.points)
        rep = outline_report(area.points, proj, gaps, edge, surface)
        other = self.areas.get((muscle, "Ins" if kind == "Ori" else "Ori"))
        if other is not None and other.usable:
            other_proj = locator.closest(other.points)[0]
            dist = contour_distance(proj, other_proj)
            rep["other_area_mm"] = round(dist, 3)
            if dist < edge:
                rep["problems"].append("touches the other area")
        return rep

    # ------------------------------------------------------------------ saving

    def save(self, folder: str | Path) -> Path:
        """Write every usable area (edited or not) in the ``manual`` layout and
        ``manual.json``; old outline files of these muscles are replaced."""
        from mskpipe.io.attachment_vtk import write_points

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        entries: dict[str, dict[str, Any]] = {}
        for key in self.area_keys():
            area = self.areas.get(key)
            muscle_dir = folder / key[0]
            if muscle_dir.is_dir():
                for old in muscle_dir.glob(f"*_{key[1]}.vtk"):
                    old.unlink()
            if area is None or not area.usable:
                continue
            path = write_points(layout_path(folder, *key, area.stem), area.points)
            entries.setdefault(key[0], {})[key[1]] = {
                "file": path.relative_to(folder).as_posix(),
                "source": area.source,
                "n_points": len(area.points),
                "body": area.body or self.nearest_body(area.points),
            }
        index = {
            "format": FORMAT,
            "version": VERSION,
            "run": str(self.run_dir),
            "side": self.side,
            "frame": "ras",
            "units": "mm",
            "saved": datetime.now(UTC).isoformat(timespec="seconds"),
            "areas": entries,
        }
        (folder / INDEX_FILE).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        self.dirty = False
        return folder

    def default_folder(self) -> Path:
        return self.run_dir / "attachments_manual"


def _run_with_outlines(ws: Any, side: str) -> Path | None:
    """Most recent other run in the same runs folder with the same meshes (step ``mesh``
    fingerprint) and side whose attachments step completed."""
    from mskpipe.core.manifest import Manifest

    try:
        mesh_fp = Manifest.load(ws.root / "manifest.json").steps["mesh"].fingerprint
    except (OSError, ValueError, KeyError):
        return None
    for path in sorted(ws.root.parent.glob("*/manifest.json"), reverse=True):
        run = path.parent
        if run == ws.root or not (run / "05_attachments" / "attachments.json").is_file():
            continue
        try:
            steps = Manifest.load(path).steps
            index = json.loads((run / "05_attachments/attachments.json").read_text("utf-8"))
        except (OSError, ValueError):
            continue
        mesh, att = steps.get("mesh"), steps.get("attachments")
        if (
            mesh is not None
            and mesh.fingerprint == mesh_fp
            and att is not None
            and att.status in ("completed", "cached")
            and index.get("side") == side
        ):
            return run
    return None


def _same_path(a: str | Path, b: str | Path) -> bool:
    import os

    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except (OSError, ValueError):
        return False


def run_with_bones(runs_dir: str | Path, image: str | Path) -> Path | None:
    """Most recent run in ``runs_dir`` of ``image`` (file or DICOM folder, as given to the
    run) whose step ``mesh`` completed: its bone meshes tell the frame of outline files."""
    from mskpipe.core.manifest import Manifest

    runs = Path(runs_dir)
    if not runs.is_dir():
        return None
    for path in sorted(runs.glob("*/input.json"), reverse=True):
        run = path.parent
        try:
            given = json.loads(path.read_text(encoding="utf-8")).get("image")
        except (OSError, ValueError, AttributeError):
            continue
        if not given or not _same_path(given, image):
            continue
        if not any((run / "03_mesh" / "bones").glob("*.stl")):
            continue
        try:
            mesh = Manifest.load(run / "manifest.json").steps.get("mesh")
        except (OSError, ValueError):
            continue
        if mesh is not None and mesh.status in ("completed", "cached"):
            return run
    return None


def bone_vertices(run_dir: str | Path) -> list[np.ndarray]:
    """Vertices of the bone meshes of a run (``03_mesh/bones/*.stl``)."""
    from mskpipe.io.mesh_io import MeshIOError, read_mesh

    out = []
    for path in sorted((Path(run_dir) / "03_mesh" / "bones").glob("*.stl")):
        try:
            out.append(read_mesh(path).vertices)
        except (MeshIOError, OSError, ValueError):
            continue
    return out


__all__ = [
    "AttachmentSession",
    "EditableArea",
    "SessionError",
    "bone_vertices",
    "run_with_bones",
]
