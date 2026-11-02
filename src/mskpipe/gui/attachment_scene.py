# SPDX-License-Identifier: Apache-2.0
"""3D scene of the attachment editor (VTK renderer; the window belongs to the dialog).

Shows the bones of the side (the body of the current area solid, the others faded), the
current muscle (transparent, to see how far the outline lies from it: the tendon the
segmentation misses), the current outline as a tube with numbered point handles, the
outline the run computed (grey) and the other area of the muscle. Picking helpers map
mouse positions to bone surface points, handles and outline segments.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mskpipe.qc.viewer import BACKGROUND, BONE, MUSCLE, _vtk

OUTLINE = (0.15, 0.55, 0.95)
OTHER = (0.95, 0.65, 0.15)
AUTO = (0.6, 0.6, 0.6)
SELECTED = (1.0, 0.95, 0.25)
CONTEXT_OPACITY = 0.3
HANDLE_RADIUS_MM = 1.6
TUBE_RADIUS_MM = 0.5


def _polydata(v: dict[str, Any], mesh) -> Any:
    from mskpipe.geometry.surface import to_polydata

    return to_polydata(mesh)


class AttachmentScene:
    def __init__(self, bodies: Mapping[str, Any], muscles: Mapping[str, Any] | None = None):
        """``bodies``: body name -> TriMesh; ``muscles``: muscle name -> TriMesh (loaded
        on first display through :meth:`set_muscle_mesh` when not given)."""
        v = self._v = _vtk()
        self.renderer = v["rnd"].vtkRenderer()
        self.renderer.SetBackground(*BACKGROUND)
        self.bone_actors: dict[str, Any] = {}
        for name, mesh in bodies.items():
            self.bone_actors[name] = self._surface(_polydata(v, mesh), BONE, 1.0)
        self.muscle_actors: dict[str, Any] = {}
        for name, mesh in (muscles or {}).items():
            self.set_muscle_mesh(name, mesh)
        self.show_muscle = True
        self.show_auto = True
        self.current_body: str | None = None
        self.current_muscle: str | None = None
        self.selected: int | None = None
        self._points = np.empty((0, 3))
        # outline layers
        self.outline = self._line_layer(OUTLINE, TUBE_RADIUS_MM)
        self.other = self._line_layer(OTHER, TUBE_RADIUS_MM * 0.8)
        self.auto = self._line_layer(AUTO, TUBE_RADIUS_MM * 0.6)
        self.handles = self._glyph_layer(OUTLINE, HANDLE_RADIUS_MM)
        self.handle_sel = self._glyph_layer(SELECTED, HANDLE_RADIUS_MM * 1.4)
        self.labels = self._label_layer()
        self.text = v["rnd"].vtkTextActor()
        self.text.GetTextProperty().SetFontSize(15)
        self.text.SetPosition(10, 10)
        self.renderer.AddViewProp(self.text)
        self.picker = v["rnd"].vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        self.picker.PickFromListOn()
        for actor in self.bone_actors.values():
            self.picker.AddPickList(actor)

    # ------------------------------------------------------------------ actors

    def _surface(self, poly, color, opacity) -> Any:
        v = self._v
        normals = v["fcore"].vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.SplittingOff()
        mapper = v["rnd"].vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = v["rnd"].vtkActor()
        actor.SetMapper(mapper)
        p = actor.GetProperty()
        p.SetColor(*color)
        p.SetOpacity(opacity)
        p.SetSpecular(0.15)
        self.renderer.AddActor(actor)
        return actor

    def set_muscle_mesh(self, name: str, mesh) -> None:
        if name in self.muscle_actors:
            return
        actor = self._surface(_polydata(self._v, mesh), MUSCLE, 0.35)
        actor.PickableOff()
        actor.SetVisibility(False)
        self.muscle_actors[name] = actor

    def _line_layer(self, color, radius) -> dict[str, Any]:
        v = self._v
        poly = v["data"].vtkPolyData()
        tube = v["fcore"].vtkTubeFilter()
        tube.SetInputData(poly)
        tube.SetRadius(radius)
        tube.SetNumberOfSides(10)
        mapper = v["rnd"].vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        actor = v["rnd"].vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.PickableOff()
        self.renderer.AddActor(actor)
        return {"poly": poly, "tube": tube, "actor": actor}

    def _glyph_layer(self, color, radius) -> dict[str, Any]:
        v = self._v
        poly = v["data"].vtkPolyData()
        sphere = v["src"].vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(14)
        sphere.SetPhiResolution(10)
        glyph = v["fcore"].vtkGlyph3D()
        glyph.SetInputData(poly)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.ScalingOff()
        mapper = v["rnd"].vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        actor = v["rnd"].vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.PickableOff()
        self.renderer.AddActor(actor)
        return {"poly": poly, "glyph": glyph, "actor": actor}

    def _label_layer(self) -> dict[str, Any]:
        v = self._v
        from vtkmodules.vtkRenderingLabel import vtkLabeledDataMapper

        poly = v["data"].vtkPolyData()
        mapper = vtkLabeledDataMapper()
        mapper.SetInputData(poly)
        mapper.SetLabelModeToLabelFieldData()
        prop = mapper.GetLabelTextProperty()
        prop.SetFontSize(13)
        prop.SetColor(1, 1, 1)
        prop.SetBold(True)
        prop.SetShadow(True)
        actor = v["rnd"].vtkActor2D()
        actor.SetMapper(mapper)
        self.renderer.AddViewProp(actor)
        return {"poly": poly, "mapper": mapper, "actor": actor}

    def _set_points(self, poly, points: np.ndarray, closed: bool, labels: bool = False) -> None:
        v = self._v
        pts = v["core"].vtkPoints()
        for p in np.asarray(points, dtype=float).reshape(-1, 3):
            pts.InsertNextPoint(*p)
        poly.Initialize()
        poly.SetPoints(pts)
        n = len(points)
        if closed and n >= 2:
            lines = v["data"].vtkCellArray()
            ids = [*range(n), 0] if n >= 3 else list(range(n))
            lines.InsertNextCell(len(ids))
            for i in ids:
                lines.InsertCellPoint(i)
            poly.SetLines(lines)
        if labels:
            arr = v["core"].vtkStringArray()
            arr.SetName("label")
            for i in range(n):
                arr.InsertNextValue(str(i + 1))
            poly.GetPointData().AddArray(arr)
        poly.Modified()

    # ------------------------------------------------------------------ content

    def show_area(
        self,
        muscle: str,
        body: str | None,
        points: np.ndarray,
        auto: np.ndarray | None = None,
        other: np.ndarray | None = None,
    ) -> None:
        """Display one area: its body solid, the rest faded."""
        self.current_muscle, self.current_body = muscle, body
        self._points = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.selected is not None and self.selected >= len(self._points):
            self.selected = None
        for name, actor in self.bone_actors.items():
            solid = body is None or name == body
            actor.GetProperty().SetOpacity(1.0 if solid else CONTEXT_OPACITY)
        for name, actor in self.muscle_actors.items():
            actor.SetVisibility(self.show_muscle and name == muscle)
        self._set_points(self.outline["poly"], self._points, closed=True)
        self._set_points(self.handles["poly"], self._points, closed=False)
        self._set_points(self.labels["poly"], self._points, closed=False, labels=True)
        sel = self._points[[self.selected]] if self.selected is not None else np.empty((0, 3))
        self._set_points(self.handle_sel["poly"], sel, closed=False)
        auto = np.empty((0, 3)) if auto is None or not self.show_auto else auto
        self._set_points(self.auto["poly"], auto, closed=True)
        self._set_points(self.other["poly"], np.empty((0, 3)) if other is None else other, True)

    def set_text(self, text: str) -> None:
        self.text.SetInput(text)

    # ------------------------------------------------------------------ camera

    def focus(self, points: np.ndarray, outward: np.ndarray | None = None) -> None:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        cam = self.renderer.GetActiveCamera()
        if not len(pts):
            self.overview()
            return
        center = pts.mean(axis=0)
        span = float(np.ptp(pts, axis=0).max()) if len(pts) > 1 else 20.0
        normal = np.array([0.0, -1.0, 0.0]) if outward is None else np.asarray(outward, float)
        if np.linalg.norm(normal) < 1e-9:
            normal = np.array([0.0, -1.0, 0.0])
        normal = normal / np.linalg.norm(normal)
        cam.SetFocalPoint(*center)
        cam.SetPosition(*(center + normal * max(4.0 * span, 150.0)))
        cam.SetViewUp(*([0, 0, 1.0] if abs(normal[2]) < 0.9 else [0, 1.0, 0]))
        self.renderer.ResetCameraClippingRange()

    def overview(self) -> None:
        cam = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(0, 0, 0)
        cam.SetPosition(0, -1, 0)  # RAS: from anterior
        cam.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()

    # ------------------------------------------------------------------ picking

    def to_display(self, points: np.ndarray) -> np.ndarray:
        out = []
        for p in np.asarray(points, dtype=float).reshape(-1, 3):
            self.renderer.SetWorldPoint(*p, 1.0)
            self.renderer.WorldToDisplay()
            out.append(self.renderer.GetDisplayPoint()[:2])
        return np.asarray(out, dtype=float).reshape(-1, 2)

    def handle_at(self, x: float, y: float, tol_px: float = 9.0) -> int | None:
        """Index of the point handle under the mouse."""
        if not len(self._points):
            return None
        d = np.linalg.norm(self.to_display(self._points) - (x, y), axis=1)
        i = int(np.argmin(d))
        return i if d[i] <= tol_px else None

    def segment_at(self, x: float, y: float, tol_px: float = 6.0) -> int | None:
        """Insertion index for a click on the outline between two handles."""
        n = len(self._points)
        if n < 2:
            return None
        disp = self.to_display(self._points)
        q = np.array([x, y], dtype=float)
        best, best_d = None, tol_px
        for i in range(n if n >= 3 else 1):
            a, b = disp[i], disp[(i + 1) % n]
            ab = b - a
            t = float(np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
            if not 0.05 < t < 0.95:
                continue
            d = float(np.linalg.norm(a + t * ab - q))
            if d < best_d:
                best, best_d = i + 1, d
        return best

    def pick_bone(self, x: float, y: float) -> tuple[str, np.ndarray] | None:
        """(body, surface point) under the mouse."""
        if not self.picker.Pick(x, y, 0, self.renderer):
            return None
        actor = self.picker.GetActor()
        body = next((n for n, a in self.bone_actors.items() if a is actor), None)
        if body is None:
            return None
        return body, np.asarray(self.picker.GetPickPosition(), dtype=float)


__all__ = ["AttachmentScene"]
