# SPDX-License-Identifier: Apache-2.0
"""3D view of the run's meshes focused on each defect (VTK, no extra dependency).

* :class:`DefectScene` - renderer with the bone/muscle meshes, the repair patches and a
  marker per defect; :meth:`DefectScene.focus` points the camera at a defect (looking along
  the local surface normal, so a hole is seen through).
* :func:`show` - standalone interactive window (``mskpipe qc RUN --show``):
  mouse = rotate/pan/zoom, ``n``/``p`` = next/previous defect, ``a`` = whole model,
  ``o`` = other structures transparent/solid, ``q`` = quit.
* :func:`snapshots` - one PNG per defect, off screen (``mskpipe qc RUN --png``, report).

The GUI embeds the same scene (:mod:`mskpipe.gui.qc_dialog`).

Coordinates: the meshes of ``03_mesh/`` and the defect centres are both in the world
frame of the input image (NIfTI RAS, mm).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mskpipe.qc.defects import PATCHES_FILE, Defect, DefectReport, collect

BONE = (0.89, 0.85, 0.79)
MUSCLE = (0.75, 0.35, 0.33)
FIXED = (0.20, 0.70, 0.35)
UNFIXED = (0.90, 0.20, 0.15)
BACKGROUND = (0.13, 0.14, 0.17)
MESH_SUFFIXES = (".stl", ".obj", ".vtp", ".ply")


def _vtk():
    import vtkmodules.vtkInteractionStyle
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401  (registers the OpenGL backend)
    from vtkmodules import (
        vtkCommonCore,
        vtkCommonDataModel,
        vtkCommonTransforms,
        vtkFiltersCore,
        vtkFiltersGeneral,
        vtkFiltersSources,
        vtkIOGeometry,
        vtkIOImage,
        vtkIOPLY,
        vtkIOXML,
        vtkRenderingAnnotation,
        vtkRenderingCore,
    )

    return {
        "core": vtkCommonCore,
        "data": vtkCommonDataModel,
        "tf": vtkCommonTransforms,
        "fcore": vtkFiltersCore,
        "fgen": vtkFiltersGeneral,
        "src": vtkFiltersSources,
        "geo": vtkIOGeometry,
        "img": vtkIOImage,
        "ply": vtkIOPLY,
        "xml": vtkIOXML,
        "ann": vtkRenderingAnnotation,
        "rnd": vtkRenderingCore,
    }


def read_mesh(path: Path):
    v = _vtk()
    suffix = path.suffix.lower()
    reader = {
        ".stl": v["geo"].vtkSTLReader,
        ".obj": v["geo"].vtkOBJReader,
        ".ply": v["ply"].vtkPLYReader,
        ".vtp": v["xml"].vtkXMLPolyDataReader,
    }[suffix]()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def find_meshes(mesh_dir: Path) -> dict[str, tuple[Path, str]]:
    """name -> (path, 'bone' | 'muscle') from ``03_mesh/`` (bones .stl, muscles .obj)."""
    out: dict[str, tuple[Path, str]] = {}
    for p in sorted(mesh_dir.rglob("*")):
        if p.suffix.lower() not in MESH_SUFFIXES:
            continue
        kind = (
            "muscle"
            if ("muscle" in p.parent.name.lower() or p.suffix.lower() == ".obj")
            else "bone"
        )
        out.setdefault(p.stem, (p, kind))
    return out


def patch_surfaces(patches: Path) -> dict[int, object]:
    """Defect index (0-based) -> surface of the voxels its repair added."""
    import nibabel as nib

    v = _vtk()
    img = nib.load(str(patches))
    data = np.asarray(img.dataobj).astype(np.int16)
    labels = [int(x) for x in np.unique(data) if x > 0]
    if not labels:
        return {}
    from vtkmodules.util.numpy_support import numpy_to_vtk

    image = v["data"].vtkImageData()
    image.SetDimensions(*data.shape)
    image.GetPointData().SetScalars(numpy_to_vtk(data.ravel(order="F"), deep=True))
    mc = v["fgen"].vtkDiscreteFlyingEdges3D()
    mc.SetInputData(image)
    for i, lab in enumerate(labels):
        mc.SetValue(i, lab)
    mc.Update()
    from vtkmodules.vtkCommonMath import vtkMatrix4x4

    m = vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            m.SetElement(r, c, float(img.affine[r, c]))
    tf = v["tf"].vtkTransform()
    tf.SetMatrix(m)
    out: dict[int, object] = {}
    for lab in labels:
        th = v["fcore"].vtkThreshold()
        th.SetInputConnection(mc.GetOutputPort())
        th.SetLowerThreshold(lab - 0.5)
        th.SetUpperThreshold(lab + 0.5)
        out[lab - 1] = _transform(_geometry(th), tf)
    return out


def _geometry(alg):
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    g = vtkGeometryFilter()
    g.SetInputConnection(alg.GetOutputPort())
    g.Update()
    return g.GetOutput()


def _transform(poly, tf):
    from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter

    f = vtkTransformPolyDataFilter()
    f.SetInputData(poly)
    f.SetTransform(tf)
    f.Update()
    return f.GetOutput()


class DefectScene:
    """Meshes + defect markers in one renderer; ``focus(i)`` frames defect ``i``."""

    def __init__(
        self,
        meshes: dict[str, tuple[Path, str]],
        report: DefectReport,
        patches: dict[int, object] | None = None,
    ) -> None:
        v = self._v = _vtk()
        self.report = report
        self.renderer = v["rnd"].vtkRenderer()
        self.renderer.SetBackground(*BACKGROUND)
        self.actors: dict[str, object] = {}
        self.polys: dict[str, object] = {}
        self.kinds: dict[str, str] = {}
        for name, (path, kind) in meshes.items():
            poly = read_mesh(path)
            if poly.GetNumberOfPoints() == 0:
                continue
            self.polys[name] = poly
            self.kinds[name] = kind
            self.actors[name] = self._actor(poly, BONE if kind == "bone" else MUSCLE)
        self.markers: list[list[object]] = []  # per defect: sphere / gap line
        self.patches: dict[int, object] = {}  # repair patches, always visible
        for i, d in enumerate(report.defects):
            if patches and i in patches:
                self.patches[i] = self._actor(patches[i], FIXED if d.fixed else UNFIXED)
            self.markers.append(self._marker(d))
        self.text = v["rnd"].vtkTextActor()
        prop = self.text.GetTextProperty()
        prop.SetFontSize(15)
        prop.SetColor(1, 1, 1)
        prop.SetBackgroundColor(0, 0, 0)
        prop.SetBackgroundOpacity(0.55)
        self.text.SetDisplayPosition(12, 12)
        self.renderer.AddActor2D(self.text)
        self.current: int | None = None
        self.context_opacity = 0.25
        self.overview()

    # ------------------------------------------------------------------ building blocks
    def _actor(self, poly, color, opacity: float = 1.0):
        v = self._v
        normals = v["fcore"].vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.SplittingOff()
        mapper = v["rnd"].vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()  # patches carry label scalars
        actor = v["rnd"].vtkActor()
        actor.SetMapper(mapper)
        p = actor.GetProperty()
        p.SetColor(*color)
        p.SetOpacity(opacity)
        p.SetSpecular(0.15)
        p.BackfaceCullingOff()
        self.renderer.AddActor(actor)
        return actor

    def _marker(self, d: Defect) -> list[object]:
        v = self._v
        out: list[object] = []
        color = FIXED if d.fixed else UNFIXED
        if d.center_mm is not None:
            s = v["src"].vtkSphereSource()
            s.SetCenter(*d.center_mm)
            s.SetRadius(max((d.size_mm or 10.0) / 2 + 4.0, 6.0))
            s.SetThetaResolution(48)
            s.SetPhiResolution(24)
            s.Update()
            shell = self._actor(s.GetOutput(), color, 0.22)
            shell.GetProperty().SetSpecular(0.0)
            out.append(shell)
        if d.end_a_mm and d.end_b_mm:
            line = v["src"].vtkLineSource()
            line.SetPoint1(*d.end_a_mm)
            line.SetPoint2(*d.end_b_mm)
            tube = v["fcore"].vtkTubeFilter()
            tube.SetInputConnection(line.GetOutputPort())
            tube.SetRadius(0.8)
            tube.SetNumberOfSides(12)
            tube.Update()
            out.append(self._actor(tube.GetOutput(), color))
        return out

    # ------------------------------------------------------------------ views
    def _label(self, i: int | None) -> str:
        n = len(self.report.defects)
        if i is None:
            bad = len(self.report.unfixed)
            head = f"{n} defect(s), {bad} not fixed" if n else "No defects"
            return head + ("   [n] next  [p] previous  [o] context  [a] all" if n else "")
        d = self.report.defects[i]
        return f"[{i + 1}/{n}] {d.message()}\n{d.detail}"

    def overview(self) -> None:
        self.current = None
        for actor in self.actors.values():
            actor.GetProperty().SetOpacity(1.0)
        self._show_markers(None)
        self.text.SetInput(self._label(None))
        cam = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(0, 0, 0)
        cam.SetPosition(0, -1, 0)  # RAS: from anterior
        cam.SetViewUp(0, 0, 1)  # superior up
        self.renderer.ResetCamera()

    def focus(self, i: int) -> None:
        """Camera on defect ``i``: along the surface normal, other structures faded."""
        d = self.report.defects[i]
        self.current = i
        self.text.SetInput(self._label(i))
        self._show_markers(i)
        for name, actor in self.actors.items():
            own = name == d.structure or name.startswith(d.structure.rsplit("_", 1)[0])
            actor.GetProperty().SetOpacity(1.0 if own else self.context_opacity)
        center = np.asarray(
            d.center_mm if d.center_mm is not None else self._centroid(d.structure), dtype=float
        )
        size = d.size_mm or 30.0
        normal = self._normal_near(d.structure, center, max(size, 10.0))
        if d.end_a_mm and d.end_b_mm:  # gap: look across the bone axis
            axis = np.subtract(d.end_b_mm, d.end_a_mm)
            axis /= np.linalg.norm(axis) or 1.0
            normal = np.cross(axis, [0, 0, 1.0])
            if np.linalg.norm(normal) < 1e-3:
                normal = np.cross(axis, [1.0, 0, 0])
            normal /= np.linalg.norm(normal)
        cam = self.renderer.GetActiveCamera()
        dist = max(5.0 * size, 120.0)
        cam.SetFocalPoint(*center)
        cam.SetPosition(*(center + normal * dist))
        up = [0, 0, 1.0] if abs(normal[2]) < 0.9 else [0, 1.0, 0]
        cam.SetViewUp(*up)
        self.renderer.ResetCameraClippingRange()

    def _show_markers(self, i: int | None) -> None:
        """All markers in the overview, only the current one when focused."""
        for j, parts in enumerate(self.markers):
            for actor in parts:
                actor.SetVisibility(i is None or i == j)

    def toggle_context(self) -> None:
        self.context_opacity = 1.0 if self.context_opacity < 1.0 else 0.25
        if self.current is not None:
            self.focus(self.current)

    def _centroid(self, name: str) -> np.ndarray:
        poly = self.polys.get(name)
        if poly is None:
            return np.zeros(3)
        b = poly.GetBounds()
        return np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])

    def _normal_near(self, name: str, center: np.ndarray, radius: float) -> np.ndarray:
        """Mean outward normal of the structure's surface near ``center`` (view direction).
        A hole has the plate on all sides; the dominant normal axis is the plate normal."""
        from vtkmodules.util.numpy_support import vtk_to_numpy

        poly = self.polys.get(name)
        if poly is None:
            return np.array([0.0, -1.0, 0.0])  # anterior view in RAS
        pts = vtk_to_numpy(poly.GetPoints().GetData())
        near = np.linalg.norm(pts - center, axis=1) < radius
        if near.sum() < 10:
            near = np.argsort(np.linalg.norm(pts - center, axis=1))[:200]
        p = pts[near]
        _, _, vt = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)
        n = vt[2]  # plate normal (smallest spread)
        if n @ (center - self._centroid(name)) < 0:
            n = -n
        if n @ [0, -1.0, 0] < -0.5:  # prefer seeing it from the front
            n = -n
        return n / np.linalg.norm(n)


def load_scene(run_dir: Path) -> DefectScene:
    report = collect(run_dir)
    meshes = find_meshes(run_dir / "03_mesh")
    patches_file = run_dir / "02_labelmap" / PATCHES_FILE
    patches = patch_surfaces(patches_file) if patches_file.exists() else {}
    return DefectScene(meshes, report, patches)


def show(run_dir: Path, start: int | None = None, size: Sequence[int] = (1200, 800)) -> None:
    """Interactive window; starts at defect ``start`` (default: first unfixed or first)."""
    v = _vtk()
    scene = load_scene(run_dir)
    window = v["rnd"].vtkRenderWindow()
    window.SetSize(*size)
    window.SetWindowName("msk-pipe - geometry check")
    window.AddRenderer(scene.renderer)
    iren = v["rnd"].vtkRenderWindowInteractor()
    iren.SetRenderWindow(window)
    iren.SetInteractorStyle(_trackball())
    n = len(scene.report.defects)

    def on_key(obj, _event):
        key = obj.GetKeySym().lower()
        cur = scene.current
        if key == "n" and n:
            scene.focus(0 if cur is None else (cur + 1) % n)
        elif key == "p" and n:
            scene.focus(n - 1 if cur is None else (cur - 1) % n)
        elif key == "a":
            scene.overview()
        elif key == "o":
            scene.toggle_context()
        window.Render()

    iren.AddObserver("KeyPressEvent", on_key)
    if n:
        unfixed = [i for i, d in enumerate(scene.report.defects) if not d.fixed]
        scene.focus(start if start is not None else (unfixed[0] if unfixed else 0))
    window.Render()
    iren.Start()


def _trackball():
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

    return vtkInteractorStyleTrackballCamera()


def snapshots(
    run_dir: Path, out_dir: Path | None = None, size: Sequence[int] = (1000, 750)
) -> list[Path]:
    """PNG per defect (+ overview) rendered off screen; returns the files."""
    v = _vtk()
    scene = load_scene(run_dir)
    out_dir = out_dir or run_dir / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    window = v["rnd"].vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(*size)
    window.AddRenderer(scene.renderer)
    files: list[Path] = []

    def grab(name: str) -> None:
        window.Render()
        w2i = v["rnd"].vtkWindowToImageFilter()
        w2i.SetInput(window)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        writer = v["img"].vtkPNGWriter()
        path = out_dir / name
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        files.append(path)

    scene.overview()
    grab("overview.png")
    for i, d in enumerate(scene.report.defects):
        scene.focus(i)
        grab(f"defect_{i + 1:02d}_{d.structure}_{d.kind}.png")
    window.Finalize()
    return files


__all__ = ["DefectScene", "find_meshes", "load_scene", "show", "snapshots"]
