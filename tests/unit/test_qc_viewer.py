# SPDX-License-Identifier: Apache-2.0
"""3D defect view: scene building and camera focus (no window, no OpenGL needed)."""

import numpy as np
import pytest

pytest.importorskip("vtkmodules")

from mskpipe.qc.defects import Defect, DefectReport
from mskpipe.qc.viewer import DefectScene, find_meshes


def _plate_stl(path, z=0.0):
    from vtkmodules.vtkFiltersCore import vtkTriangleFilter
    from vtkmodules.vtkFiltersSources import vtkCubeSource
    from vtkmodules.vtkIOGeometry import vtkSTLWriter

    cube = vtkCubeSource()
    cube.SetBounds(0, 100, 0, 80, z, z + 4)
    tri = vtkTriangleFilter()
    tri.SetInputConnection(cube.GetOutputPort())
    w = vtkSTLWriter()
    w.SetFileName(str(path))
    w.SetInputConnection(tri.GetOutputPort())
    w.Write()


@pytest.fixture
def run(tmp_path):
    (tmp_path / "03_mesh" / "bones").mkdir(parents=True)
    _plate_stl(tmp_path / "03_mesh" / "bones" / "hip_right.stl")
    return tmp_path


def _hole(fixed=True):
    return Defect("hip_right", "hole", [30, 40, 2], [30.0, 40.0, 2.0], 14.0, fixed, "plate patch")


def test_find_meshes(run):
    assert find_meshes(run / "03_mesh") == {
        "hip_right": (run / "03_mesh" / "bones" / "hip_right.stl", "bone")
    }


def test_focus_looks_through_the_plate(run):
    scene = DefectScene(find_meshes(run / "03_mesh"), DefectReport([_hole(), _hole(False)]))
    scene.focus(1)
    cam = scene.renderer.GetActiveCamera()
    assert np.allclose(cam.GetFocalPoint(), (30, 40, 2))
    view = np.subtract(cam.GetPosition(), cam.GetFocalPoint())
    view /= np.linalg.norm(view)
    assert abs(view[2]) > 0.99  # along the plate normal
    assert "NOT fixed" in scene.text.GetInput()
    vis = [a.GetVisibility() for a in scene.markers[0]]
    assert not any(vis)  # only the current marker is shown
    scene.overview()
    assert all(a.GetVisibility() for parts in scene.markers for a in parts)
