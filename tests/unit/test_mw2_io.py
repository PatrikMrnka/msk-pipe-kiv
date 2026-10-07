# SPDX-License-Identifier: Apache-2.0
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from osim_models import write_staple_model

from mskpipe.io.mw2 import AreaSpec, MuscleSpec, SetupSpec, write_model, write_motion, write_setup
from mskpipe.io.osim import read_osim


def test_motion_file_layout(tmp_path):
    rows = np.array([[1.5, 0.0], [2.0, 0.0], [4.0, 0.0]])
    path = write_motion(tmp_path / "hip_flexion_r.mot", ["hip_flexion_r", "knee_angle_r"], rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:6] == [
        "hip_flexion_r",
        "version=1",
        "nRows=3",
        "nColumns=3",
        "inDegrees=yes",
        "endheader",
    ]
    assert lines[6].split("\t") == ["time", "hip_flexion_r", "knee_angle_r"]
    values = np.array([[float(v) for v in line.split("\t")] for line in lines[7:]])
    np.testing.assert_allclose(values[:, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(values[:, 1:], rows)


def _spec(**kw) -> SetupSpec:
    muscle = MuscleSpec(
        "iliacus_r",
        "muscles/iliacus_r/Iliacus.obj",
        AreaSpec("origin", "pelvis", "muscles/iliacus_r/Iliacus_Ori.vtk"),
        AreaSpec("insertion", "femur_r", "muscles/iliacus_r/Iliacus_Ins.vtk"),
    )
    base = {
        "name": "mskpipe_s1",
        "model_file": "bone_model.osim",
        "motion_file": "hip_flexion_r.mot",
        "output_model_file": "bone_model_mw2.osim",
        "muscles": [muscle],
        "coordinate": "hip_flexion_r",
        "num_of_lines": 100,
        "line_res": 15,
        "decomposition_method": "kukacka",
        "bone_weights": "InverseDistance",
        "analysis_prefix": "analysis/",
    }
    return SetupSpec(**{**base, **kw})


def test_setup_xml(tmp_path):
    root = ET.parse(write_setup(tmp_path / "setup.xml", _spec())).getroot()
    assert root.tag == "OpenSimDocument" and root.get("Version") == "20302"
    tool = root.find("MuscleGeneratorTool")
    assert tool.findtext("model_file") == "bone_model.osim"
    assert tool.findtext("output_muscle_analysis_file_prefix") == "analysis/"
    assert tool.find("export_folder") is None
    gen = tool.find("MuscleGeneratorSet/objects/MuscleGenerator")
    assert gen.get("name") == "iliacus_r"
    assert gen.findtext("num_of_lines") == "100" and gen.findtext("coordinate") == "hip_flexion_r"
    mesh = gen.find("MuscleGeometry/Mesh")
    assert mesh.findtext("scale_factors") == "0.001 0.001 0.001"
    areas = gen.findall("MuscleGeometry/attachment_areas/AttachmentArea")
    assert [a.findtext("type") for a in areas] == ["origin", "insertion"]
    assert [a.findtext("body") for a in areas] == ["pelvis", "femur_r"]
    algo = tool.find("kinematics_fibre_algorithm")
    assert [c.tag for c in algo] == ["Luca2018viaPointsAlgorithm"]
    assert algo.findtext("Luca2018viaPointsAlgorithm/bone_weights_algorithm") == "InverseDistance"


def test_setup_export_folder_needs_slash(tmp_path):
    with pytest.raises(ValueError, match="end with"):
        write_setup(tmp_path / "setup.xml", _spec(export_folder="fibres"))


def test_write_model_edits(tmp_path):
    src = write_staple_model(tmp_path / "in" / "m.osim")
    dst = write_model(
        src,
        tmp_path / "out" / "m.osim",
        mesh_files={"pelvis": "Geometry/pelvis.obj"},
        defaults={"hip_flexion_r": 0.25, "pelvis_tx": 0.2},
        ranges={"pelvis_tilt": (-2.0, 2.0)},
        frame_orientations={("knee_r", "tibia_r_offset"): np.array([0.1, 0.2, 0.3])},
    )
    text = dst.read_text(encoding="utf-8")
    assert "<!--geometry-->" in text  # comments kept
    model = read_osim(dst)
    assert model.bodies["pelvis"].mesh_files == ("Geometry/pelvis.obj",)
    assert model.bodies["femur_r"].mesh_files == ("Geometry\\femur_r.obj",)
    assert model.joints["hip_r"].default_values["hip_flexion_r"] == 0.25
    assert model.joints["ground_pelvis"].default_values["pelvis_tx"] == 0.2
    assert model.joints["ground_pelvis"].coordinates["pelvis_tilt"] == (-2.0, 2.0)
    np.testing.assert_allclose(model.joints["knee_r"].child.orientation, [0.1, 0.2, 0.3])
    coord = ET.parse(dst).getroot().find(".//Coordinate[@name='hip_flexion_r']")
    assert coord[0].tag == "default_value"


def test_write_model_unknown_names(tmp_path):
    src = write_staple_model(tmp_path / "m.osim")
    with pytest.raises(ValueError, match="not found"):
        write_model(src, tmp_path / "o.osim", defaults={"nope": 1.0})
