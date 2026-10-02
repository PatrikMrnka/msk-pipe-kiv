# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from osim_models import PELVIS_ORIGIN, write_hip_model

from mskpipe.io.osim import OsimError, orientation_matrix, read_osim
from mskpipe.steps.skeleton import skeleton_qc


def test_orientation_is_frame_fixed_xyz():
    a, b, c = 0.3, -0.7, 1.1
    rx = orientation_matrix([a, 0, 0])
    ry = orientation_matrix([0, b, 0])
    rz = orientation_matrix([0, 0, c])
    r = orientation_matrix([a, b, c])
    assert np.allclose(r, rx @ ry @ rz)
    assert np.allclose(r.T @ r, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)
    assert np.allclose(orientation_matrix([0, 0, np.pi / 2]) @ [1, 0, 0], [0, 1, 0])


def test_read_hip_model(tmp_path):
    model = read_osim(write_hip_model(tmp_path / "m.osim"))
    assert model.name == "auto2020_hip_R" and model.version == 40500
    assert list(model.bodies) == ["pelvis", "femur_r", "tibia_r"]
    assert model.bodies["femur_r"].mesh_files == ("Geometry/femur_r.obj",)
    assert model.bodies["pelvis"].mass == pytest.approx(9.7)

    gp = model.joints["ground_pelvis"]
    assert gp.type == "CustomJoint"
    assert (gp.parent_frame, gp.child_frame) == ("ground_offset", "pelvis_offset")
    assert gp.parent.parent_name == "ground" and gp.child.parent_name == "pelvis"
    assert np.allclose(gp.child.translation, PELVIS_ORIGIN)
    assert gp.coordinates["pelvis_tilt"] == (-1.5, 1.5)
    assert model.markers["RASI"].body == "pelvis"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("<not xml", "not valid XML"),
        ("<Other><Model/></Other>", "not an OpenSim model"),
        (
            '<OpenSimDocument><Model name="m"><MarkerSet><objects><Marker name="M">'
            "<location>1 2</location></Marker></objects></MarkerSet></Model></OpenSimDocument>",
            "bad <location>",
        ),
    ],
)
def test_read_errors(tmp_path, text, match):
    path = tmp_path / "bad.osim"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(OsimError, match=match):
        read_osim(path)


def test_qc_right_hip_model(tmp_path):
    model = read_osim(write_hip_model(tmp_path / "m.osim", femur_length_mm=401.0))
    qc = skeleton_qc(model, "r")
    assert qc["side_consistent"] and qc["problems"] == []
    assert qc["hip_center_in_pelvis_mm"] == pytest.approx([-55.0, -82.0, 95.0], abs=1e-6)
    assert qc["femur_length_mm"] == pytest.approx(401.0)
    assert qc["asis_width_mm"] == pytest.approx(240.0)


def test_qc_left_hip_model(tmp_path):
    path = write_hip_model(tmp_path / "m.osim", side="l", hip_in_pelvis_mm=(-55, -82, -95))
    qc = skeleton_qc(read_osim(path), "l")
    assert qc["side_consistent"] and qc["problems"] == []


def test_qc_detects_mirrored_model(tmp_path):
    path = write_hip_model(tmp_path / "m.osim", hip_in_pelvis_mm=(-55, -82, -95))
    qc = skeleton_qc(read_osim(path), "r")
    assert qc["side_consistent"] is False
    assert "wrong side" in qc["problems"][0]


def test_qc_femur_length_and_missing_joints(tmp_path):
    short = read_osim(write_hip_model(tmp_path / "a.osim", femur_length_mm=120.0))
    assert "femur length 120 mm" in skeleton_qc(short, "r")["problems"][0]

    right = read_osim(write_hip_model(tmp_path / "b.osim"))
    qc = skeleton_qc(right, "l")  # joints of the other side are missing
    assert "side_consistent" not in qc
    assert qc["problems"] == ["joint 'hip_l' missing", "joint 'knee_l' missing"]
