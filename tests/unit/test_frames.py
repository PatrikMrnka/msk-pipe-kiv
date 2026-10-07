# SPDX-License-Identifier: Apache-2.0
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from osim_models import write_hip_model

from mskpipe.geometry.frames import (
    RAS_TO_OPENSIM,
    FrameError,
    check_rotation,
    homogeneous,
    rotate_osim_file,
    rotate_points,
    xyz_angles,
)
from mskpipe.io.mesh_io import TriMesh
from mskpipe.io.osim import orientation_matrix, read_osim

R = RAS_TO_OPENSIM


def pose(frame) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = frame.rotation
    m[:3, 3] = frame.translation
    return m


def test_ras_to_opensim_is_a_proper_rotation():
    check_rotation(R)
    assert np.allclose(R @ [0, 0, 1], [0, 1, 0])  # superior -> Y (up in OpenSim)
    assert np.allclose(R @ [0, 1, 0], [1, 0, 0])  # anterior -> X
    assert np.allclose(R @ [1, 0, 0], [0, 0, 1])  # right -> Z


def test_mirroring_is_rejected():
    with pytest.raises(FrameError):
        check_rotation(np.diag([-1.0, 1.0, 1.0]))


@pytest.mark.parametrize("seed", range(20))
def test_xyz_angles_inverts_orientation_matrix(seed):
    angles = np.random.default_rng(seed).uniform(-np.pi, np.pi, 3)
    angles[1] /= 2.0  # middle angle in (-pi/2, pi/2)
    assert np.allclose(
        orientation_matrix(xyz_angles(orientation_matrix(angles))),
        orientation_matrix(angles),
        atol=1e-12,
    )


@pytest.mark.parametrize("b", [np.pi / 2, -np.pi / 2])
def test_xyz_angles_gimbal_lock(b):
    m = orientation_matrix([0.3, b, -0.4])
    assert np.allclose(orientation_matrix(xyz_angles(m)), m, atol=1e-9)


def test_isb_pelvis_frame_in_ras_becomes_upright():
    # ISB pelvis axes (x anterior, y superior, z right) expressed in RAS = R.T
    angles = xyz_angles(R @ R.T)
    assert np.allclose(angles, 0.0)


def test_rotate_osim_keeps_joint_relations_and_ground(tmp_path):
    src = write_hip_model(tmp_path / "in.osim")
    dst = tmp_path / "out.osim"
    counts = rotate_osim_file(src, dst, R)
    a, b = read_osim(src), read_osim(dst)
    assert counts["frames_on_ground"] == 1
    assert counts["frames"] == 5
    assert counts["markers"] == 2

    rot4 = homogeneous(R)
    for name, joint in a.joints.items():
        new = b.joints[name]
        if joint.parent.parent_name == "ground":
            assert np.allclose(pose(new.parent), pose(joint.parent))
        else:  # frames on bodies: rotated with the common frame
            assert np.allclose(pose(new.parent), rot4 @ pose(joint.parent))
        assert np.allclose(pose(new.child), rot4 @ pose(joint.child))
        if joint.parent.parent_name != "ground":
            rel_a = np.linalg.inv(pose(joint.parent)) @ pose(joint.child)
            rel_b = np.linalg.inv(pose(new.parent)) @ pose(new.child)
            assert np.allclose(rel_a, rel_b)  # same coordinates give the same pose
    for name, marker in a.markers.items():
        assert np.allclose(b.markers[name].location, R @ marker.location)


def test_rotate_osim_mass_properties(tmp_path):
    src = tmp_path / "m.osim"
    src.write_text(
        '<?xml version="1.0" encoding="UTF-8" ?><OpenSimDocument Version="40500"><Model name="m">'
        '<BodySet><objects><Body name="b"><mass>2</mass><mass_center>1 2 3</mass_center>'
        "<inertia>1 2 3 0.1 0.2 0.3</inertia></Body></objects></BodySet>"
        "</Model></OpenSimDocument>",
        encoding="utf-8",
    )
    rotate_osim_file(src, tmp_path / "o.osim", R)
    body = ET.parse(tmp_path / "o.osim").getroot().find("Model/BodySet/objects/Body")
    c = np.array([float(v) for v in body.findtext("mass_center").split()])
    xx, yy, zz, xy, xz, yz = (float(v) for v in body.findtext("inertia").split())
    assert np.allclose(c, R @ [1, 2, 3])
    i0 = np.array([[1, 0.1, 0.2], [0.1, 2, 0.3], [0.2, 0.3, 3]])
    assert np.allclose([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], R @ i0 @ R.T)


def test_wrap_objects_are_refused(tmp_path):
    src = tmp_path / "w.osim"
    src.write_text(
        '<OpenSimDocument Version="40500"><Model name="m"><WrapObjectSet><objects>'
        '<WrapCylinder name="c"/></objects></WrapObjectSet></Model></OpenSimDocument>',
        encoding="utf-8",
    )
    with pytest.raises(FrameError):
        rotate_osim_file(src, tmp_path / "o.osim", R)


def test_meshes_and_points_keep_winding():
    mesh = TriMesh(np.eye(3), np.array([[0, 1, 2]]))
    out = mesh.transformed(homogeneous(R))
    assert np.array_equal(out.faces, mesh.faces)
    assert np.allclose(out.vertices, rotate_points(mesh.vertices, R))
