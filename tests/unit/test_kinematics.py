# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from osim_models import (
    IMAGE_HIP_DEG,
    IMAGE_KNEE_DEG,
    KNEE_OFF_AXIS_DEG,
    PELVIS_ORIGIN,
    write_hip_model,
    write_staple_model,
)

from mskpipe.geometry.kinematics import (
    KinematicsError,
    axis_rotation,
    body_transforms,
    frame_matrix,
    image_pose,
    joint_order,
    joint_transform,
    pose_errors,
)
from mskpipe.io.mw2 import orientation_angles, write_model
from mskpipe.io.osim import orientation_matrix, read_osim


def test_axis_rotation_matches_orientation_convention():
    a, b, c = 0.3, -0.7, 1.1
    r = axis_rotation([1, 0, 0], a) @ axis_rotation([0, 1, 0], b) @ axis_rotation([0, 0, 1], c)
    np.testing.assert_allclose(r, orientation_matrix([a, b, c]), atol=1e-15)


def test_orientation_angles_inverts_orientation_matrix():
    rng = np.random.default_rng(0)
    for _ in range(200):
        angles = rng.uniform(-np.pi, np.pi, 3)
        angles[1] = rng.uniform(-1.5, 1.5)
        r = orientation_matrix(angles)
        np.testing.assert_allclose(orientation_matrix(orientation_angles(r)), r, atol=1e-12)
    gimbal = orientation_matrix([0.4, np.pi / 2, 0.0])
    np.testing.assert_allclose(orientation_matrix(orientation_angles(gimbal)), gimbal, atol=1e-12)


def test_transform_axes_are_read(tmp_path):
    model = read_osim(write_staple_model(tmp_path / "m.osim"))
    hip = model.joints["hip_r"]
    assert [a.name for a in hip.axes][:3] == ["rotation1", "rotation2", "rotation3"]
    assert [a.coordinate for a in hip.axes] == [
        "hip_flexion_r",
        "hip_adduction_r",
        "hip_rotation_r",
        "",
        "",
        "",
    ]
    assert hip.axes[0].function == "linear" and hip.axes[0].value(0.5) == 0.5
    assert hip.axes[3].function == "constant" and not hip.axes[3].rotational
    assert model.joints["knee_r"].default_values == {"knee_angle_r": 0.0}


def test_custom_joint_rotations_are_body_fixed(tmp_path):
    hip = read_osim(write_staple_model(tmp_path / "m.osim")).joints["hip_r"]
    q = {"hip_flexion_r": 0.4, "hip_adduction_r": -0.2, "hip_rotation_r": 0.3}
    expected = (
        axis_rotation([0, 0, 1], 0.4)
        @ axis_rotation([1, 0, 0], -0.2)
        @ axis_rotation([0, 1, 0], 0.3)
    )
    np.testing.assert_allclose(joint_transform(hip, q)[:3, :3], expected, atol=1e-15)


def test_ground_pelvis_translation_in_parent_frame(tmp_path):
    model = read_osim(write_staple_model(tmp_path / "m.osim"))
    gp = model.joints["ground_pelvis"]
    m = joint_transform(gp, {"pelvis_tilt": 1.0, "pelvis_tx": 0.1, "pelvis_tz": -0.2})
    np.testing.assert_allclose(m[:3, 3], [0.1, 0.0, -0.2])


def test_image_pose_puts_bodies_on_ground(tmp_path):
    model = read_osim(write_staple_model(tmp_path / "m.osim"))
    pose = image_pose(model)
    deg = pose.degrees()
    assert deg["pelvis_tilt"] == pytest.approx(90.0, abs=1e-6)
    np.testing.assert_allclose(deg["pelvis_tx"], PELVIS_ORIGIN[0], atol=1e-9)
    hip = [deg[f"hip_{n}_r"] for n in ("flexion", "adduction", "rotation")]
    np.testing.assert_allclose(hip, IMAGE_HIP_DEG, atol=1e-6)
    assert deg["knee_angle_r"] == pytest.approx(IMAGE_KNEE_DEG, abs=1e-5)
    assert pose.joints["hip_r"]["rotation_deg"] < 1e-8
    assert pose.joints["knee_r"]["rotation_deg"] == pytest.approx(KNEE_OFF_AXIS_DEG, abs=1e-6)

    x = body_transforms(model, pose.values)
    np.testing.assert_allclose(x["pelvis"], np.eye(4), atol=1e-9)
    np.testing.assert_allclose(x["femur_r"], np.eye(4), atol=1e-9)
    tibia = np.array([[0.0, 0.0, -0.88]])  # ankle, 0.4 m below the knee
    err = pose_errors(model, pose.values, {"tibia_r": tibia})["tibia_r"]
    assert err["rotation_deg"] == pytest.approx(KNEE_OFF_AXIS_DEG, abs=1e-6)
    assert err["max_displacement_mm"] == pytest.approx(
        2 * 400 * np.sin(np.radians(KNEE_OFF_AXIS_DEG) / 2), rel=0.05
    )


def test_aligned_child_frame_removes_the_residual(tmp_path):
    src = write_staple_model(tmp_path / "m.osim")
    model = read_osim(src)
    pose = image_pose(model)
    knee = model.joints["knee_r"]
    x = frame_matrix(knee.parent) @ joint_transform(knee, pose.values)
    dst = write_model(
        src,
        tmp_path / "out" / "m.osim",
        defaults=pose.values,
        frame_orientations={("knee_r", knee.child_frame): orientation_angles(x[:3, :3])},
    )
    edited = read_osim(dst)
    assert edited.joints["hip_r"].child.orientation.tolist() == (
        model.joints["hip_r"].child.orientation.tolist()
    )  # same frame name in another joint: untouched
    x = body_transforms(edited, {})  # defaults = image pose
    for body in ("pelvis", "femur_r", "tibia_r"):
        np.testing.assert_allclose(x[body], np.eye(4), atol=1e-9)


def test_joint_order_and_errors(tmp_path):
    model = read_osim(write_staple_model(tmp_path / "m.osim"))
    assert [j.name for j in joint_order(model)] == ["ground_pelvis", "hip_r", "knee_r"]
    no_axes = read_osim(write_hip_model(tmp_path / "h.osim"))
    with pytest.raises(KinematicsError, match="no SpatialTransform"):
        image_pose(no_axes)
