# SPDX-License-Identifier: Apache-2.0
"""Anatomical comparison of skeletal models built from different bones/frames."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from osim_models import write_hip_model
from scipy.spatial.transform import Rotation

from mskpipe.io.osim import orientation_matrix
from mskpipe.validation.skeleton import compare_anatomical


def _vec(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def _fmt(v) -> str:
    return " ".join(repr(float(x)) for x in v)


def move_model(src: Path, dst: Path, rot: np.ndarray, shift_m: np.ndarray) -> Path:
    """The same model with its bones (= body frames) moved rigidly, as another image frame."""
    tree = ET.parse(src)
    for frame in tree.iter("PhysicalOffsetFrame"):
        if frame.findtext("socket_parent") == "/ground":
            continue
        t, o = frame.find("translation"), frame.find("orientation")
        t.text = _fmt(rot @ _vec(t.text) + shift_m)
        m = rot @ orientation_matrix(_vec(o.text))
        o.text = _fmt(Rotation.from_matrix(m).as_euler("XYZ"))
    for marker in tree.iter("Marker"):
        loc = marker.find("location")
        loc.text = _fmt(rot @ _vec(loc.text) + shift_m)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def test_orientation_convention_matches_scipy():
    angles = np.array([0.3, -0.7, 1.1])
    expected = Rotation.from_euler("XYZ", angles).as_matrix()
    np.testing.assert_allclose(orientation_matrix(angles), expected, atol=1e-12)


def test_rigidly_moved_model_is_identical(tmp_path):
    a = write_hip_model(tmp_path / "a.osim")
    # BP voxel frame vs. world: LPS <-> RAS and a 1.1 m shift
    rot = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    b = move_model(a, tmp_path / "b.osim", rot, np.array([0.01, -0.2, -1.105]))
    report = compare_anatomical(a, b)
    for j in report["joints"].values():
        assert j["centre_distance_mm"] == pytest.approx(0, abs=1e-6)
        assert j["child_rotation_deg"] == pytest.approx(0, abs=1e-4)
    assert max(report["markers_distance_mm"].values()) == pytest.approx(0, abs=1e-6)


def test_different_bones_give_mm_and_deg(tmp_path):
    a = write_hip_model(tmp_path / "a.osim", femur_length_mm=400.0)
    b = write_hip_model(tmp_path / "b.osim", femur_length_mm=405.0)
    report = compare_anatomical(a, b)
    assert report["joints"]["hip_r"]["centre_distance_mm"] == pytest.approx(0, abs=1e-9)
    assert report["joints"]["knee_r"]["centre_distance_mm"] == pytest.approx(5.0, abs=1e-6)
    assert report["qc"]["femur_length_mm"] == {"ours": 400.0, "reference": 405.0}
    assert report["only_in_ours"] == report["only_in_reference"] == []
