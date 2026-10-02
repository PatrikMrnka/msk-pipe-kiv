# SPDX-License-Identifier: Apache-2.0
"""Step ``skeleton`` (pystaple) against the model MATLAB STAPLE built in the BP pipeline.

Needs pystaple and reference data (not in the repository)::

    $MSKPIPE_REFERENCE_DIR/skeleton/
        pelvis_no_sacrum.stl, femur_r.stl, tibia_r.stl   BP: bones_STAPLE/
        bone_model.osim                                  BP: bone_model.osim (same run)

The BP model has an extra sacrum display mesh (added after STAPLE); it is ignored.
CI runs the same check on pystaple's copy of these data (tools/skeleton_parity.py --fetch).

Run: ``pixi run -e dev-cpu pytest -m reference tests/steps -v``
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.reference

REF = os.environ.get("MSKPIPE_REFERENCE_DIR")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    pytest.importorskip("pystaple")
    if not REF:
        pytest.skip("MSKPIPE_REFERENCE_DIR not set")
    root = Path(REF) / "skeleton"
    if not (root / "bone_model.osim").is_file():
        pytest.skip(f"no reference model in {root}")
    from mskpipe.validation.skeleton import find_bones, run_parity

    return run_parity(find_bones(root), root / "bone_model.osim", tmp_path_factory.mktemp("sk"))


def test_same_model_as_matlab_staple(report, record_property):
    parity = report["parity"]
    for unit, value in parity["max_abs_diff"].items():
        record_property(f"max_abs_diff_{unit}", value)
    print("\nparity:", parity["max_abs_diff"], parity["ignored"])
    assert parity["passed"], parity["failures"]


def test_model_follows_rigid_transform_of_bones(report):
    rigid = report["rigid_frame"]
    print(f"\nrigid: {rigid['max_position_diff_m']:.3g} m, {rigid['max_rotation_diff']:.3g}")
    assert rigid["passed"], rigid


def test_quality_checks(report):
    qc = report["qc"]
    assert qc["side_consistent"] and qc["problems"] == []
