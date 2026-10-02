# SPDX-License-Identifier: Apache-2.0
"""Step ``labelmap`` cleaning against cleaned masks of the BP pipeline (``clean_masks.py``).

Needs reference data (not in the repository)::

    $MSKPIPE_REFERENCE_DIR/labelmap/raw/<n>.nii.gz      BP masks before cleaning
    $MSKPIPE_REFERENCE_DIR/labelmap/cleaned/<n>.nii.gz  BP cleaned masks, same run

The only intended differences are BP's voting "hole filling" (which grows the masks by
0.4-2 %) and voxels BP gave to two masks, so ours must lie inside BP's masks with
Dice >= 0.985 (thin fibula ~0.99, others > 0.99).

Run: ``pixi run -e dev-cpu pytest -m reference tests/steps -v -s``
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.reference

REF = os.environ.get("MSKPIPE_REFERENCE_DIR")
MIN_DICE = 0.985


@pytest.fixture(scope="module")
def report():
    if not REF:
        pytest.skip("MSKPIPE_REFERENCE_DIR not set")
    root = Path(REF) / "labelmap"
    if not (root / "raw").is_dir() or not (root / "cleaned").is_dir():
        pytest.skip(f"no raw/ and cleaned/ masks in {root}")
    from mskpipe.validation.labelmap import compare_with_bp, format_report

    result = compare_with_bp(root / "raw", root / "cleaned")
    print("\n" + format_report(result))
    return result


def test_matches_bp_cleaning(report, record_property):
    assert report["structures"], "no structure could be compared"
    for name, r in report["structures"].items():
        record_property(f"dice_{name}", round(r["dice"], 5))
        assert r["only_ours"] == 0, f"{name}: {r['only_ours']} voxels outside the BP mask"
        assert r["dice"] >= MIN_DICE, f"{name}: Dice {r['dice']:.5f}"
        assert r["dice"] > r["dice_raw_vs_bp"], f"{name}: cleaning moved away from BP"


def test_no_overlaps_and_all_ok(report):
    assert all(s["status"] == "ok" for s in report["labelmap"])
    assert all(s["contested"] == 0 for s in report["labelmap"])  # BP masks did not touch
