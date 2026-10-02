# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from mskpipe.labelmap.clean import (
    CleanParams,
    clean_mask,
    element_radius,
    padded_box,
    structuring_element,
    unpad,
)

ISO = np.array([1.0, 1.0, 1.0])


def test_bp_ball_equivalent():
    # SimpleITK ball radius 1 voxel = 19 offsets (distance <= 1.5 voxels)
    assert structuring_element(1.5, ISO).sum() == 19
    assert structuring_element(1.0, ISO).sum() == 7
    assert structuring_element(2.5, ISO).sum() == 81  # SimpleITK radius 2


def test_anisotropic_element():
    se = structuring_element(3.0, np.array([1.0, 1.0, 3.0]))
    assert se.shape == (7, 7, 3)
    assert list(element_radius(3.0, np.array([0.5, 1.0, 4.0]))) == [6, 3, 0]


def _cube(shape=(30, 30, 30), lo=8, hi=22):
    m = np.zeros(shape, dtype=bool)
    m[lo:hi, lo:hi, lo:hi] = True
    return m


def test_fills_cavity_and_closes_gap():
    m = _cube()
    m[12:18, 12:18, 12:18] = False  # enclosed cavity (larger than the closing ball)
    m[8:22, 15, 8:22] = False  # one-voxel slit through the cube
    out, stats = clean_mask(m, ISO, CleanParams())
    assert out[9:21, 9:21, 9:21].all()  # corners are rounded by the opening
    assert stats["fill_added"] > 0 and stats["closing_added"] > 0


def test_growth_blocked_by_other_structures():
    m = _cube()
    m[8:22, 15, 8:22] = False
    blocked = np.zeros_like(m)
    blocked[8:22, 15, 8:22] = True  # the slit belongs to another structure
    out, stats = clean_mask(m, ISO, CleanParams(), blocked=blocked)
    assert not out[8:22, 15, 8:22].any()
    assert stats["closing_added"] == 0


def test_opening_and_components():
    m = _cube()
    m[22:26, 15, 15] = True  # one-voxel spur, removed by opening
    m[2:5, 2:5, 2:5] = True  # small separate blob (27 voxels)
    out, stats = clean_mask(m, ISO, CleanParams())
    assert not out[22:26, 15, 15].any() and not out[2:5, 2:5, 2:5].any()
    assert stats["components_removed"] == 1

    keep_all, stats = clean_mask(m, ISO, CleanParams(min_component_fraction=0.0))
    assert keep_all[3, 3, 3] and stats["components_removed"] == 0


def test_disabled_operations_keep_mask():
    m = _cube()
    m[2:5, 2:5, 2:5] = True
    params = CleanParams(closing_mm=0, opening_mm=0, fill_holes=False, min_component_fraction=0)
    out, stats = clean_mask(m, ISO, params)
    assert np.array_equal(out, m)
    assert stats == {"components": 2, "components_removed": 0}


@pytest.mark.parametrize("lo, hi", [(0, 6), (24, 30), (10, 20)])
def test_padded_crop_equals_whole_volume(lo, hi):
    rng = np.random.default_rng(1)
    vol = np.zeros((30, 30, 30), dtype=bool)
    vol[lo:hi, 5:25, 3:27] = rng.random((hi - lo, 20, 24)) > 0.25
    params = CleanParams(closing_mm=2.0, opening_mm=1.5, min_component_fraction=0.0)
    pad = np.array([3, 3, 3])
    whole, _ = clean_mask(np.pad(vol, 3), ISO, params)
    whole = whole[3:-3, 3:-3, 3:-3]

    box = (slice(lo, hi), slice(5, 25), slice(3, 27))
    grown, padding = padded_box(vol.shape, box, pad)
    crop, _ = clean_mask(np.pad(vol[grown], padding), ISO, params)
    out = np.zeros_like(vol)
    out[grown] = unpad(crop, padding)
    assert np.array_equal(out, whole)
