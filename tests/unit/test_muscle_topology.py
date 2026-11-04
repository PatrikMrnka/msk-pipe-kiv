# SPDX-License-Identifier: Apache-2.0
"""Tunnels through muscles in the label map (mskpipe.labelmap.muscle_topology)."""

import numpy as np

from mskpipe.labelmap.muscle_topology import enclosed_in_slices, repair_muscle_tunnels
from mskpipe.labelmap.topology import genus, surface_topology

EYE = np.eye(4)
BONE, A, B = 1, 11, 13


def _through() -> np.ndarray:
    """Muscle A (block) with a strip of muscle B running through it: B is a block beside A
    with a bar that enters A, crosses it and sticks out on the far side."""
    d = np.zeros((70, 40, 40), dtype=np.uint8)
    d[20:50, 8:32, 8:32] = A
    d[2:16, 8:32, 8:32] = B  # body of B, separated from A by background
    d[2:58, 18:22, 18:22] = B  # bar of B through A (tip beyond A at x 50-58)
    return d


def _ring() -> np.ndarray:
    """Muscle ring around a bone rod; the ring is thin (2 voxels) on one side."""
    d = np.zeros((50, 50, 30), dtype=np.uint8)
    i, j, k = np.indices(d.shape)
    r = np.hypot(i - 25, j - 25)
    ring = (r >= 10) & (r <= 17) & (k >= 10) & (k < 20)
    thin = (i > 25 + 10) & (np.abs(j - 25) < 4)
    ring &= ~thin | ((k >= 14) & (k < 16))
    d[ring] = A
    d[(r <= 6)] = BONE
    return d


def test_enclosed_in_slices():
    m = np.zeros((9, 9, 3), dtype=bool)
    m[1:8, 1:8, 1] = True
    m[4, 4, 1] = False
    out = enclosed_in_slices(m)
    assert out[4, 4, 1] and out.sum() == 1


def test_strip_of_other_muscle_transferred():
    d = _through()
    assert genus(d == A) == 1 and genus(d == B) == 0
    before_b = int((d == B).sum())
    res = repair_muscle_tunnels(d, EYE, {"a": A, "b": B}, [BONE])
    assert [r.structure for r in res] == ["a"]
    r = res[0]
    assert r.genus_before == 1 and r.genus_after == 0 and r.filled == 1 and r.cut == 0
    assert genus(d == A) == 0
    # the bar inside A (30 x 4 x 4) went to A, its tip beyond A (8 x 4 x 4) to background
    assert r.taken_from == {"b": 30 * 16}
    assert (d == B).sum() == before_b - 30 * 16 - 8 * 16
    assert surface_topology(d == B) == (1, 0, 0)
    assert len(r.defects) == 1 and r.defects[0].fixed and r.defects[0].method == "voxel transfer"
    assert r.patches[0].sum() == r.added_voxels == 30 * 16
    assert "detached" in r.defects[0].detail


def test_ring_around_bone_is_cut():
    d = _ring()
    bone = d == BONE
    assert genus(d == A) == 1
    res = repair_muscle_tunnels(d, EYE, {"a": A}, [BONE])
    r = res[0]
    assert r.genus_after == 0 and r.cut == 1 and r.filled == 0 and r.added_voxels == 0
    assert surface_topology(d == A) == (1, 0, 0)
    assert np.array_equal(d == BONE, bone)  # bone never taken
    assert 0 < r.removed_voxels <= 0.05 * (r.removed_voxels + (d == A).sum())
    # the cut is on the thin side of the ring
    assert r.defects[0].method == "ring cut" and r.defects[0].center_ijk[0] > 35
    assert r.patches == [None]


def test_genus_zero_untouched():
    d = np.zeros((20, 20, 20), dtype=np.uint8)
    d[3:17, 3:17, 3:17] = A
    d[0:3, 5:15, 5:15] = B
    copy = d.copy()
    assert repair_muscle_tunnels(d, EYE, {"a": A, "b": B}, [BONE]) == []
    assert np.array_equal(d, copy)


def test_volume_limit_keeps_the_muscle():
    d = _through()
    copy = d.copy()
    res = repair_muscle_tunnels(d, EYE, {"a": A, "b": B}, [BONE], max_volume_change=0.001)
    assert res[0].genus_after == 1 and res[0].defects == []
    assert np.array_equal(d, copy)


def test_thick_ring_not_cut_below_its_radius():
    d = np.zeros((50, 50, 30), dtype=np.uint8)
    i, j, k = np.indices(d.shape)
    r = np.hypot(i - 25, j - 25)
    d[(r >= 10) & (r <= 17) & (k >= 10) & (k < 20)] = A  # 7 x 10 voxels everywhere
    d[r <= 6] = BONE
    copy = d.copy()
    res = repair_muscle_tunnels(d, EYE, {"a": A}, [BONE], max_cut_radius_mm=2.0)
    assert res[0].genus_after == 1 and res[0].cut == 0
    assert np.array_equal(d, copy)
