# SPDX-License-Identifier: Apache-2.0
"""Holes in thin bone plates and gaps in long bones (mskpipe.labelmap.topology)."""

import numpy as np
import pytest
from scipy import ndimage

from mskpipe.labelmap.topology import euler_number, genus, repair_gaps, repair_holes
from mskpipe.qc.defects import DefectReport, length_check

EYE = np.eye(4)


def _disc_hole(mask, center, radius, axes=(0, 1)):
    g = np.indices(mask.shape).astype(float)
    r2 = sum((g[a] - center[a]) ** 2 for a in axes)
    mask[r2 < radius**2] = False


def _plate(foramen=True, holes=((40, 40, 7),)):
    """Thin plate (4 voxels) 120 x 90 = 'hip bone': big foramen + small wing holes."""
    m = np.zeros((130, 100, 12), dtype=bool)
    m[5:125, 5:95, 4:8] = True
    if foramen:
        _disc_hole(m, (95, 50, 0), 16)  # 32 mm opening = obturator foramen
    for x, y, r in holes:
        _disc_hole(m, (x, y, 0), r)
    return m


def test_euler_number_basic():
    one = np.zeros((3, 3, 3), bool)
    one[1, 1, 1] = True
    assert euler_number(one) == 1
    ring = np.zeros((7, 7, 3), bool)
    ring[1:6, 1:6, 1] = True
    ring[3, 3, 1] = False
    assert euler_number(ring) == 0 and genus(ring) == 1
    two = np.zeros((7, 3, 3), bool)
    two[1, 1, 1] = two[5, 1, 1] = True
    assert euler_number(two) == 2


def test_wing_holes_filled_foramen_kept():
    m = _plate(holes=((35, 30, 7), (45, 70, 10)))
    assert genus(m) == 3
    rep = repair_holes(m, EYE, "hip_right", expected_genus=1)
    assert rep.genus_before == 3 and rep.genus_after == 1
    holes = [d for d in rep.defects if d.kind == "hole"]
    assert len(holes) == 2 and all(d.fixed for d in holes)
    # foramen untouched, wing holes closed
    assert not rep.mask[95, 50, 5] and rep.mask[35, 30, 5] and rep.mask[45, 70, 5]
    # the defect locations point at the holes
    centers = sorted(tuple(round(c) for c in d.center_ijk[:2]) for d in holes)
    assert centers == [(35, 30), (45, 70)]
    # only the holes were added (no fill of the plate border / notches)
    added = rep.mask & ~m
    assert added.sum() == rep.added_voxels
    assert added.sum() < 2 * np.pi * (7**2 + 10**2) * 4


def test_locate_only():
    m = _plate(holes=((40, 40, 7),))
    rep = repair_holes(m, EYE, "hip_right", expected_genus=1, repair=False)
    assert rep.mask is not None and (rep.mask == m).all()
    assert len(rep.defects) == 1 and not rep.defects[0].fixed


def test_hole_too_large_reported_unfixed():
    m = _plate(foramen=False, holes=((40, 50, 25),))  # 50 mm hole, femur expects genus 0
    rep = repair_holes(m, EYE, "femur_r", expected_genus=0, max_hole_mm=40)
    assert rep.genus_after == 1
    assert [d.fixed for d in rep.defects] == [False]
    assert DefectReport(rep.defects).unfixed


def test_hole_from_opening_restored_first():
    m = _plate(holes=())
    thin = m.copy()
    thin[30:40, 30:40, 4:8] = False
    thin[30:40, 30:40, 5] = True  # 1-voxel sheet, removed by an opening
    opened = ndimage.binary_opening(thin, structure=ndimage.generate_binary_structure(3, 1))
    opened = ndimage.binary_closing(opened) | (opened & ~thin)  # keep it simple: one comp
    labels, _ = ndimage.label(opened)
    opened = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
    removed = thin & ~opened
    assert genus(opened) == 2
    rep = repair_holes(opened, EYE, "hip_left", expected_genus=1, removed_by_opening=removed)
    assert rep.genus_after == 1
    assert rep.defects[0].method == "opening"


def test_blocked_voxels_never_added():
    m = _plate(holes=((40, 40, 6),))
    blocked = np.zeros_like(m)
    blocked[40, 40, :] = True  # another structure goes through the hole
    rep = repair_holes(m, EYE, "hip_right", expected_genus=1, blocked=blocked)
    assert not (rep.mask & blocked).any()


def _fibula(gap=(70, 90)):
    m = np.zeros((40, 40, 160), dtype=bool)
    g = np.indices(m.shape).astype(float)
    x = 20 + 0.05 * g[2]  # slightly oblique shaft
    m[((g[0] - x) ** 2 + (g[1] - 20) ** 2) < 3.5**2] = True
    m[:, :, :5] = m[:, :, 155:] = False
    m[:, :, gap[0] : gap[1]] = False
    return m


def test_gap_bridged():
    m = _fibula()
    assert ndimage.label(m, structure=np.ones((3, 3, 3)))[1] == 2
    rep = repair_gaps(m, EYE, "fibula_r", max_gap_mm=40)
    assert rep.fragments == 2
    assert ndimage.label(rep.mask, structure=np.ones((3, 3, 3)))[1] == 1
    (d,) = rep.defects
    assert d.kind == "gap" and d.fixed and d.size_mm == pytest.approx(20, abs=2)
    assert d.center_ijk[2] == pytest.approx(80, abs=2)
    # the bridge is about as thick as the shaft
    sl = rep.mask[:, :, 80].sum()
    assert 0.6 * np.pi * 3.5**2 < sl < 1.6 * np.pi * 3.5**2


def test_gap_too_long_not_bridged():
    rep = repair_gaps(_fibula(gap=(30, 110)), EYE, "fibula_r", max_gap_mm=40)
    (d,) = rep.defects
    assert not d.fixed and "not bridged" in d.method


def test_side_by_side_fragments_ignored():
    m = np.zeros((40, 40, 60), dtype=bool)
    m[5:10, 5:10, 5:55] = True
    m[25:30, 5:10, 5:55] = True  # parallel, e.g. tibia part next to fibula
    assert repair_gaps(m, EYE, "x").defects == []


def test_length_check_and_report(tmp_path):
    d = length_check("fibula_r", 250, "tibia_r", 360, 0.85)
    assert d is not None and d.kind == "short" and not d.fixed
    assert length_check("fibula_r", 350, "tibia_r", 360, 0.85) is None
    rep = DefectReport([d])
    rep.write(tmp_path)
    back = DefectReport.read(tmp_path)
    assert back.defects[0].structure == "fibula_r" and back.summary()["unfixed"] == 1


def test_prune_patches_drops_unneeded_pieces():
    from mskpipe.labelmap.topology import prune_patches

    m = _plate(holes=((40, 40, 6),))
    g = np.indices(m.shape)
    hole = ((g[0] - 40) ** 2 + (g[1] - 40) ** 2 < 36) & (g[2] >= 4) & (g[2] < 8)
    stray = np.zeros_like(m)
    stray[60:64, 60:64, 8:10] = True  # donor voxels on the plate surface, far from the hole
    patch = hole | stray
    (kept,) = prune_patches(m, m | patch, [patch])
    assert genus(m | kept) == 1 and not (kept & stray).any() and (kept & hole).any()


def _thin_tip_fibula():
    """Shaft (radius 3) tapering to 1-voxel tips at a 20 mm gap, 3 mm slices like MRI."""
    m = np.zeros((30, 30, 60), dtype=bool)
    g = np.indices(m.shape).astype(float)
    r2 = (g[0] - 15) ** 2 + (g[1] - 15) ** 2
    m[r2 < 9] = True
    m[:, :, 24:31] = False  # 21 mm gap
    m[:, :, 23] = r2[:, :, 23] < 1  # thin tips: removed by an opening
    m[:, :, 31] = r2[:, :, 31] < 1
    return m


def test_clean_mask_bridge_survives_opening():
    """Regression (TLEM 2.0 MRI fibula): the opening cut the bridge at the thin tips and the
    component filter then dropped the smaller fragment."""
    from mskpipe.labelmap.clean import CleanParams, Topology, clean_mask

    m = _thin_tip_fibula()
    affine = np.diag([1.0, 1.0, 3.0, 1.0])
    spacing = np.array([1.0, 1.0, 3.0])
    plain, _ = clean_mask(np.pad(m, 2), spacing, CleanParams())
    assert ndimage.label(plain)[1] == 1 and plain.sum() < 0.7 * m.sum()  # half lost
    topo = Topology("fibula_r", affine, expected_genus=0, gaps=True)
    fixed, _ = clean_mask(np.pad(m, 2), spacing, CleanParams(), topology=topo)
    assert ndimage.label(fixed)[1] == 1 and fixed.sum() > 0.95 * m.sum()
    (gap,) = [d for d in topo.defects if d.kind == "gap"]
    assert gap.fixed and len(topo.patches) == len(topo.defects)


def test_genus_matches_the_mesh_surface():
    """The genus the repair works with is the genus of the surface step ``mesh`` builds."""
    from mskpipe.config import load_config
    from mskpipe.geometry.surface import mask_to_surface
    from mskpipe.geometry.topology import component_genus

    params = load_config().mesh.bones.model_copy(
        update={"smooth_iterations": 0, "target_reduction": 0.0, "min_component_fraction": 0.0}
    )
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(60):
        labels, _ = ndimage.label(rng.random((16, 16, 16)) < 0.5)  # porous blob
        m = np.pad(labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1, 1)
        g = genus(m)
        if g is None:
            continue
        mesh = mask_to_surface(m, EYE, params).mesh
        assert sum(component_genus(mesh)) == g
        checked += 1
    assert checked > 10


def test_edge_contact_does_not_close_a_tunnel():
    """A plug touching the ring only along edges leaves the tunnel open on the surface
    (26-connectivity of the voxels counted it as closed)."""
    ring = np.zeros((7, 7, 4), bool)
    ring[1:6, 1:6, 1] = True
    ring[3, 3, 1] = False
    plug = ring.copy()
    plug[3, 3, 2] = True  # above the hole, edge contact with the ring only
    assert genus(ring) == 1
    assert genus(plug) is None  # two separate surfaces: nothing was closed


def test_donor_part_spanning_wing_hole_and_foramen_is_trimmed():
    """A donor part that would also close the obturator foramen is cut to the wing hole
    (before: nothing was filled and nothing reported)."""
    m = _plate(holes=((35, 30, 7),))
    assert genus(m) == 2
    donor = np.zeros_like(m)
    donor[5:125, 5:95, 3:9] = True  # the same bone, thicker and without holes
    rep = repair_holes(m, EYE, "hip_left", expected_genus=1, donor=donor, donor_name="MuscleMap")
    assert rep.genus_after == 1
    (d,) = rep.defects
    assert d.fixed and d.method == "MuscleMap"
    assert rep.mask[35, 30, 5] and not rep.mask[95, 50, 5]  # wing hole closed, foramen open
