# SPDX-License-Identifier: Apache-2.0
"""Topology checks/repair and attachment outline checks for Muscle Wrapping 2.x."""

import numpy as np
import pytest

from mskpipe.config.schema import MeshParams
from mskpipe.geometry.outline import (
    contour_distance,
    merge_close,
    outline_report,
    polygon_area,
    self_crossing,
)
from mskpipe.geometry.surface import mask_to_surface
from mskpipe.geometry.topology import (
    check_surface,
    mean_edge_length,
    orient,
    remesh_mask,
    repair_candidates,
    repair_mask,
    repair_surface,
)
from mskpipe.io.mesh_io import TriMesh

GRID = np.indices((60, 60, 44)).astype(float)
CENTER = np.array([30.0, 30.0, 22.0])


def _ball(radius=12.0):
    return ((GRID - CENTER[:, None, None, None]) ** 2).sum(0) < radius**2


def _torus(ring=14.0, tube=6.0):
    r = np.sqrt(((GRID[:2] - CENTER[:2, None, None, None]) ** 2).sum(0))
    return (r - ring) ** 2 + (GRID[2] - CENTER[2]) ** 2 < tube**2


def _surface(mask):
    return mask_to_surface(mask, np.eye(4), MeshParams()).mesh


@pytest.fixture(scope="module")
def sphere() -> TriMesh:
    return _surface(_ball())


def test_check_closed_sphere(sphere):
    c = check_surface(sphere)
    assert c["ok"] and c["genus"] == 0 and c["components"] == 1 and c["problems"] == []


def test_check_torus_has_genus_one():
    c = check_surface(_surface(_torus()))
    assert not c["ok"] and c["genus"] == 1 and c["problems"] == ["genus: 1"]


def test_check_open_and_inward(sphere):
    open_mesh = TriMesh(sphere.vertices, sphere.faces[5:])
    c = check_surface(open_mesh)
    assert c["boundary_edges"] > 0 and c["genus"] is None and not c["ok"]
    inward = check_surface(TriMesh(sphere.vertices, sphere.faces[:, ::-1]))
    assert inward["problems"] == ["inward orientation"]


def test_repair_components_orientation_duplicates(sphere):
    small = TriMesh(sphere.vertices[:200] + 100.0, sphere.faces[:50])
    faces = sphere.faces.copy()
    faces[::3] = faces[::3, ::-1]  # inconsistent winding
    broken = TriMesh(
        np.vstack([sphere.vertices, small.vertices]),
        np.vstack([faces, faces[:4], small.faces + sphere.n_vertices]),
    )
    before = check_surface(broken)
    assert before["components"] == 2 and before["duplicate_faces"] == 4
    fixed, fixes = repair_surface(broken)
    after = check_surface(fixed)
    assert after["ok"], after["problems"]
    assert fixed.n_faces == sphere.n_faces and after["volume"] > 0
    assert any("small components" in f for f in fixes)
    assert any("duplicate" in f for f in fixes)
    assert any("reoriented" in f for f in fixes)


def test_orient_flips_whole_inward_mesh(sphere):
    out, flipped = orient(TriMesh(sphere.vertices, sphere.faces[:, ::-1]))
    assert flipped == sphere.n_faces and check_surface(out)["ok"]


def test_remesh_closes_a_tunnel():
    """Ellipsoid with a thin tunnel (genus 1): closing in the remesh removes it."""
    mask = (
        (GRID - CENTER[:, None, None, None]) ** 2
        / np.array([12, 12, 18.0])[:, None, None, None] ** 2
    ).sum(0) < 1
    tunnel = (GRID[0] - CENTER[0]) ** 2 + (GRID[2] - CENTER[2]) ** 2 <= 1.0
    mask &= ~tunnel
    assert check_surface(_surface(mask))["genus"] == 1
    kept = remesh_mask(mask, np.eye(4), MeshParams(), 0.0, offset=(0, 0, 0), full_shape=mask.shape)
    assert check_surface(kept)["genus"] == 1  # no closing: topology kept
    closed = remesh_mask(
        mask, np.eye(4), MeshParams(), 3.0, offset=(0, 0, 0), full_shape=mask.shape
    )
    c = check_surface(closed)
    assert c["ok"] and c["genus"] == 0


# ------------------------------------------------------------------------- outlines


def _ring(center, radius, n=12, axis=2):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.zeros((n, 3))
    i, j = [k for k in range(3) if k != axis]
    pts[:, i], pts[:, j] = radius * np.cos(a), radius * np.sin(a)
    return pts + np.asarray(center, dtype=float)


def test_outline_helpers():
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    assert polygon_area(square) == pytest.approx(1.0)
    assert not self_crossing(square, 1e-6)
    bowtie = square[[0, 2, 1, 3]]
    assert self_crossing(bowtie, 1e-6)
    assert len(merge_close(np.vstack([square, square + 1e-4]), 1e-3)) == 4
    assert contour_distance(square, square + np.array([0, 0, 2.0])) == pytest.approx(2.0)
    assert contour_distance(square, square + np.array([0.5, 0, 0])) == pytest.approx(0.0)


def test_outline_report_ok_collapsed_crossing(sphere):
    from mskpipe.geometry.compare import SurfaceLocator

    loc = SurfaceLocator(sphere)
    edge = mean_edge_length(sphere)
    area = 4 * np.pi * 12**2

    near = _ring(CENTER + np.array([0, 0, 13.0]), 5.0)
    proj, gap = loc.closest(near)
    ok = outline_report(near, proj, gap, edge, area)
    assert ok["problems"] == [] and ok["distinct"] == 12 and ok["gap_mean_mm"] < 2.0

    far = _ring(CENTER + np.array([0, 0, 150.0]), 5.0)  # missing tendon: all points hit the pole
    proj, gap = loc.closest(far)
    collapsed = outline_report(far, proj, gap, edge, area)
    assert "collapsed" in collapsed["problems"] and collapsed["gap_mean_mm"] > 100

    crossed = near[[0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]]
    proj, gap = loc.closest(crossed)
    assert "crossing" in outline_report(crossed, proj, gap, edge, area)["problems"]


# ------------------------------------------------------------------------- repair loop


def _block_with_handle(thickness=2):
    """Block with a thin arch on top (posts + beam, ``thickness`` voxels): genus 1. The arch
    spans 30 x 19 voxels, so closing up to 6 mm cannot fill it; opening cuts it."""
    m = np.zeros((60, 60, 80), dtype=bool)
    m[10:50, 10:50, 10:40] = True
    t = thickness
    m[15 : 15 + t, 28 : 28 + t, 40:60] = True
    m[45 : 45 + t, 28 : 28 + t, 40:60] = True
    m[15 : 45 + t, 28 : 28 + t, 60 : 60 + t] = True
    return m


def _remesh(mask, closing=0.0, opening=0.0):
    return remesh_mask(
        mask, np.eye(4), MeshParams(), closing, opening, offset=(0, 0, 0), full_shape=mask.shape
    )


def test_closing_does_not_cut_a_thin_handle_opening_does():
    mask = _block_with_handle()
    assert check_surface(_remesh(mask))["genus"] == 1
    assert check_surface(_remesh(mask, closing=6.0))["genus"] == 1
    fixed = check_surface(_remesh(mask, opening=1.0))
    assert fixed["ok"] and fixed["genus"] == 0


def test_candidate_order():
    c = repair_candidates(3.0, 2.0)
    assert c == [
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, 1.5),
        (0.0, 2.0),
        (1.5, 0.0),
        (3.0, 0.0),
        (1.5, 1.0),
        (1.5, 1.5),
        (1.5, 2.0),
        (3.0, 1.0),
        (3.0, 1.5),
        (3.0, 2.0),
    ]
    assert repair_candidates(0.0, 0.0) == [(0.0, 0.0)]


def _repair(mask, reference, **kw):
    args = {"max_closing_mm": 6.0, "max_opening_mm": 3.0, "max_volume_change": 0.05, **kw}
    return repair_mask(
        mask,
        np.eye(4),
        MeshParams(),
        offset=(0, 0, 0),
        full_shape=mask.shape,
        reference_volume=reference,
        **args,
    )


def test_repair_mask_takes_first_opening():
    mask = _block_with_handle()
    original = check_surface(_remesh(mask))["volume"]
    rep = _repair(mask, original)
    assert rep.ok and rep.mesh is not None and check_surface(rep.mesh)["ok"]
    used = rep.attempts[rep.best]
    assert (used["closing_mm"], used["opening_mm"]) == (0.0, 1.0)
    assert used["accepted"] and abs(used["volume_ratio"] - 1) < 0.05
    assert [a["accepted"] for a in rep.attempts] == [False, True]


def test_repair_mask_thin_diagonal_sheet():
    """Sheet one voxel thick, voxels touching only by edges (checkerboard) on a block."""
    mask = np.zeros((50, 50, 50), dtype=bool)
    mask[10:40, 10:40, 10:30] = True
    ii, jj = np.indices((30, 30))
    sheet = (ii + jj) % 2 == 0
    mask[10:40, 10:40, 30][sheet] = True
    mask[10:40, 10:40, 31][~sheet] = True  # second layer, diagonal contacts only
    original = abs(check_surface(_remesh(mask))["volume"])
    rep = _repair(mask, original)
    assert rep.ok and check_surface(rep.mesh)["ok"]


def test_repair_mask_returns_best_failed_attempt():
    """Torus with a hole wider than any closing and a tube thicker than any opening."""
    mask = _torus(ring=14.0, tube=5.0)
    original = check_surface(_surface(mask))["volume"]
    rep = _repair(mask, original)
    assert not rep.ok and rep.mesh is not None
    assert len(rep.attempts) == len(repair_candidates(6.0, 3.0))
    assert not any(a["accepted"] for a in rep.attempts)
    best = rep.attempts[rep.best]
    assert best["genus"] == 1 and best["components"] == 1
    assert abs(best["volume_ratio"] - 1) == min(
        abs(a["volume_ratio"] - 1) for a in rep.attempts if a.get("genus") == 1
    )


def test_repair_mask_volume_guard():
    """Opening that removes the handle is accepted only within the volume limit."""
    mask = _block_with_handle()
    original = check_surface(_remesh(mask))["volume"]
    rep = _repair(mask, original * 1.2)  # pretend the muscle was 20 % larger
    assert not rep.ok
    assert any("volume change" in p for a in rep.attempts for p in a["problems"])
