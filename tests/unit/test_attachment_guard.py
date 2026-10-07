# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from mskpipe.geometry.attachment_guard import (
    GuardError,
    inflate_outline,
    is_degenerate,
    mean_edge_length,
    outline_stats,
)
from mskpipe.io.mesh_io import TriMesh


def _midpoint(verts: list, cache: dict, a: int, b: int) -> int:
    key = (min(a, b), max(a, b))
    if key not in cache:
        m = verts[a] + verts[b]
        verts.append(m / np.linalg.norm(m))
        cache[key] = len(verts) - 1
    return cache[key]


def icosphere(radius: float = 20.0, subdivisions: int = 4) -> TriMesh:
    t = (1.0 + 5**0.5) / 2.0
    v = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    f = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    verts = [np.asarray(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        new = []
        for a, b, c in faces:
            ab, bc, ca = (_midpoint(verts, cache, u, w) for u, w in ((a, b), (b, c), (c, a)))
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return TriMesh(np.asarray(verts) * radius, np.asarray(faces))


def test_stats_and_degeneracy():
    one = outline_stats(np.tile([1.0, 2.0, 3.0], (6, 1)))
    assert (one.n_points, one.distinct, one.span_mm) == (6, 1, 0.0)
    assert is_degenerate(one)
    ring = np.c_[
        10 * np.cos(np.linspace(0, 6, 12)), 10 * np.sin(np.linspace(0, 6, 12)), 0 * np.ones(12)
    ]
    assert not is_degenerate(outline_stats(ring))
    assert is_degenerate(outline_stats(ring[:2]))  # two points are never a patch


def test_collapsed_outline_is_inflated_to_a_simple_loop_on_the_surface():
    mesh = icosphere()
    tip = mesh.vertices[np.argmax(mesh.vertices[:, 2])]
    out = inflate_outline(mesh, np.tile(tip, (6, 1)), radius_mm=5.0)
    assert out.before.distinct == 1
    assert out.after.distinct >= 3
    assert not is_degenerate(out.after)
    assert out.patch_fraction < 0.05
    # points are mesh vertices and lie around the collapse point
    d = np.linalg.norm(mesh.vertices[None] - out.points[:, None], axis=2).min(axis=1)
    assert np.allclose(d, 0.0)
    assert np.all(np.linalg.norm(out.points - tip, axis=1) <= out.radius_mm + 1e-9)
    # closed, ordered loop: consecutive points are mesh neighbours (short steps)
    steps = np.linalg.norm(np.diff(np.vstack([out.points, out.points[:1]]), axis=0), axis=1)
    assert steps.max() < 3 * mean_edge_length(mesh)


def test_radius_grows_when_smaller_than_an_edge():
    mesh = icosphere(subdivisions=2)  # coarse: edges ~ 6 mm
    tip = mesh.vertices[0]
    out = inflate_outline(mesh, tip[None], radius_mm=0.5)
    assert out.radius_mm > 0.5
    assert out.after.distinct >= 3


def test_max_points_subsamples_in_order():
    mesh = icosphere(subdivisions=5)
    tip = mesh.vertices[0]
    out = inflate_outline(mesh, tip[None], radius_mm=8.0, max_points=10)
    assert len(out.points) == 10


def test_patch_too_large_is_an_error():
    mesh = icosphere()
    with pytest.raises(GuardError):
        inflate_outline(mesh, mesh.vertices[:1], radius_mm=30.0, max_fraction=0.25)
