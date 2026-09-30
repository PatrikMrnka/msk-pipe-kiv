# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from mskpipe.config.schema import MeshParams
from mskpipe.geometry.compare import surface_distance
from mskpipe.geometry.metrics import mesh_stats
from mskpipe.geometry.surface import SurfaceError, mask_to_surface

PARAMS = MeshParams()


def ball(shape=(40, 40, 40), center=(20, 20, 20), radius=10.0, spacing=(1.0, 1.0, 1.0)):
    grid = np.indices(shape).astype(float)
    dist2 = sum(((g - c) * s) ** 2 for g, c, s in zip(grid, center, spacing, strict=True))
    return dist2 <= radius**2


def test_sphere_is_closed_with_correct_volume():
    r = mask_to_surface(ball(), np.eye(4), PARAMS)
    stats = mesh_stats(r.mesh)
    assert stats["closed"]
    assert stats["volume"] == pytest.approx(4 / 3 * np.pi * 10**3, rel=0.03)
    assert stats["area"] == pytest.approx(4 * np.pi * 10**2, rel=0.05)
    assert r.info["components_found"] == r.info["components_kept"] == 1
    assert not r.info["touches_border"]
    assert np.allclose(r.mesh.vertices.mean(0), 20, atol=0.2)


def test_decimation_reduces_faces():
    full = mask_to_surface(ball(), np.eye(4), MeshParams(target_reduction=0.0))
    reduced = mask_to_surface(ball(), np.eye(4), MeshParams(target_reduction=0.8))
    assert reduced.mesh.n_faces == pytest.approx(0.2 * full.mesh.n_faces, rel=0.05)


def test_deterministic():
    a = mask_to_surface(ball(), np.eye(4), PARAMS).mesh
    b = mask_to_surface(ball(), np.eye(4), PARAMS).mesh
    assert np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces)


def test_world_coordinates_anisotropic_and_mirrored():
    spacing = (0.8, 0.8, 2.0)
    mask = ball(shape=(50, 50, 20), center=(25, 25, 10), radius=12.0, spacing=spacing)
    affine = np.diag([-spacing[0], spacing[1], spacing[2], 1.0])  # mirrored x (det < 0)
    affine[:3, 3] = [100.0, -50.0, 30.0]
    # no decimation: quadric decimation may leave a few non-manifold edges
    r = mask_to_surface(mask, affine, MeshParams(target_reduction=0.0))
    stats = mesh_stats(r.mesh)
    assert stats["closed"] and stats["volume"] > 0  # outward orientation kept
    assert stats["volume"] == pytest.approx(r.info["mask_volume"], rel=0.03)
    expected_center = affine[:3, :3] @ np.array([25, 25, 10]) + affine[:3, 3]
    assert np.allclose(r.mesh.vertices.mean(0), expected_center, atol=0.3)


def test_crop_offset_matches_full_volume():
    mask = ball()
    full = mask_to_surface(mask, np.eye(4), PARAMS).mesh
    crop = mask_to_surface(
        mask[8:33, 8:33, 8:33], np.eye(4), PARAMS, offset=(8, 8, 8), full_shape=mask.shape
    ).mesh
    assert surface_distance(full, crop).hausdorff < 1e-4


def test_border_is_closed_and_reported():
    mask = ball(center=(20, 20, 3))  # cut by the k = 0 face
    r = mask_to_surface(mask, np.eye(4), PARAMS)
    assert r.info["touches_border"]
    assert mesh_stats(r.mesh)["closed"]


def two_blobs(small_radius: float) -> np.ndarray:
    return ball((60, 40, 40), (18, 20, 20), 10.0) | ball((60, 40, 40), (46, 20, 20), small_radius)


@pytest.mark.parametrize(
    "fraction, small_radius, kept", [(0.1, 3.0, 1), (0.1, 10.0, 2), (1.0, 9.0, 1), (0.0, 3.0, 2)]
)
def test_component_filter(fraction, small_radius, kept):
    r = mask_to_surface(
        two_blobs(small_radius), np.eye(4), MeshParams(min_component_fraction=fraction)
    )
    assert r.info["components_found"] == 2
    assert r.info["components_kept"] == kept


def test_no_smoothing_no_decimation():
    params = MeshParams(smooth_iterations=0, target_reduction=0.0)
    stats = mesh_stats(mask_to_surface(ball(), np.eye(4), params).mesh)
    assert stats["closed"]


def test_empty_mask():
    with pytest.raises(SurfaceError, match="empty"):
        mask_to_surface(np.zeros((5, 5, 5), bool), np.eye(4), PARAMS)


def test_surface_distance_of_shifted_sphere():
    a = mask_to_surface(ball(), np.eye(4), PARAMS).mesh
    shift = np.eye(4)
    shift[0, 3] = 0.5
    d = surface_distance(a, a.transformed(shift))
    assert d.hausdorff == pytest.approx(0.5, abs=0.05)
    assert 0.2 < d.mean < 0.5
    assert surface_distance(a, a).hausdorff == 0.0
