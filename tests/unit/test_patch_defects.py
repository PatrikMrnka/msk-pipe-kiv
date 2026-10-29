# SPDX-License-Identifier: Apache-2.0
"""Local patch of bone surfaces (:func:`mskpipe.geometry.topology.patch_defects`).

Smoothing a thin bone (e.g. the sacrum between its foramina) can pull vertices of two
nearby sheets onto each other: a few non-manifold edges, degenerate triangles and a hole
that a new surface from the same mask reproduces. The patch cuts the faces around them
and closes the hole; every other face stays as it was.
"""

import numpy as np
import pytest
from scipy import ndimage

from mskpipe.config import load_config
from mskpipe.geometry.checked import blocking_problems, build_surface
from mskpipe.geometry.surface import mask_to_surface
from mskpipe.geometry.topology import check_surface, patch_defects
from mskpipe.io.mesh_io import TriMesh

CONFIG = load_config()


def blob(seed: int) -> np.ndarray:
    """Smooth random solid with thin necks; seed 1 gives the defect of the sacrum seen
    on a CT (boundary edges: 2, non manifold edges: 1, misoriented edges: 1)."""
    rng = np.random.default_rng(seed)
    mask = ndimage.gaussian_filter(rng.standard_normal((56, 56, 56)), 2.5) > 0.02
    mask[:3] = mask[-3:] = False
    mask[:, :3] = mask[:, -3:] = False
    mask[:, :, :3] = mask[:, :, -3:] = False
    labels, _ = ndimage.label(mask)
    return labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1


def build(mask):
    return build_surface(
        mask,
        np.eye(4),
        "bone",
        CONFIG.mesh.bones,
        CONFIG.mesh.repair,
        offset=(0, 0, 0),
        full_shape=mask.shape,
    )


def test_smoothing_defect_of_a_thin_bone_is_patched_locally():
    mask = blob(1)
    original = mask_to_surface(mask, np.eye(4), CONFIG.mesh.bones).mesh
    first = blocking_problems("bone", check_surface(original))
    assert any(p.startswith("non manifold edges") for p in first)  # the case of the CT

    b = build(mask)
    assert b.valid and b.repaired and b.decimation == "quadric"
    assert b.repair["used"] == "local patch"
    assert b.repair["volume_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert b.repair["distance_to_original_mm"]["mean"] < 0.01

    # all but a few faces are the BP faces, unchanged
    def keys(mesh):
        return {tuple(sorted(map(tuple, np.round(mesh.vertices[f], 6)))) for f in mesh.faces}

    assert len(keys(original) & keys(b.mesh)) > 0.99 * original.n_faces


@pytest.mark.parametrize("seed", [0, 4, 6, 9])
def test_other_smoothing_defects_are_patched(seed):
    b = build(blob(seed))
    assert b.valid and b.repair["used"] == "local patch"


def test_closed_surface_is_left_alone():
    grid = np.indices((30, 30, 30))
    ball = ((grid - 15) ** 2).sum(0) < 100
    mesh = mask_to_surface(ball, np.eye(4), CONFIG.mesh.bones).mesh
    patched, fixes = patch_defects(mesh)
    assert not any("closed" in f for f in fixes)
    assert check_surface(patched)["ok"]


def test_both_parts_are_kept_and_a_missing_face_is_closed():
    grid = np.indices((40, 30, 30))
    two = (((grid - np.array([10, 15, 15]).reshape(3, 1, 1, 1)) ** 2).sum(0) < 49) | (
        ((grid - np.array([30, 15, 15]).reshape(3, 1, 1, 1)) ** 2).sum(0) < 49
    )
    mesh = mask_to_surface(two, np.eye(4), CONFIG.mesh.bones).mesh
    holed = TriMesh(mesh.vertices, mesh.faces[1:])
    patched, fixes = patch_defects(holed)
    check = check_surface(patched)
    assert check["boundary_edges"] == 0 and check["non_manifold_edges"] == 0
    assert check["components"] == 2 and "closed 1 holes" in " ".join(fixes)


def test_large_defects_are_not_patched():
    rng = np.random.default_rng(0)  # thin sheets everywhere: hundreds of defects
    noise = ndimage.gaussian_filter(rng.standard_normal((48, 48, 48)), 1.6)
    sheets = np.abs(noise) < 0.035
    sheets[:3] = sheets[-3:] = False
    labels, _ = ndimage.label(sheets)
    sheets = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
    mesh = mask_to_surface(sheets, np.eye(4), CONFIG.mesh.bones).mesh
    patched, fixes = patch_defects(mesh)
    assert fixes == [] and patched is mesh
