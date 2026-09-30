# SPDX-License-Identifier: Apache-2.0
"""Step ``mesh`` against meshes of the BP pipeline (C++ ``mask_to_mesh``, VTK 9.4).

Needs reference data (not in the repository)::

    $MSKPIPE_REFERENCE_DIR/mesh/
        bones/<name>.nii.gz     cleaned binary mask (BP: bone_segmentations/cleaned/)
        bones/<name>.stl        BP mesh (BP: bone_models/)
        muscles/<name>.nii.gz   cleaned binary mask (BP: muscle_segmentations/cleaned/)
        muscles/<name>.obj      BP mesh (BP: muscle_models/)

BP meshes are in "voxel index x spacing" coordinates; ours are in world coordinates, so
ours are mapped back before comparison. With VTK 9.4 the meshes must be identical (up to
float32 rounding); other VTK versions get a tolerance.

Run: ``pixi run -e dev-cpu pytest -m reference tests/steps -v``
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

REF = os.environ.get("MSKPIPE_REFERENCE_DIR")
CASES = {"bones": ".stl", "muscles": ".obj"}


def _cases() -> list:
    if not REF:
        return [pytest.param(None, None, marks=pytest.mark.skip("MSKPIPE_REFERENCE_DIR not set"))]
    root = Path(REF) / "mesh"
    cases = [
        pytest.param(kind, mask, id=f"{kind}/{mask.name.removesuffix('.nii.gz')}")
        for kind in CASES
        for mask in sorted((root / kind).glob("*.nii.gz"))
    ]
    return cases or [pytest.param(None, None, marks=pytest.mark.skip(f"no masks in {root}"))]


@pytest.mark.parametrize("kind, mask_path", _cases())
def test_matches_bp_mesh(kind, mask_path, record_property):
    from mskpipe.config import PipelineConfig
    from mskpipe.geometry.compare import surface_distance
    from mskpipe.geometry.metrics import mesh_stats
    from mskpipe.geometry.surface import mask_to_surface, vtk_version
    from mskpipe.io.mesh_io import read_mesh
    from mskpipe.io.nifti import load_labelmap

    ref_path = mask_path.with_name(mask_path.name.removesuffix(".nii.gz") + CASES[kind])
    if not ref_path.is_file():
        pytest.skip(f"reference mesh missing: {ref_path.name}")

    volume = load_labelmap(mask_path)
    params = getattr(PipelineConfig().mesh, kind)
    ours = mask_to_surface(volume.data > 0, volume.affine, params).mesh

    to_bp = np.eye(4)  # world -> BP frame (index x spacing)
    inv = np.linalg.inv(volume.rotation)
    to_bp[:3, :3], to_bp[:3, 3] = inv, -inv @ volume.affine[:3, 3]
    ours = ours.transformed(to_bp)
    ref = read_mesh(ref_path)

    dist = surface_distance(ours, ref)
    s_ours, s_ref = mesh_stats(ours), mesh_stats(ref)
    report = {
        "vtk": vtk_version(),
        "faces": [s_ours["n_faces"], s_ref["n_faces"]],
        "volume": [s_ours["volume"], s_ref["volume"]],
        **dist.as_dict(),
    }
    record_property("comparison", json.dumps(report))
    print(json.dumps(report, indent=2))

    if vtk_version().startswith("9.4."):
        assert s_ours["n_faces"] == s_ref["n_faces"]
        assert dist.hausdorff < 1e-3
    else:
        assert s_ours["n_faces"] == pytest.approx(s_ref["n_faces"], rel=0.01)
        assert dist.mean < 0.3 and dist.hausdorff < 1.5
    assert s_ours["volume"] == pytest.approx(s_ref["volume"], rel=0.01)
