# SPDX-License-Identifier: Apache-2.0
"""TLEM 2.0 MRI as DICOM vs the NIfTI written by ``tlem_to_mskpipe.py`` (local data only).

    $MSKPIPE_TLEM_DIR/TLEM 2.0 - MRI/TLEM 2.0 - MRI - Dicom/*.dcm   Mimics DICOM export
    $MSKPIPE_TLEM_DIR/mskpipe/mri/mri.nii.gz                        converted reference

The voxels and the spacing must be identical. The in-plane origin differs on purpose:
the reference NIfTI takes it from the pelvis station of the raw scanner data (the frame of
the TLEM segmentations), the Mimics export has the origin of the ankle station.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

ROOT = os.environ.get("MSKPIPE_TLEM_DIR")
DICOM = Path(ROOT or ".") / "TLEM 2.0 - MRI" / "TLEM 2.0 - MRI - Dicom"
NIFTI = Path(ROOT or ".") / "mskpipe" / "mri" / "mri.nii.gz"


@pytest.fixture(scope="module")
def volumes():
    if not ROOT:
        pytest.skip("MSKPIPE_TLEM_DIR not set")
    if not DICOM.is_dir() or not NIFTI.is_file():
        pytest.skip(f"TLEM 2.0 MRI data not found in {ROOT}")
    import nibabel as nib

    from mskpipe.io.dicom import dcm2niix_executable, scan

    if dcm2niix_executable() is None:
        pytest.skip("dcm2niix not installed")
    result = scan(DICOM)
    vol = result.select()
    dcm = nib.as_closest_canonical(nib.load(vol.nifti))
    ref = nib.as_closest_canonical(nib.load(NIFTI))
    return result, vol, dcm, ref


def test_tlem_mri_dicom_is_one_mr_series(volumes):
    result, vol, _, _ = volumes
    assert [v.id for v in result.usable] == [vol.id]
    assert (vol.modality, vol.files, vol.shape[:3]) == ("mri", 350, (369, 369, 350))
    assert vol.spacing == pytest.approx((1.041, 1.041, 3.0), abs=1e-3)


def test_tlem_mri_dicom_voxels_match_the_reference(volumes):
    _, _, dcm, ref = volumes
    assert dcm.shape == ref.shape
    assert np.array_equal(np.asanyarray(dcm.dataobj), np.asanyarray(ref.dataobj))
    assert np.allclose(dcm.affine[:3, :3], ref.affine[:3, :3], atol=1e-4)
    shift = dcm.affine[:3, 3] - ref.affine[:3, 3]
    assert shift[2] == pytest.approx(0.0, abs=1e-3)  # only the in-plane origin differs
    assert np.linalg.norm(shift[:2]) == pytest.approx(39.7, abs=1.0)
