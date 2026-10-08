# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mskpipe import api
from mskpipe.io.modality import detect_modality, guess_from_intensities, sidecar_path


def save(path: Path, data: np.ndarray, *, slope_inter=None, descrip: bytes = b"") -> Path:
    img = nib.Nifti1Image(data, np.eye(4))
    if slope_inter is not None:
        img.header.set_slope_inter(*slope_inter)
    if descrip:
        img.header["descrip"] = descrip
    nib.save(img, str(path))
    return path


def ct_volume() -> np.ndarray:
    ct = np.full((40, 40, 30), -1024, np.int16)
    ct[10:30, 10:30, :] = 40  # body
    ct[18:22, 18:22, :] = 900  # bone
    return ct


def mri_volume() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 900, (40, 40, 30)).astype(np.uint16)


def test_ct_from_intensities(tmp_path):
    guess = detect_modality(save(tmp_path / "ct.nii.gz", ct_volume()))
    assert (guess.modality, guess.source) == ("ct", "intensities")
    assert "HU" in guess.reason and str(guess).startswith("CT (intensities")


def test_ct_with_rescale_intercept(tmp_path):
    raw = (ct_volume().astype(np.int32) + 1024).astype(np.uint16)  # stored 0..4095
    guess = detect_modality(save(tmp_path / "ct.nii.gz", raw, slope_inter=(1, -1024)))
    assert guess.modality == "ct"


def test_mri_from_intensities_and_header(tmp_path):
    assert detect_modality(save(tmp_path / "a.nii.gz", mri_volume())).modality == "mri"
    guess = detect_modality(save(tmp_path / "b.nii", ct_volume(), descrip=b"TE=4.6;Time=1"))
    assert (guess.modality, guess.source) == ("mri", "header")


def test_sidecar_wins(tmp_path):
    image = save(tmp_path / "scan.nii.gz", mri_volume())
    assert sidecar_path(image) == tmp_path / "scan.json"
    (tmp_path / "scan.json").write_text(json.dumps({"Modality": "CT"}), encoding="utf-8")
    assert (detect_modality(image).modality, detect_modality(image).source) == ("ct", "sidecar")
    (tmp_path / "scan.json").write_text("{broken", encoding="utf-8")
    assert detect_modality(image).source == "intensities"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([-1024.0] * 50 + [40.0] * 50), "ct"),
        (np.array([0.0, 10.0, 500.0]), "mri"),
        (np.array([-30.0, 10.0, 500.0]), "mri"),  # slightly negative after bias correction
        (np.array([-400.0, 10.0, 500.0]), None),  # neither
        (np.array([-1000.0] + [50.0] * 999), None),  # one air voxel is not a CT
        (np.array([np.nan]), None),
    ],
)
def test_intensity_rules(values, expected):
    assert guess_from_intensities(values).modality == expected


def test_prepare_run_auto(tmp_path):
    image = save(tmp_path / "s01.nii.gz", ct_volume())
    prepared = api.prepare_run(api.RunRequest(image=image, modality="auto"))
    assert prepared.modality == "ct"
    assert prepared.notes and prepared.notes[0].startswith("Modality: CT")
    odd = save(tmp_path / "odd.nii.gz", np.full((10, 10, 10), -400, np.int16))
    with pytest.raises(api.SetupError, match="Cannot detect the modality"):
        api.prepare_run(api.RunRequest(image=odd, modality="auto"))
    with pytest.raises(api.SetupError, match="ct, mri or auto"):
        api.RunRequest(image=image, modality="pet").check()
