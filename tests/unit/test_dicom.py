# SPDX-License-Identifier: Apache-2.0
"""DICOM input: conversion and checks (real dcm2niix), API, run folder, CLI and batch."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from dicom_data import PATIENT, ct_folder, write_series
from fake_pipeline import write_nifti
from typer.testing import CliRunner

from mskpipe import api
from mskpipe.batch import expand, load_batch
from mskpipe.cli import app
from mskpipe.config import InputSpec, load_config
from mskpipe.core.workspace import Workspace
from mskpipe.io import dicom
from mskpipe.io.dicom import DicomError

# ---------------------------------------------------------------------------- scan


def test_single_series_is_converted_in_hounsfield_units(tmp_path):
    folder = write_series(tmp_path / "ct", "CT")
    result = dicom.scan(folder)
    vol = result.select()
    assert (vol.id, vol.modality, vol.files, vol.shape[:3]) == ("3", "ct", 12, (20, 16, 12))
    assert vol.spacing == pytest.approx((0.7, 0.8, 2.0))
    data = np.asanyarray(nib.load(vol.nifti).dataobj)
    assert data.min() == -1024 and data.max() == 976  # rescale intercept applied
    assert vol.usable and vol.label().startswith("Series 3 'Body': CT, 20 x 16 x 12 voxels")


def test_sidecar_has_modality_but_no_patient_data(tmp_path):
    vol = dicom.scan(write_series(tmp_path / "ct", "CT")).select()
    meta = json.loads(Path(vol.sidecar).read_text(encoding="utf-8"))
    assert meta["Modality"] == "CT"
    text = json.dumps(meta)
    assert PATIENT.split("^")[0] not in text and "ID-0042" not in text and "1970" not in text


def test_localizer_is_not_usable(tmp_path):
    result = dicom.scan(ct_folder(tmp_path / "ct"))
    scout = result.get("1")
    assert not scout.usable and "localizer" in scout.status()
    assert result.select().id == "3"
    assert "not usable: localizer" in result.describe()


def test_several_series_need_a_choice(tmp_path):
    folder = ct_folder(tmp_path / "ct")
    write_series(folder, "CT", number=5, slices=8, description="Lung")
    result = dicom.scan(folder)
    assert [v.id for v in result.usable] == ["3", "5"]
    with pytest.raises(DicomError, match=r"2 usable series .* choose one \(3, 5\)"):
        result.select()
    assert result.select("5").shape[2] == 8
    with pytest.raises(DicomError, match="No series 9"):
        result.select("9")
    with pytest.raises(DicomError, match="Series 1 cannot be used: localizer"):
        result.select("1")


def test_mr_and_other_modalities(tmp_path):
    assert dicom.scan(write_series(tmp_path / "mr", "MR", number=7)).select().modality == "mri"
    pet = dicom.scan(write_series(tmp_path / "pet", "PT", number=2))
    assert "only CT and MR" in pet.get("2").status()
    with pytest.raises(DicomError, match="No usable 3D CT or MR series"):
        pet.select()


def test_missing_slice_is_resampled(tmp_path):
    result = dicom.scan(write_series(tmp_path / "gap", "CT", number=2, skip=5))
    assert result.get("2").replaced_by == "2_Eq_1"
    vol = result.select()
    assert vol.id == "2_Eq_1" and vol.shape[2] == 12
    assert any("equidistant" in w for w in vol.warnings)
    assert result.select("2").id == "2_Eq_1"  # the uncorrected volume is never used


def test_missing_slice_that_cannot_be_corrected(tmp_path):
    result = dicom.scan(write_series(tmp_path / "gap", "CT", number=2, skip=5, signed=False))
    with pytest.raises(DicomError, match="irregular slice spacing"):
        result.select()


def test_gantry_tilt_is_corrected(tmp_path):
    result = dicom.scan(write_series(tmp_path / "tilt", "CT", number=4, tilt_mm=0.5))
    vol = result.select()
    assert vol.id == "4_Tilt_1" and "gantry tilt corrected" in vol.warnings
    assert result.get("4").replaced_by == "4_Tilt_1"


def test_not_dicom(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DicomError, match="empty"):
        dicom.scan(tmp_path / "empty")
    (tmp_path / "text").mkdir()
    (tmp_path / "text" / "notes.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(DicomError, match="No DICOM images"):
        dicom.scan(tmp_path / "text")
    with pytest.raises(DicomError, match="not found"):
        dicom.scan(tmp_path / "missing")


def test_conversion_is_cached(tmp_path, monkeypatch):
    folder = write_series(tmp_path / "ct", "CT")
    first = dicom.scan(folder)
    monkeypatch.setattr(dicom, "_run", lambda *a: pytest.fail("converted again"))
    assert dicom.scan(folder) == first
    monkeypatch.undo()
    write_series(folder, "CT", number=5, slices=8)  # the folder changed: converted again
    assert [v.id for v in dicom.scan(folder).usable] == ["3", "5"]


def test_cache_keeps_the_last_conversions(tmp_path):
    for i in range(dicom.CACHE_KEEP + 2):
        dicom.scan(write_series(tmp_path / f"ct{i}", "CT", slices=3))
    assert len(list(dicom.cache_root().iterdir())) == dicom.CACHE_KEEP


def test_missing_dcm2niix(tmp_path, monkeypatch):
    monkeypatch.setattr(dicom, "dcm2niix_executable", lambda: None)
    with pytest.raises(DicomError, match="dcm2niix was not found"):
        dicom.scan(write_series(tmp_path / "ct", "CT", slices=3))


# ---------------------------------------------------------------------------- input spec


def test_input_spec_rules(tmp_path):
    folder = write_series(tmp_path / "TLEM 2.0 - MRI - Dicom", "MR", slices=3)
    spec = InputSpec(image=folder, modality="mri")
    assert spec.is_dicom and spec.subject_id == "TLEM_2.0_-_MRI_-_Dicom"
    with pytest.raises(ValueError, match="only used with a DICOM folder"):
        InputSpec(image=tmp_path / "a.nii.gz", modality="ct", series="3")
    with pytest.raises(ValueError, match=r"NIfTI file .* or a folder of DICOM files"):
        InputSpec(image=next(folder.glob("*.dcm")), modality="mri")
    # a resumed run keeps its series even if the folder is gone
    assert InputSpec(image=tmp_path / "gone", modality="ct", series="3").is_dicom


# ---------------------------------------------------------------------------- api


def request(image, tmp_path, **kw) -> api.RunRequest:
    return api.RunRequest(image=image, runs_dir=tmp_path / "runs", **{"modality": "auto", **kw})


def test_prepare_dicom_with_auto_modality(tmp_path):
    folder = ct_folder(tmp_path / "ct")
    prepared = api.prepare_run(request(folder, tmp_path))
    assert prepared.spec.modality == "ct" and prepared.spec.series == "3"
    assert prepared.image.name == "3.nii.gz" and prepared.image.is_file()
    assert any(n.startswith("DICOM: Series 3") for n in prepared.notes)
    assert any("Modality: CT (dicom" in n for n in prepared.notes)
    assert api.check_image(prepared.image) == []


def test_prepare_dicom_errors(tmp_path):
    folder = ct_folder(tmp_path / "ct")
    write_series(folder, "CT", number=5, slices=8)
    with pytest.raises(api.SetupError, match="choose one"):
        api.prepare_run(request(folder, tmp_path))
    assert api.prepare_run(request(folder, tmp_path, series="5")).spec.series == "5"
    with pytest.raises(api.SetupError, match="No DICOM images"):
        (tmp_path / "text").mkdir()
        (tmp_path / "text" / "a.txt").write_text("x", encoding="utf-8")
        api.prepare_run(request(tmp_path / "text", tmp_path))
    with pytest.raises(api.SetupError, match="series"):
        nifti = write_nifti(tmp_path / "s.nii.gz")
        api.prepare_run(request(nifti, tmp_path, modality="ct", series="3"))
    with pytest.raises(api.SetupError, match="keeps its input"):
        api.RunRequest(resume=tmp_path, series="3").check()


def test_prepare_dicom_file_means_its_folder(tmp_path):
    folder = write_series(tmp_path / "ct", "CT")
    prepared = api.prepare_run(request(next(folder.glob("*.dcm")), tmp_path))
    assert prepared.spec.image == folder
    assert any("using its folder" in n for n in prepared.notes)


def test_prepare_rejects_invalid_nifti(tmp_path):
    single = tmp_path / "slice.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 1), np.int16), np.eye(4)), str(single))
    with pytest.raises(api.SetupError, match="single slice"):
        api.prepare_run(request(single, tmp_path, modality="ct"))
    broken = tmp_path / "broken.nii.gz"
    broken.write_bytes(b"no nifti")
    with pytest.raises(api.SetupError, match="Cannot read NIfTI"):
        api.prepare_run(request(broken, tmp_path, modality="ct"))
    flat = tmp_path / "flat.nii.gz"
    affine = np.eye(4)
    affine[0, 3] = np.nan
    img = nib.Nifti1Image(np.zeros((8, 8, 4), np.int16), np.eye(4))
    img.set_qform(None, code=0)
    img.set_sform(affine, code=1)
    nib.save(img, str(flat))
    with pytest.raises(api.SetupError, match="Invalid geometry"):
        api.prepare_run(request(flat, tmp_path, modality="ct"))


def test_input_volume(tmp_path):
    folder = write_series(tmp_path / "ct", "CT")
    assert api.input_volume(folder).name == "3.nii.gz"
    nifti = write_nifti(tmp_path / "s.nii.gz")
    assert api.input_volume(nifti) == nifti
    with pytest.raises(api.SetupError, match="not found"):
        api.input_volume(tmp_path / "missing.nii.gz")


# ---------------------------------------------------------------------------- run folder


def test_workspace_from_dicom(tmp_path):
    folder = ct_folder(tmp_path / "ct")
    config = load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}"])
    spec = InputSpec(image=folder, modality="ct")
    ws = Workspace.create(spec, config)
    record = ws.load_input_record()
    assert record["image"] == str(folder.resolve())
    source = record["source"]
    assert (source["format"], source["series"], source["files"]) == ("dicom", "3", 12)
    assert source["converter"].startswith("dcm2niix v")
    assert ws.input_image.name == "input.nii.gz"
    assert json.loads((ws.input_image.parent / "input.json").read_text())["Modality"] == "CT"
    assert ws.load_input().series == "3"
    # the same DICOM content gives the same key (results are reused)
    again = Workspace.create(InputSpec(image=folder, modality="ct", series="3"), config)
    assert again.run_key == ws.run_key and again.root != ws.root


def test_run_from_dicom(tmp_path):
    from fake_pipeline import fake_steps, reset

    reset()
    folder = ct_folder(tmp_path / "ct")
    prepared = api.prepare_run(request(folder, tmp_path, device="cpu"))
    summary = api.run(prepared, steps=fake_steps())
    assert summary.status == "completed", summary
    record = json.loads((Path(summary.run_dir) / "input.json").read_text(encoding="utf-8"))
    assert record["source"]["series"] == "3" and record["modality"] == "ct"


# ---------------------------------------------------------------------------- cli, batch


def test_cli_dicom_lists_and_writes(tmp_path):
    folder = ct_folder(tmp_path / "ct")
    write_series(folder, "CT", number=5, slices=8)
    runner = CliRunner()
    out = runner.invoke(app, ["dicom", str(folder)])
    assert out.exit_code == 0, out.output
    assert "ok  Series 3 'Body'" in out.output and "not usable: localizer" in out.output
    out = runner.invoke(app, ["dicom", str(folder), "-o", str(tmp_path / "x.nii.gz")])
    assert out.exit_code == 2 and "choose one" in out.output
    out = runner.invoke(app, ["dicom", str(folder), "--series", "5", "-o", str(tmp_path / "ct5")])
    assert out.exit_code == 0, out.output
    assert nib.load(tmp_path / "ct5.nii.gz").shape == (20, 16, 8)
    assert (tmp_path / "ct5.json").is_file()
    data = json.loads(runner.invoke(app, ["dicom", str(folder), "--json"]).output)
    assert [v["id"] for v in data["volumes"]] == ["1", "3", "5"]


def test_cli_dicom_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    out = CliRunner().invoke(app, ["dicom", str(tmp_path / "empty")])
    assert out.exit_code == 2 and "empty" in out.output


def test_batch_with_dicom_folder(tmp_path):
    folder = ct_folder(tmp_path / "data" / "patient.01")
    (tmp_path / "batch.yaml").write_text(
        "jobs:\n  - image: data/patient.01\n    modality: ct\n    series: 3\n", encoding="utf-8"
    )
    batch, base = load_batch(tmp_path / "batch.yaml")
    [run] = expand(batch, base)
    assert run.job == "patient.01"  # the folder name, dots kept
    assert run.request.image == folder and run.request.series == "3"
