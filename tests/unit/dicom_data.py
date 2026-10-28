# SPDX-License-Identifier: Apache-2.0
"""Small synthetic DICOM series for the tests (pydicom; converted by the real dcm2niix)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

from mskpipe.io.dicom import dcm2niix_executable  # noqa: E402

if dcm2niix_executable() is None:
    pytest.skip("dcm2niix not installed", allow_module_level=True)

SOP_CLASS = {
    "CT": "1.2.840.10008.5.1.4.1.1.2",
    "MR": "1.2.840.10008.5.1.4.1.1.4",
    "PT": "1.2.840.10008.5.1.4.1.1.128",
}
ROWS, COLS = 16, 20
PIXEL_MM = (0.8, 0.7)  # row spacing, column spacing
SLICE_MM = 2.0
PATIENT = "Doe^Jane"


def write_series(
    folder: Path,
    modality: str = "CT",
    *,
    number: int = 3,
    slices: int = 12,
    description: str = "Body",
    localizer: bool = False,
    skip: int | None = None,
    signed: bool = True,
    tilt_mm: float = 0.0,
) -> Path:
    """One series of ``slices`` axial images in ``folder`` (created).

    CT is stored with a rescale intercept of -1024 (air -1024 HU, soft tissue 40, bone 976);
    ``skip`` leaves out one slice (irregular spacing), ``tilt_mm`` shifts every slice in y
    (gantry tilt); ``signed=False`` stores unsigned pixels (dcm2niix cannot resample them).
    """
    folder.mkdir(parents=True, exist_ok=True)
    series_uid, study_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    for i in range(slices):
        if i == skip:
            continue
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SOP_CLASS[modality]
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = modality
        ds.Manufacturer = "TEST"
        ds.PatientName = PATIENT
        ds.PatientID = "ID-0042"
        ds.PatientBirthDate = "19700101"
        ds.PatientPosition = "FFS"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.StudyDate = ds.SeriesDate = "20240101"
        ds.AcquisitionTime = "120000"
        ds.SeriesNumber = number
        ds.InstanceNumber = i + 1
        ds.SeriesDescription = description
        ds.ImageType = ["ORIGINAL", "PRIMARY", "LOCALIZER" if localizer else "AXIAL"]
        ds.Rows, ds.Columns = ROWS, COLS
        ds.PixelSpacing = list(PIXEL_MM)
        ds.SliceThickness = SLICE_MM
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [-10.0, -20.0 + i * tilt_mm, i * SLICE_MM]
        if tilt_mm:
            ds.GantryDetectorTilt = float(np.degrees(np.arctan2(tilt_mm, SLICE_MM)))
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1 if signed else 0
        if modality == "CT":
            ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
            pixels = np.zeros((ROWS, COLS), np.int32)
            pixels[4:12, 4:14] = 1064
            pixels[7:9, 7:9] = 2000
        else:
            if modality == "MR":
                ds.EchoTime, ds.RepetitionTime = 10, 500
            pixels = (np.arange(ROWS * COLS).reshape(ROWS, COLS) % 300 + i).astype(np.int32)
        ds.PixelData = pixels.astype(np.int16 if signed else np.uint16).tobytes()
        ds.save_as(folder / f"S{number:03d}_{i:04d}.dcm", enforce_file_format=True)
    return folder


def ct_folder(path: Path) -> Path:
    """A typical CT export: a localizer and one CT series."""
    write_series(path, "CT", number=1, slices=1, description="Topogram", localizer=True)
    return write_series(path, "CT", number=3)


def copy_tree(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst
