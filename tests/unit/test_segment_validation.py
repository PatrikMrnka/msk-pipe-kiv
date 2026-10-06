# SPDX-License-Identifier: Apache-2.0
"""Validation of raw segmentations (mskpipe.validation.segment, tools/segment_compare.py)."""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mskpipe.config import InputSpec, load_config
from mskpipe.core.workspace import Workspace
from mskpipe.io.segmentation import SEGMENTATION_FILE, SegmentationIndex, ToolOutput
from mskpipe.labelmap.scheme import load_scheme
from mskpipe.validation.segment import (
    RunSegmentation,
    SegmentationValidationError,
    compare_reference,
    format_report,
    sanity_check,
)

SHAPE = (40, 24, 64)
RAS = np.eye(4)  # +x = patient's right = increasing i
TS_LABELS = {
    "femur_left": 75,
    "femur_right": 76,
    "hip_left": 77,
    "hip_right": 78,
    "sacrum": 25,
    "vertebrae_S1": 26,
}
APP_LABELS = {"tibia": 9, "fibula": 10}
ROOT = Path(__file__).resolve().parents[2]


def _x(side: str, width: int = 6) -> slice:
    return slice(28, 28 + width) if side == "r" else slice(6, 6 + width)


def _ts_volume() -> np.ndarray:
    data = np.zeros(SHAPE, np.uint8)
    for side, word in (("r", "right"), ("l", "left")):
        data[_x(side, 8), 4:16, 48:58] = TS_LABELS[f"hip_{word}"]
        data[_x(side), 6:14, 26:44] = TS_LABELS[f"femur_{word}"]
    data[18:22, 10:16, 50:56] = TS_LABELS["sacrum"]  # midline, between the hip bones
    data[18:22, 10:16, 56:60] = TS_LABELS["vertebrae_S1"]
    return data


def _appendicular_volume() -> np.ndarray:
    """TotalSegmentator appendicular_bones: one tibia and one fibula label for both legs."""
    data = np.zeros(SHAPE, np.uint8)
    for side in ("r", "l"):
        data[_x(side, 4), 6:10, 4:20] = APP_LABELS["tibia"]
        data[_x(side, 4), 12:15, 4:20] = APP_LABELS["fibula"]
    return data


def _mm_volume() -> tuple[np.ndarray, dict[str, int]]:
    """MuscleMap-like output: every label of the scheme's musclemap source, small blocks."""
    scheme = load_scheme()
    names = sorted({n for labels in scheme.sources["musclemap"].labels.values() for n in labels})
    data = np.zeros(SHAPE, np.uint16)
    labels: dict[str, int] = {}
    slot = {"r": 0, "l": 0}
    for i, name in enumerate(names, start=1):
        value = 7000 + i
        labels[name] = value
        side = name[-1]
        if name.startswith(("tibia", "fibula")):
            y = slice(6, 10) if name.startswith("tibia") else slice(12, 15)
            data[_x(side, 4), y, 4:20] = value
        elif name.startswith(("ilium", "femur")):
            continue  # bones come from TotalSegmentator in the default config
        else:
            k = slot[side]
            slot[side] += 1
            y, z = 18 + (k % 3) * 2, 2 + (k // 3) * 3
            data[_x(side, 4), y : y + 2, z : z + 2] = value
    return data, labels


def _write_run(seg_dir: Path, affine: np.ndarray = RAS, ts_labels: dict | None = None) -> None:
    seg_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(_ts_volume(), affine), seg_dir / "totalsegmentator_total.nii.gz")
    app_file = "totalsegmentator_appendicular_bones.nii.gz"
    nib.save(nib.Nifti1Image(_appendicular_volume(), affine), seg_dir / app_file)
    mm, mm_labels = _mm_volume()
    nib.save(nib.Nifti1Image(mm, affine), seg_dir / "musclemap_wholebody.nii.gz")
    SegmentationIndex(
        outputs=(
            ToolOutput(
                tool="totalsegmentator",
                task="total",
                file="totalsegmentator_total.nii.gz",
                labels=ts_labels or TS_LABELS,
            ),
            ToolOutput(
                tool="totalsegmentator",
                task="appendicular_bones",
                file=app_file,
                labels=APP_LABELS,
            ),
            ToolOutput(
                tool="musclemap",
                task="wholebody",
                file="musclemap_wholebody.nii.gz",
                labels=mm_labels,
            ),
        )
    ).save(seg_dir / SEGMENTATION_FILE)


@pytest.fixture
def run(tmp_path: Path) -> Path:
    _write_run(tmp_path / "01_segment")
    return tmp_path / "01_segment"


def _sanity(seg_dir: Path) -> dict:
    return sanity_check(RunSegmentation(seg_dir), load_config(), load_scheme())


def test_sanity_ok(run: Path) -> None:
    report = _sanity(run)
    assert report["problems"] == []
    assert report["side_dx_mm"]["femur"] > 0
    assert report["side_dx_mm"]["gluteus_maximus"] > 0
    assert all(r["status"] == "ok" for r in report["structures"].values())
    assert "Sanity checks: OK" in format_report(report)


def test_sanity_tibia_from_musclemap(run: Path) -> None:
    config = load_config(overrides=["segmentation.tibia_fibula=musclemap"])
    report = sanity_check(RunSegmentation(run), config, load_scheme())
    assert report["problems"] == []
    assert report["structures"]["tibia_r"]["tool"] == "musclemap"
    assert report["side_dx_mm"]["tibia"] > 0


def test_swapped_label_ids_are_detected(tmp_path: Path) -> None:
    swapped = dict(TS_LABELS, femur_left=76, femur_right=75)
    _write_run(tmp_path, ts_labels=swapped)
    problems = _sanity(tmp_path)["problems"]
    assert any(p.startswith("femur_r is not right of femur_l") for p in problems)


def test_flipped_grid_is_detected(tmp_path: Path) -> None:
    _write_run(tmp_path, affine=np.diag([1.0, 1.0, -1.0, 1.0]))
    problems = _sanity(tmp_path)["problems"]
    assert any("pelvis > femur > tibia" in p for p in problems)


def test_mirrored_grid_is_detected(tmp_path: Path) -> None:
    _write_run(tmp_path, affine=np.diag([-1.0, 1.0, 1.0, 1.0]))
    problems = _sanity(tmp_path)["problems"]
    assert any("is not right of" in p for p in problems)


def test_missing_structure(tmp_path: Path) -> None:
    _write_run(tmp_path)
    img = nib.load(tmp_path / "musclemap_wholebody.nii.gz")
    data = np.asarray(img.dataobj)
    index = SegmentationIndex.load(tmp_path / SEGMENTATION_FILE)
    value = index.outputs[2].labels["gracilis_l"]
    data[data == value] = 0
    nib.save(nib.Nifti1Image(data, img.affine), tmp_path / "musclemap_wholebody.nii.gz")
    report = _sanity(tmp_path)
    assert report["structures"]["gracilis_l"]["status"] == "missing"
    assert any(p.startswith("gracilis_l:") for p in report["problems"])


def _reference(run: Path, ref: Path, shift: int = 0) -> dict:
    """BP-like reference masks cut from the run itself (optionally shifted along x)."""
    ref.mkdir()
    ts = nib.load(run / "totalsegmentator_total.nii.gz")
    mm = nib.load(run / "musclemap_wholebody.nii.gz")
    app = nib.load(run / "totalsegmentator_appendicular_bones.nii.gz")
    index = SegmentationIndex.load(run / SEGMENTATION_FILE)
    mm_labels = index.outputs[2].labels
    ts_data, mm_data = np.asarray(ts.dataobj), np.asarray(mm.dataobj)
    masks = {
        "hip_left": ts_data == TS_LABELS["hip_left"],
        "femur_right": ts_data == TS_LABELS["femur_right"],
        "tibia": np.asarray(app.dataobj) == APP_LABELS["tibia"],
        "gluteus_maximus_r": mm_data == mm_labels["gluteus_maximus_r"],
        "biceps_femoris_r": mm_data == mm_labels["biceps_femoris_long_head_r"],
    }
    for name, mask in masks.items():
        mask = np.roll(mask, shift, axis=0).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, RAS), ref / f"{name}.nii.gz")
    return compare_reference(RunSegmentation(run), ref, load_scheme(), load_config())


def test_reference_identical(run: Path, tmp_path: Path) -> None:
    report = _reference(run, tmp_path / "ref")
    rows = report["structures"]
    assert set(rows) == {
        "hip_left",
        "femur_right",
        "tibia",
        "gluteus_maximus_r",
        "biceps_femoris_r",
    }
    for r in rows.values():
        assert r["dice"] == 1.0
        assert r["assd_mm"] == 0.0
        assert r["centroid_shift_mm"] == 0.0
    assert rows["tibia"]["tool"] == "totalsegmentator"  # default source, as BP
    assert rows["tibia"]["task"] == "appendicular_bones"
    assert rows["biceps_femoris_r"]["labels"] == ["biceps_femoris_long_head_r"]
    assert report["unmatched"] == [] and report["affine_matches"]


def test_reference_shifted(run: Path, tmp_path: Path) -> None:
    rows = _reference(run, tmp_path / "ref", shift=1)["structures"]
    assert rows["femur_right"]["centroid_shift_mm"] == pytest.approx(1.0)
    assert rows["femur_right"]["dice"] < 1.0
    assert 0 < rows["femur_right"]["hd95_mm"] <= 1.0


def test_reference_on_other_grid(run: Path, tmp_path: Path) -> None:
    ref = tmp_path / "ref"
    ref.mkdir()
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), np.uint8), RAS), ref / "femur_right.nii.gz")
    with pytest.raises(SegmentationValidationError, match="input grid"):
        compare_reference(RunSegmentation(run), ref, load_scheme(), load_config())


def test_tool_exit_codes(tmp_path: Path) -> None:
    image = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(SHAPE, np.int16), RAS), image)
    config = load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}"])
    ws = Workspace.create(InputSpec(image=image, modality="ct"), config)
    _write_run(ws.step_dir("segment"))

    spec = importlib.util.spec_from_file_location(
        "segment_compare", ROOT / "tools" / "segment_compare.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    out = tmp_path / "report.json"
    assert tool.main([str(ws.root), "--json", str(out)]) == 0
    assert out.is_file()

    _write_run(ws.step_dir("segment"), ts_labels=dict(TS_LABELS, femur_left=76, femur_right=75))
    assert tool.main([str(ws.root)]) == 1
