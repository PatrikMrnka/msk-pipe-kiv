# SPDX-License-Identifier: Apache-2.0
"""TotalSegmentator and MuscleMap wrappers without running the tools."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mskpipe.config import load_config
from mskpipe.labelmap.scheme import load_scheme
from mskpipe.plugins.base import SegmentationRequest, TaskRequest
from mskpipe.plugins.segmenters import musclemap as mm
from mskpipe.plugins.segmenters import totalsegmentator as ts

# ---------------------------------------------------------------------- TotalSegmentator


@pytest.mark.parametrize(
    ("tasks", "modality", "expected"),
    [
        (("total", "total_mr"), "ct", "total"),
        (("total", "total_mr"), "mri", "total_mr"),
        (("appendicular_bones", "appendicular_bones_mr"), "mri", "appendicular_bones_mr"),
        ((), "ct", "total"),
        ((), "mri", "total_mr"),
    ],
)
def test_choose_task(tasks, modality, expected):
    assert ts.choose_task(tasks, modality) == expected


def test_choose_task_without_mri_variant():
    with pytest.raises(ts.TotalSegmentatorError, match="modality 'mri'"):
        ts.choose_task(("appendicular_bones",), "mri")


def test_plan_runs_merges_jobs_of_one_task():
    request = SegmentationRequest(
        "ct",
        (
            TaskRequest(("total", "total_mr"), frozenset({"hip_left", "femur_right"})),
            TaskRequest(("total",), frozenset({"gluteus_maximus_right"})),
            TaskRequest(("appendicular_bones",), frozenset({"tibia"})),
        ),
    )
    assert ts.plan_runs(request) == {
        "total": frozenset({"hip_left", "femur_right", "gluteus_maximus_right"}),
        "appendicular_bones": frozenset({"tibia"}),
    }


def test_build_command_defaults(tmp_path):
    cfg = load_config().segmentation.totalsegmentator
    argv = ts.build_command(
        tmp_path / "in.nii.gz", tmp_path / "out.nii.gz", "total", {"hip_right", "femur_left"},
        cfg, device="cpu",
    )  # fmt: skip
    assert argv[:3] == [sys.executable, "-m", ts.MODULE]
    assert "--ml" in argv
    assert argv[argv.index("-ta") + 1] == "total"
    assert argv[argv.index("-d") + 1] == "cpu"
    i = argv.index("-rs")
    assert argv[i + 1 : i + 3] == ["femur_left", "hip_right"]
    assert "--higher_order_resampling" in argv
    assert "-f" not in argv


def test_build_command_options(tmp_path):
    cfg = load_config(
        overrides=[
            "segmentation.totalsegmentator.fast=true",
            "segmentation.totalsegmentator.higher_order_resampling=false",
            "segmentation.totalsegmentator.roi_subset=false",
            'segmentation.totalsegmentator.extra_args=["--robust_crop"]',
        ]
    ).segmentation.totalsegmentator
    argv = ts.build_command(Path("i"), Path("o"), "total_mr", {"hip_left"}, cfg, device="cuda")
    assert argv[argv.index("-d") + 1] == "gpu"
    assert "-rs" not in argv and "-f" in argv
    assert "--higher_order_resampling" not in argv
    assert argv[-1] == "--robust_crop"


def test_build_command_appendicular_ignores_roi_and_fast(tmp_path):
    cfg = load_config(overrides=["segmentation.totalsegmentator.fast=true"])
    argv = ts.build_command(
        Path("i"), Path("o"), "appendicular_bones", {"tibia"}, cfg.segmentation.totalsegmentator,
        device="cpu",
    )  # fmt: skip
    assert "-rs" not in argv and "-f" not in argv


CARET = (
    '<?xml version="1.0" encoding="UTF-8"?> <CaretExtension>  <VolumeInformation Index="0">'
    "<LabelTable>"
    '<Label Key="77" Red="1" Green="0" Blue="0" Alpha="1"><![CDATA[hip_left]]></Label>\n'
    '<Label Key="78" Red="1" Green="0" Blue="0" Alpha="1"><![CDATA[hip_right]]></Label>\n'
    "</LabelTable></VolumeInformation></CaretExtension>\n              "
)


def test_parse_label_xml():
    assert ts.parse_label_xml(CARET) == {"hip_left": 77, "hip_right": 78}


def test_parse_label_xml_invalid():
    with pytest.raises(ts.TotalSegmentatorError):
        ts.parse_label_xml("<CaretExtension>")
    with pytest.raises(ts.TotalSegmentatorError, match="no labels"):
        ts.parse_label_xml("<CaretExtension/>")


def test_read_label_map_from_nifti_extension(tmp_path):
    img = nib.Nifti1Image(np.zeros((2, 2, 2), np.uint8), np.eye(4))
    img.header.extensions.append(nib.nifti1.Nifti1Extension(0, CARET.encode("utf-8")))
    path = tmp_path / "ts.nii.gz"
    nib.save(img, path)
    assert ts.read_label_map(path) == {"hip_left": 77, "hip_right": 78}

    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.uint8), np.eye(4)), path)
    with pytest.raises(ts.TotalSegmentatorError, match="no label table"):
        ts.read_label_map(path)


# ---------------------------------------------------------------------- MuscleMap


@pytest.mark.parametrize(
    ("anatomy", "side", "expected"),
    [
        ("gluteus maximus", "right", "gluteus_maximus_r"),
        ("tensor fasciae latae", "left", "tensor_fasciae_latae_l"),
        ("biceps femoris long head", "right", "biceps_femoris_long_head_r"),
        ("Gemelli and Quadratus femoris", "Left", "gemelli_and_quadratus_femoris_l"),
        ("sacrum", "no side", "sacrum"),
    ],
)
def test_label_name(anatomy, side, expected):
    assert mm.label_name(anatomy, side) == expected


def _model_json(labels):
    return {"labels": [dict(zip(("anatomy", "side", "value"), x, strict=True)) for x in labels]}


def test_parse_model_labels():
    cfg = _model_json([("ilium", "left", 6151), ("ilium", "right", 6152), ("sacrum", "", 6160)])
    assert mm.parse_model_labels(cfg) == {"ilium_l": 6151, "ilium_r": 6152, "sacrum": 6160}
    with pytest.raises(mm.MuscleMapError, match="two values"):
        mm.parse_model_labels(_model_json([("femur", "left", 1), ("femur", "left", 2)]))
    with pytest.raises(mm.MuscleMapError, match="no labels"):
        mm.parse_model_labels({"labels": []})


def test_scheme_names_match_musclemap_1_4():
    """Every MuscleMap name in the scheme exists in model 1.4 (names checked on the model)."""
    v14 = {
        "gluteus_minimus", "gluteus_medius", "gluteus_maximus", "tensor_fasciae_latae", "iliacus",
        "ilium", "femur", "piriformis", "pectineus", "obturator_internus", "obturator_externus",
        "vastus_lateralis", "vastus_intermedius", "vastus_medialis", "rectus_femoris",
        "sartorius", "gracilis", "semimembranosus", "semitendinosus", "biceps_femoris_long_head",
        "biceps_femoris_short_head", "adductor_magnus", "adductor_longus", "adductor_brevis",
        "tibia", "fibula",
    }  # fmt: skip
    names = {f"{b}_{s}" for b in v14 for s in "lr"}
    used = {n for labels in load_scheme().sources["musclemap"].labels.values() for n in labels}
    assert used <= names, sorted(used - names)


def test_read_model_labels_from_cache(tmp_path):
    path = mm.model_json_path("1.4", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_model_json([("femur", "right", 6172)])), encoding="utf-8")
    assert mm.read_model_labels("1.4", tmp_path) == {"femur_r": 6172}
    with pytest.raises(mm.MuscleMapError, match="first run"):
        mm.read_model_labels("9.9", tmp_path)


def test_model_version_pinned_and_overridable():
    assert mm.model_version(load_config()) == mm.MODEL_VERSION
    cfg = load_config(overrides=["segmentation.musclemap.model_version='1.3'"])
    assert mm.model_version(cfg) == "1.3"
    assert mm.MuscleMap.cache_identity(cfg)["model_version"] == "1.3"


@pytest.mark.parametrize(
    ("dtype", "copy"),
    [(np.uint8, True), (np.int8, True), (np.int16, False), (np.uint16, False), (np.float32, False)],
)
def test_needs_float_copy(dtype, copy):
    assert mm.needs_float_copy(np.dtype(dtype)) is copy


def test_prepare_input_converts_8bit(tmp_path):
    data = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    src = tmp_path / "mri.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([0.8, 0.8, 3, 1])), src)
    out = mm.prepare_input(src, tmp_path)
    assert out != src
    img = nib.load(out)
    assert img.get_data_dtype() == np.float32
    np.testing.assert_array_equal(img.get_fdata(), data)
    np.testing.assert_allclose(img.affine, np.diag([0.8, 0.8, 3, 1]))

    ct = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(data.astype(np.int16), np.eye(4)), ct)
    assert mm.prepare_input(ct, tmp_path) == ct


def test_normalize_output(tmp_path):
    data = np.zeros((4, 4, 130), np.int16)
    data[0, 0, 0], data[1, 1, 129] = 6172, 7182
    src = tmp_path / "x_dseg.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), src)
    dst = tmp_path / "mm.nii.gz"
    mm.normalize_output(src, dst, {6172, 7182, 8161})
    out = nib.load(dst)
    assert out.get_data_dtype() == np.uint16
    np.testing.assert_array_equal(np.asanyarray(out.dataobj), data)


def test_normalize_output_detects_scaled_labels(tmp_path):
    """An 8-bit header makes nibabel scale the IDs (MuscleMap saves with the input header)."""
    header = nib.Nifti1Header()
    header.set_data_dtype(np.uint8)
    data = np.zeros((4, 4, 4), np.int16)
    data[0, 0, 0], data[1, 0, 0] = 7222, 6102
    src = tmp_path / "x_dseg.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4), header), src)
    with pytest.raises(mm.MuscleMapError, match=r"non-integer|not labels"):
        mm.normalize_output(src, tmp_path / "mm.nii.gz", {6102, 7222})


def test_mm_build_command_and_output_path(tmp_path):
    argv = mm.build_command(
        tmp_path / "input.nii.gz", tmp_path, version="1.4", device="cpu", overlap=90.0,
        chunk_size="auto", extra_args=[],
    )  # fmt: skip
    assert argv[argv.index("--model_version") + 1] == "1.4"
    assert argv[argv.index("-g") + 1] == "N"
    assert argv[argv.index("-s") + 1] == "90"
    assert argv[argv.index("-r") + 1] == "wholebody"
    assert mm.output_path(tmp_path / "input.nii.gz", tmp_path).name == "input_dseg.nii.gz"
    assert mm.output_path(tmp_path / "a.nii", tmp_path).name == "a_dseg.nii.gz"


# ---------------------------------------------------------------------- scheme requests


def test_requests_default_config():
    req = load_scheme().requests(load_config(), "ct")
    assert set(req) == {"totalsegmentator", "musclemap"}
    jobs = {job.tasks: job.labels for job in req["totalsegmentator"].jobs}
    assert jobs[("total", "total_mr")] == {
        "hip_left",
        "hip_right",
        "sacrum",
        "vertebrae_S1",
        "femur_left",
        "femur_right",
    }
    assert jobs[("appendicular_bones", "appendicular_bones_mr")] == {"tibia", "fibula"}
    assert ts.plan_runs(req["totalsegmentator"]).keys() == {"total", "appendicular_bones"}
    mm_labels = req["musclemap"].labels
    assert "biceps_femoris_long_head_r" in mm_labels
    assert not {"femur_r", "tibia_r", "fibula_r"} & mm_labels  # bones from TotalSegmentator


def test_requests_tibia_from_musclemap():
    cfg = load_config(overrides=["segmentation.tibia_fibula=musclemap"])
    req = load_scheme().requests(cfg, "ct")
    assert len(req["totalsegmentator"].jobs) == 1
    assert {"tibia_l", "tibia_r", "fibula_r", "fibula_l"} <= req["musclemap"].labels


def test_requests_appendicular_and_ts_muscles():
    cfg = load_config(
        overrides=[
            "segmentation.tibia_fibula=ts_appendicular",
            "segmentation.muscles=totalsegmentator",
        ]
    )
    req = load_scheme().requests(cfg, "mri")
    assert set(req) == {"totalsegmentator"}
    plan = ts.plan_runs(req["totalsegmentator"])
    assert plan["appendicular_bones_mr"] == {"tibia", "fibula"}
    assert "gluteus_medius_left" in plan["total_mr"]


def test_split_labels_drops_names_the_task_lacks(monkeypatch):
    known = {
        "total": frozenset({"sacrum", "vertebrae_S1", "hip_left"}),
        "total_mr": frozenset({"sacrum", "hip_left"}),
    }
    monkeypatch.setattr(ts, "known_labels", known.get)
    wanted = {"sacrum", "vertebrae_S1", "hip_left"}
    assert ts.split_labels("total", wanted) == (frozenset(wanted), frozenset())
    assert ts.split_labels("total_mr", wanted) == (
        frozenset({"sacrum", "hip_left"}),
        frozenset({"vertebrae_S1"}),
    )


def test_split_labels_without_totalsegmentator_keeps_all(monkeypatch):
    monkeypatch.setattr(ts, "known_labels", lambda task: None)
    assert ts.split_labels("total_mr", {"vertebrae_S1"}) == (
        frozenset({"vertebrae_S1"}),
        frozenset(),
    )
