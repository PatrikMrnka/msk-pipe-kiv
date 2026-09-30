from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mskpipe.config import (
    ConfigError,
    InputSpec,
    Modality,
    PipelineConfig,
    dump_config,
    load_config,
    render_template,
)
from mskpipe.config.loader import parse_override


def test_defaults_are_valid():
    cfg = load_config()
    assert cfg == PipelineConfig()
    assert cfg.runtime.device == "cpu"
    assert cfg.segmentation.totalsegmentator.higher_order_resampling


def test_partial_override_keeps_class_specific_defaults():
    cfg = load_config(overrides=["mesh.bones.smooth_iterations=5"])
    assert cfg.mesh.bones.smooth_iterations == 5
    assert cfg.mesh.bones.passband == PipelineConfig().mesh.bones.passband


def test_unknown_key_reports_dotted_path(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("mesh:\n  bones:\n    smothing: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"mesh\.bones\.smothing"):
        load_config(path)


def test_empty_file_gives_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == PipelineConfig()


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("runtime.device=gpu", {"runtime": {"device": "gpu"}}),
        ("runtime.threads=8", {"runtime": {"threads": 8}}),
        ("runtime.cache=false", {"runtime": {"cache": False}}),
        ("runtime.threads=", {"runtime": {"threads": None}}),
        (
            "export.muscles=[iliacus_r, sartorius_r]",
            {"export": {"muscles": ["iliacus_r", "sartorius_r"]}},
        ),
    ],
)
def test_parse_override(item, expected):
    assert parse_override(item) == expected


@pytest.mark.parametrize("item", ["noequals", "=1", "a..b=1", ".a=1"])
def test_parse_override_rejects_malformed(item):
    with pytest.raises(ConfigError):
        parse_override(item)


def test_override_wins_over_file(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("runtime:\n  device: gpu\n", encoding="utf-8")
    assert load_config(path).runtime.device == "gpu"
    assert load_config(path, ["runtime.device=cpu"]).runtime.device == "cpu"


def test_out_of_range_value():
    with pytest.raises(ConfigError, match=r"mesh\.bones\.target_reduction"):
        load_config(overrides=["mesh.bones.target_reduction=1.0"])


def test_export_muscles_must_match_skeleton_side():
    with pytest.raises(ConfigError, match="iliacus_l"):
        load_config(overrides=["skeleton.side=r", "export.muscles=[iliacus_l]"])


def test_template_is_valid_and_equals_defaults():
    assert PipelineConfig.model_validate(yaml.safe_load(render_template())) == PipelineConfig()


def test_dump_roundtrip(tmp_path):
    cfg = load_config(overrides=["runtime.device=auto", "attachments.params={k: 3}"])
    path = tmp_path / "config.resolved.yaml"
    dump_config(cfg, path)
    assert load_config(path) == cfg


def test_fingerprint_depends_only_on_selected_sections():
    base = PipelineConfig()
    other_runtime = load_config(overrides=["runtime.threads=4"])
    other_mesh = load_config(overrides=["mesh.muscles.target_reduction=0.5"])
    assert base.fingerprint("mesh") == other_runtime.fingerprint("mesh")
    assert base.fingerprint("mesh") != other_mesh.fingerprint("mesh")
    assert base.fingerprint("skeleton") == other_mesh.fingerprint("skeleton")
    with pytest.raises(ValueError):
        base.fingerprint("nonexistent")


def test_input_spec_derives_subject_id():
    spec = InputSpec(image=Path("data/LHDL CT.nii.gz"), modality="ct")
    assert spec.subject_id == "LHDL_CT"
    assert spec.modality is Modality.CT


def test_input_spec_rejects_non_nifti():
    with pytest.raises(ValidationError):
        InputSpec(image=Path("scan.nrrd"), modality="ct")


def test_defaults_match_bp_config():
    cfg = PipelineConfig()
    assert (cfg.labelmap.bones.min_voxels, cfg.labelmap.muscles.min_voxels) == (2000, 1000)
    assert cfg.labelmap.muscles.bone_subtraction_dilation_radius == 0
    assert (cfg.mesh.bones.smooth_iterations, cfg.mesh.muscles.smooth_iterations) == (40, 30)
    assert cfg.mesh.bones.target_reduction == cfg.mesh.muscles.target_reduction == 0.8
    assert cfg.mesh.bones.passband == cfg.mesh.muscles.passband == 0.01
    assert cfg.attachments.method == "atlas_based"
    assert cfg.attachments.params == {"threshold_mm": 5.0, "clip_fraction": 0.22}
