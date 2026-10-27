# SPDX-License-Identifier: Apache-2.0
"""Value checks: hard bounds (errors), recommended ranges and context checks (warnings)."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import annotated_types as at
import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

from mskpipe import api
from mskpipe.cli import app
from mskpipe.config import (
    ConfigError,
    InputSpec,
    PipelineConfig,
    load_config,
    render_template,
    sanity,
)
from mskpipe.config.sanity import recommended_ranges
from mskpipe.core.registry import default_registry
from mskpipe.plugins.attachments.atlas_registration import AtlasRegistrationParams

# Numeric fields without a static recommended range, and why
NO_RECOMMENDED = {
    "runtime.threads": "depends on the CPU cores (environment check)",
    "runtime.seed": "any seed is fine",
    "segmentation.musclemap.chunk_size": "MuscleMap clamps it to the volume (2D slices)",
    "export.motion.end_deg": "depends on the coordinate (anatomical ranges in sanity)",
    "seed": "any seed is fine (attachments.params)",
}
# Numeric fields bounded by a model validator (int | 'auto' unions) instead of ge/le
VALIDATOR_BOUNDS = {
    "segmentation.totalsegmentator.nr_thr_saving",
    "segmentation.musclemap.chunk_size",
}


def _numeric_fields(model, prefix=""):
    """(dotted path, FieldInfo) of every int/float field, nested models included."""
    import types
    import typing

    from pydantic import BaseModel

    out = []
    for name, field in model.model_fields.items():
        args = typing.get_args(field.annotation) or (field.annotation,)
        if typing.get_origin(field.annotation) not in (typing.Union, types.UnionType):
            args = (field.annotation,)
        subs = [a for a in args if isinstance(a, type) and issubclass(a, BaseModel)]
        for sub in subs:
            out += _numeric_fields(sub, f"{prefix}{name}.")
        if any(a in (int, float) for a in args):
            out.append((f"{prefix}{name}", field))
    return out


def _bounds(field):
    low = next(
        (
            m.ge if isinstance(m, at.Ge) else m.gt
            for m in field.metadata
            if isinstance(m, at.Ge | at.Gt)
        ),
        None,
    )
    high = next(
        (
            m.le if isinstance(m, at.Le) else m.lt
            for m in field.metadata
            if isinstance(m, at.Le | at.Lt)
        ),
        None,
    )
    return low, high


@pytest.mark.parametrize(("model", "prefix"), [(PipelineConfig, ""), (AtlasRegistrationParams, "")])
def test_every_numeric_field_has_bounds_and_recommended_range(model, prefix):
    ranges = recommended_ranges(model)
    fields = _numeric_fields(model)
    assert fields
    for path, field in fields:
        if path == "config_version":
            continue
        low, high = _bounds(field)
        if path not in VALIDATOR_BOUNDS:
            assert low is not None and high is not None, f"{path}: no hard bounds"
        if path in NO_RECOMMENDED:
            assert path not in ranges
            continue
        assert path in ranges, f"{path}: no recommended range"
        rec = ranges[path]
        if low is not None:
            assert rec.low is None or rec.low >= low, path
        if high is not None:
            assert rec.high is None or rec.high <= high, path
        default = field.get_default(call_default_factory=True)
        if isinstance(default, int | float):
            assert rec.contains(default), f"{path}: default {default} outside {rec.text()}"


def test_defaults_raise_no_config_issues():
    assert sanity.check_config(api.resolved_config()) == []


# ---------------------------------------------------------------------------- hard bounds


@pytest.mark.parametrize(
    "override",
    [
        "runtime.seed=-1",
        f"runtime.seed={2**32}",
        "runtime.threads=4096",
        "attachments.params.seed=-1",
        f"attachments.params.seed={2**32}",
        "attachments.params.nonrigid_points=4000",  # > n_points 3000
        "attachments.params.cpd_w=0.95",
        "attachments.params.cpd_sigma_init_mm=500",
        "segmentation.musclemap.overlap=99.5",
        "labelmap.bones.opening_radius_mm=12",
        "mesh.bones.target_reduction=0.995",
        "skeleton.body_mass=5",
        "skeleton.model_name=CON",
        "export.exclude=[piriformiss]",
        "export.muscles=[gluteus_maxx_r]",
    ],
)
def test_hard_bounds(override):
    with pytest.raises(api.SetupError):
        api.resolved_config(overrides=[override])


def test_nonrigid_points_limit_only_for_its_stages():
    cfg = api.resolved_config(
        overrides=["attachments.params.nonrigid=none", "attachments.params.nonrigid_points=4000"]
    )
    assert cfg.attachments.params["nonrigid_points"] == 4000


def test_unknown_names_suggest_the_close_one():
    with pytest.raises(ConfigError, match="did you mean 'piriformis'"):
        load_config(overrides=["export.exclude=[piriformiss]"])
    with pytest.raises(ConfigError, match="did you mean 'num_of_lines'"):
        load_config(overrides=["export.num_of_line=5"])
    with pytest.raises(api.SetupError, match="did you mean 'n_points'"):
        api.resolved_config(overrides=["attachments.params.n_point=5"])


@pytest.mark.parametrize("name", ["CON", "nul", "COM1", "lpt9.scan", "s01."])
def test_subject_id_must_be_a_windows_folder_name(name, tmp_path):
    with pytest.raises(ValueError, match="Windows"):
        InputSpec(image=tmp_path / "x.nii.gz", modality="ct", subject_id=name)


def test_subject_id_from_a_reserved_file_name(tmp_path):
    spec = InputSpec(image=tmp_path / "con.nii.gz", modality="ct")
    assert spec.subject_id == "con_s"


def test_min_voxels_of_old_configs_become_ml(tmp_path):
    cfg = load_config(overrides=["labelmap.bones.min_voxels=3000"])
    assert cfg.labelmap.bones.min_volume_ml == 3.0  # 1 voxel = 1 mm^3 (BP grid)
    with pytest.raises(ConfigError, match="give only min_volume_ml"):
        load_config(overrides=["labelmap.bones.min_voxels=3000", "labelmap.bones.min_volume_ml=3"])


def test_removed_atlas_name_still_loads(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text("attachments:\n  atlas: lhdl\n", encoding="utf-8")
    assert "atlas" not in load_config(path).attachments.model_dump()


def test_template_names_recommended_ranges():
    text = render_template()
    assert "Recommended: 50 - 95." in text  # musclemap.overlap


# ---------------------------------------------------------------------------- warnings


def _fields(issues):
    return {i.field for i in issues}


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        (["mesh.muscles.passband=0.8"], "mesh.muscles.passband"),
        (["attachments.params.trim=0.6"], "attachments.params.trim"),
        (
            ["attachments.params.target_points=2000"],
            "attachments.params.target_points",
        ),
        (["segmentation.totalsegmentator.fast=true"], "segmentation.totalsegmentator.fast"),
        (
            ["segmentation.totalsegmentator.roi_subset=false"],
            "segmentation.totalsegmentator.roi_subset",
        ),
        (["segmentation.muscles=totalsegmentator"], "segmentation.muscles"),
        (["segmentation.musclemap.model_version=1.3"], "segmentation.musclemap.model_version"),
        (
            ["labelmap.muscles.opening_radius_mm=3", "labelmap.muscles.closing_radius_mm=0.5"],
            "labelmap.muscles.opening_radius_mm",
        ),
        (["mesh.repair.enabled=false"], "mesh.repair.enabled"),
        (["mesh.on_invalid=warn"], "mesh.on_invalid"),
        (["export.bone_weights=PositionOnFibreQuadraticSpline"], "export.bone_weights"),
        (["export.fibre_export=true"], "export.fibre_export"),
        (["export.motion.end_deg=150"], "export.motion.end_deg"),
        (
            ["export.motion.coordinate=knee_angle_r", "export.motion.end_deg=-150"],
            "export.motion.end_deg",
        ),
        (["export.motion.coordinate=hip_flexion_l"], "export.motion.coordinate"),
        (["export.motion.step_deg=0.05"], "export.motion.step_deg"),
    ],
)
def test_config_warnings(overrides, field):
    issues = sanity.check_config(api.resolved_config(overrides=overrides))
    assert field in _fields(issues)
    assert all(i.severity in ("warning", "info") for i in issues)


def test_cpd_is_noted_as_comparison_only():
    from mskpipe.plugins.attachments.atlas_registration import AtlasRegistration

    cfg = api.resolved_config(overrides=["attachments.params.nonrigid=cpd"])
    assert any("comparison" in n for n in AtlasRegistration.preflight_warnings(cfg, "ct"))


# ---------------------------------------------------------------------------- environment


def test_threads_above_cores(monkeypatch):
    monkeypatch.setattr(sanity.os, "cpu_count", lambda: 4)
    cfg = load_config(overrides=["runtime.threads=8"])
    assert "runtime.threads" in _fields(sanity.check_environment(cfg))
    assert "runtime.threads" not in _fields(
        sanity.check_environment(load_config(overrides=["runtime.threads=4"]))
    )


def test_saving_workers_vs_ram(monkeypatch):
    import psutil

    memory = namedtuple("memory", "total")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory(8 * sanity.GIB))
    cfg = load_config(overrides=["segmentation.totalsegmentator.nr_thr_saving=6"])
    assert "segmentation.totalsegmentator.nr_thr_saving" in _fields(sanity.check_environment(cfg))
    cfg = load_config(overrides=["segmentation.totalsegmentator.nr_thr_saving=2"])
    assert "segmentation.totalsegmentator.nr_thr_saving" not in _fields(
        sanity.check_environment(cfg)
    )


def test_runs_dir_not_writable(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    cfg = load_config(overrides=[f"runtime.runs_dir={(blocker / 'runs').as_posix()}"])
    issues = [i for i in sanity.check_environment(cfg) if i.field == "runtime.runs_dir"]
    assert issues and issues[0].severity == "error"


def test_low_disk_space(tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(sanity.shutil, "disk_usage", lambda _: usage(10, 9, sanity.GIB))
    cfg = load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}"])
    issues = [i for i in sanity.check_environment(cfg) if i.field == "runtime.runs_dir"]
    assert issues and issues[0].severity == "warning" and "GiB free" in issues[0].message


def test_gpu_requested_without_gpu(monkeypatch):
    import mskpipe.core.device as device

    monkeypatch.setattr(device, "gpu_unavailable_reason", lambda: "no NVIDIA GPU found")
    errors = [
        i
        for i in sanity.check_environment(load_config(overrides=["runtime.device=gpu"]))
        if i.severity == "error"
    ]
    assert errors and errors[0].field == "runtime.device"
    monkeypatch.setattr(device, "gpu_unavailable_reason", lambda: None)
    assert not [
        i
        for i in sanity.check_environment(load_config(overrides=["runtime.device=gpu"]))
        if i.severity == "error"
    ]


# ---------------------------------------------------------------------------- input


def _volume(path: Path, spacing, ct: bool = True) -> Path:
    data = np.full((10, 10, 8), -1024 if ct else 0, np.int16)
    data[3:7, 3:7, 2:6] = 40 if ct else 300
    affine = np.diag([*spacing, 1.0])
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, 1)
    img.set_sform(affine, 1)
    nib.save(img, str(path))
    return path


def test_input_spacing(tmp_path):
    cfg = load_config()
    assert sanity.check_input(cfg, _volume(tmp_path / "ok.nii.gz", (0.8, 0.8, 1.5))) == []
    thick = sanity.check_input(cfg, _volume(tmp_path / "thick.nii.gz", (0.8, 0.8, 4.0)))
    assert any("slices are 4 mm apart" in i.message for i in thick)
    coarse = sanity.check_input(cfg, _volume(tmp_path / "coarse.nii.gz", (2.5, 2.5, 2.5)))
    assert any("in-plane spacing" in i.message for i in coarse)


def test_morphology_radius_below_voxel(tmp_path):
    cfg = load_config(overrides=["labelmap.muscles.closing_radius_mm=0.5"])
    issues = sanity.check_input(cfg, _volume(tmp_path / "v.nii.gz", (0.8, 0.8, 1.5)))
    assert "labelmap.muscles.closing_radius_mm" in _fields(issues)


def test_modality_mismatch(tmp_path):
    cfg = load_config()
    mri = _volume(tmp_path / "mri.nii.gz", (0.8, 0.8, 1.5), ct=False)
    assert any("looks like MRI" in i.message for i in sanity.check_input(cfg, mri, "ct"))
    assert sanity.check_input(cfg, mri, "mri") == []
    assert sanity.check_input(cfg, mri) == []  # modality not given: not read


# ---------------------------------------------------------------------------- wiring


def test_preflight_reports_sanity_warnings(tmp_path):
    image = _volume(tmp_path / "s01.nii.gz", (0.8, 0.8, 1.5))
    prepared = api.prepare_run(
        api.RunRequest(
            image=image,
            modality="mri",  # the volume is CT
            runs_dir=tmp_path / "runs",
            overrides=("mesh.muscles.passband=0.8",),
            until_step="segment",
        )
    )
    issues = api.preflight(prepared, default_registry())
    assert "mesh.muscles.passband" in _fields(issues)
    assert any("looks like CT" in i.message for i in issues)


def test_cli_config_check(tmp_path):
    runner = CliRunner()
    ok = runner.invoke(app, ["config", "check"])
    assert ok.exit_code == 0 and "OK" in ok.output
    warn = runner.invoke(app, ["config", "check", "--set", "export.num_of_lines=1000"])
    assert warn.exit_code == 0 and "export.num_of_lines = 1000" in warn.output
    bad = runner.invoke(app, ["config", "check", "--set", "export.num_of_lines=0"])
    assert bad.exit_code == 2
    image = _volume(tmp_path / "s.nii.gz", (0.8, 0.8, 4.0))
    vol = runner.invoke(app, ["config", "check", "--image", str(image), "--modality", "ct"])
    assert vol.exit_code == 0 and "slices are 4 mm apart" in vol.output


def test_model_version_given_as_number():
    assert (
        load_config(
            overrides=["segmentation.musclemap.model_version=1.4"]
        ).segmentation.musclemap.model_version
        == "1.4"
    )
