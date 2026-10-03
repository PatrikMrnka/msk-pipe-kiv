# SPDX-License-Identifier: Apache-2.0
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from osim_models import write_hip_model

from mskpipe.config import InputSpec, load_config
from mskpipe.core.registry import Registry
from mskpipe.core.runner import PipelineError, run_pipeline
from mskpipe.core.step import Step, StepContext
from mskpipe.plugins.base import SkeletonPlugin
from mskpipe.plugins.skeleton.pystaple_backend import build_hip_model, required_bones
from mskpipe.steps.skeleton import SKELETON_FILE, SkeletonStep, read_skeleton_index


class FakeMesh(Step):
    """Writes ``meshes.json`` with bone files (content irrelevant to the fake backend)."""

    name = "mesh"
    bones: tuple[str, ...] = ("pelvis_no_sacrum", "femur_r", "tibia_r")

    def run(self, ctx: StepContext) -> None:
        structures = []
        for name in FakeMesh.bones:
            rel = f"bones/{name}.stl"
            (ctx.out_dir / "bones").mkdir(exist_ok=True)
            (ctx.out_dir / rel).write_bytes(b"solid x\nendsolid x\n")
            structures.append({"name": name, "kind": "bone", "status": "ok", "file": rel})
        structures.append({"name": "iliacus_r", "kind": "muscle", "status": "ok", "file": "m"})
        index = {"format": "mskpipe.meshes", "version": 1, "structures": structures}
        (ctx.out_dir / "meshes.json").write_text(json.dumps(index), encoding="utf-8")


class FakeStaple(SkeletonPlugin):
    """Stands in for pystaple: writes a synthetic model; behaviour set by class attributes."""

    name = "pystaple"
    mode = "ok"  # ok | mirrored | fail
    received: dict[str, Path] | None = None

    def build(self, ctx, bones, out_dir, params) -> Path:
        FakeStaple.received = dict(bones)
        if FakeStaple.mode == "fail":
            raise ValueError("femur head not found")
        warnings.warn("hip_r: parent_location missing", stacklevel=1)
        (out_dir / "Geometry").mkdir(exist_ok=True)
        for b in ("pelvis_no_sacrum", "femur_r", "tibia_r"):
            (out_dir / "Geometry" / f"{b}.obj").write_text("v 0 0 0\n", encoding="utf-8")
        hip = (-55.0, -82.0, -95.0 if FakeStaple.mode == "mirrored" else 95.0)
        return write_hip_model(
            out_dir / f"{ctx.config.skeleton.model_name}.osim", hip_in_pelvis_mm=hip
        )


@pytest.fixture(autouse=True)
def _reset():
    FakeMesh.bones = ("pelvis_no_sacrum", "femur_r", "tibia_r")
    FakeStaple.mode, FakeStaple.received = "ok", None


@pytest.fixture
def registry() -> Registry:
    reg = Registry()
    reg.register(FakeStaple)
    return reg


@pytest.fixture
def spec(tmp_path: Path) -> InputSpec:
    image = tmp_path / "subject.nii.gz"
    image.write_bytes(b"fake volume")
    return InputSpec(image=image, modality="ct")


def config(tmp_path: Path, *overrides: str):
    return load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}", *overrides])


def test_skeleton_step_outputs(tmp_path, spec, registry):
    result = run_pipeline([FakeMesh(), SkeletonStep(registry)], spec, config(tmp_path))
    out = result.ws.step_dir("skeleton")
    mesh_dir = result.ws.step_dir("mesh")

    assert FakeStaple.received == {n: mesh_dir / "bones" / f"{n}.stl" for n in FakeMesh.bones}
    index = read_skeleton_index(out)
    assert index["model"] == "bone_model.osim" and index["model_name"] == "auto2020_hip_R"
    assert index["geometry"] == [f"Geometry/{b}.obj" for b in sorted(required_bones("r"))]
    assert index["coordinates"]["space"] == "world_ras"
    assert index["settings"]["body_mass"] == 64.0
    assert index["warnings"] == ["hip_r: parent_location missing"]
    assert index["qc"]["side_consistent"] and index["qc"]["problems"] == []

    rec = result.manifest.steps["skeleton"]
    assert rec.status == "completed"
    assert (rec.metrics["n_bodies"], rec.metrics["n_joints"], rec.metrics["n_markers"]) == (3, 3, 2)
    assert rec.metrics["femur_length_mm"] == pytest.approx(400.0)
    assert rec.metrics["n_warnings"] == 1 and rec.metrics["qc_problems"] == []
    assert {"04_skeleton/bone_model.osim", f"04_skeleton/{SKELETON_FILE}"} <= set(rec.outputs)


def test_model_name_from_config(tmp_path, spec, registry):
    cfg = config(tmp_path, "skeleton.model_name=hip_model")
    result = run_pipeline([FakeMesh(), SkeletonStep(registry)], spec, cfg)
    assert (result.ws.step_dir("skeleton") / "hip_model.osim").is_file()


def test_mirrored_model_fails_after_writing_index(tmp_path, spec, registry):
    FakeStaple.mode = "mirrored"
    with pytest.raises(PipelineError, match="skeleton") as info:
        run_pipeline([FakeMesh(), SkeletonStep(registry)], spec, config(tmp_path))
    ws = info.value.ws
    manifest = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    assert "mirrored" in manifest["steps"]["skeleton"]["error"]
    index = read_skeleton_index(ws.step_dir("skeleton"))
    assert index["qc"]["side_consistent"] is False


def test_backend_error_becomes_step_error(tmp_path, spec, registry):
    FakeStaple.mode = "fail"
    with pytest.raises(PipelineError) as info:
        run_pipeline([FakeMesh(), SkeletonStep(registry)], spec, config(tmp_path))
    assert "pystaple: femur head not found" in str(info.value.__cause__)


def test_no_bone_meshes(tmp_path, spec, registry):
    FakeMesh.bones = ()
    with pytest.raises(PipelineError) as info:
        run_pipeline([FakeMesh(), SkeletonStep(registry)], spec, config(tmp_path))
    assert "No bone meshes" in str(info.value.__cause__)
    assert FakeStaple.received is None


def test_fingerprint_follows_config_and_plugin(registry):
    step = SkeletonStep(registry)
    base = load_config()
    extra = step.fingerprint_extra(base)
    assert extra["plugin"]["plugin"] == "skeleton/pystaple"
    assert set(extra["packages"]) == {"pystaple", "fast-simplification"}
    assert base.fingerprint("skeleton") != load_config(
        overrides=["skeleton.body_mass=80"]
    ).fingerprint("skeleton")


@pytest.mark.parametrize(
    "override",
    ["skeleton.femur_algorithm=Miranda", "skeleton.body_mass=0", "skeleton.geometry_reduction=0"],
)
def test_unsupported_settings_are_rejected(override):
    from mskpipe.config import ConfigError

    with pytest.raises(ConfigError):
        load_config(overrides=[override])


def test_missing_bones_checked_before_pystaple(tmp_path):
    cfg = load_config().skeleton
    with pytest.raises(FileNotFoundError, match="femur_r, tibia_r"):
        build_hip_model({"pelvis_no_sacrum": tmp_path / "p.stl"}, tmp_path, cfg)
    assert required_bones("l") == ("pelvis_no_sacrum", "femur_l", "tibia_l")


def test_tibia_body_includes_fibula(tmp_path, spec, registry):
    from mskpipe.io.mesh_io import TriMesh, read_mesh, write_mesh

    tri = TriMesh(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))

    class MeshWithFibula(FakeMesh):
        def run(self, ctx: StepContext) -> None:
            FakeMesh.bones = ("pelvis_no_sacrum", "femur_r", "tibia_r", "fibula_r")
            super().run(ctx)
            for name in ("tibia_r", "fibula_r"):
                write_mesh(ctx.out_dir / "bones" / f"{name}.stl", tri.transformed(np.eye(4)))

    result = run_pipeline([MeshWithFibula(), SkeletonStep(registry)], spec, config(tmp_path))
    out = result.ws.step_dir("skeleton")
    merged = out / "bodies" / "tibia_r.stl"
    assert FakeStaple.received["tibia_r"] == merged
    assert read_mesh(merged).n_faces == 2
    index = read_skeleton_index(out)
    assert index["body_geometry_sources"]["tibia_r"] == ["tibia_r", "fibula_r"]

    result = run_pipeline(
        [MeshWithFibula(), SkeletonStep(registry)],
        spec,
        config(tmp_path, "skeleton.include_fibula=false"),
    )
    mesh_dir = result.ws.step_dir("mesh")
    assert FakeStaple.received["tibia_r"] == mesh_dir / "bones" / "tibia_r.stl"
