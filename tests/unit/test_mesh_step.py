# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mskpipe.config import InputSpec, load_config
from mskpipe.core.runner import PipelineError, run_pipeline
from mskpipe.core.step import Step, StepContext
from mskpipe.geometry.metrics import mesh_stats
from mskpipe.io.labels import LABELMAP_FILE, LABELS_FILE, Label, LabelTable
from mskpipe.io.mesh_io import read_mesh
from mskpipe.steps.mesh import MESHES_FILE, MeshStep, mesh_paths

AFFINE = np.array(
    [[-1.2, 0.0, 0.0, 90.0], [0.0, 1.2, 0.0, -40.0], [0.0, 0.0, 2.0, -300.0], [0, 0, 0, 1]]
)


class FakeLabelmap(Step):
    """Writes a synthetic label map: a bone ball, a muscle ellipsoid, one empty label."""

    name = "labelmap"
    empty_only = False

    def run(self, ctx: StepContext) -> None:
        data = np.zeros((48, 40, 30), dtype=np.uint8)
        grid = np.indices(data.shape).astype(float)
        if not FakeLabelmap.empty_only:
            ball = ((grid[0] - 15) ** 2 + (grid[1] - 20) ** 2 + ((grid[2] - 15) * 1.6) ** 2) < 64
            data[ball] = 3
            ell = ((grid[0] - 34) / 9) ** 2 + ((grid[1] - 20) / 6) ** 2 + ((grid[2] - 15) / 8) ** 2
            data[ell < 1] = 7
        nib.save(nib.Nifti1Image(data, AFFINE), ctx.out_dir / LABELMAP_FILE)
        LabelTable(
            labels=(
                Label(name="femur_r", value=3, kind="bone"),
                Label(name="gluteus_maximus_r", value=7, kind="muscle"),
                Label(name="iliacus_r", value=9, kind="muscle"),
            )
        ).save(ctx.out_dir / LABELS_FILE)


@pytest.fixture(autouse=True)
def _reset():
    FakeLabelmap.empty_only = False


@pytest.fixture
def spec(tmp_path: Path) -> InputSpec:
    image = tmp_path / "subject.nii.gz"
    image.write_bytes(b"fake volume")
    return InputSpec(image=image, modality="ct")


def config(tmp_path: Path, *overrides: str):
    return load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}", *overrides])


def test_mesh_step_outputs(tmp_path, spec):
    result = run_pipeline([FakeLabelmap(), MeshStep()], spec, config(tmp_path))
    out = result.ws.step_dir("mesh")

    bones, muscles = mesh_paths(out, "bone"), mesh_paths(out, "muscle")
    assert bones == {"femur_r": out / "bones" / "femur_r.stl"}
    assert muscles == {"gluteus_maximus_r": out / "muscles" / "gluteus_maximus_r.obj"}

    index = json.loads((out / MESHES_FILE).read_text(encoding="utf-8"))
    assert index["coordinates"] == {"space": "world_ras", "units": "mm"}
    by_name = {s["name"]: s for s in index["structures"]}
    assert by_name["iliacus_r"]["status"] == "missing"
    femur = by_name["femur_r"]
    assert femur["closed"] and femur["volume_ratio"] == pytest.approx(1.0, abs=0.05)

    mesh = read_mesh(bones["femur_r"])
    assert mesh_stats(mesh)["volume"] == pytest.approx(femur["volume"], rel=1e-4)
    center = AFFINE[:3, :3] @ np.array([15, 20, 15]) + AFFINE[:3, 3]
    assert np.allclose(mesh.vertices.mean(0), center, atol=0.5)

    rec = result.manifest.steps["mesh"]
    assert rec.metrics["n_meshed"] == 2 and rec.metrics["missing"] == ["iliacus_r"]
    assert set(rec.metrics["structures"]) == {"femur_r", "gluteus_maximus_r"}
    assert "03_mesh/meshes.json" in rec.outputs


def test_mesh_params_per_kind(tmp_path, spec):
    base = run_pipeline([FakeLabelmap(), MeshStep()], spec, config(tmp_path))
    other = run_pipeline(
        [FakeLabelmap(), MeshStep()],
        spec,
        config(tmp_path, "mesh.muscles.target_reduction=0.5", "runtime.cache=false"),
    )
    faces = [r.manifest.steps["mesh"].metrics["structures"] for r in (base, other)]
    assert faces[0]["femur_r"]["n_faces"] == faces[1]["femur_r"]["n_faces"]
    assert faces[1]["gluteus_maximus_r"]["n_faces"] > faces[0]["gluteus_maximus_r"]["n_faces"]


def test_second_run_is_cached(tmp_path, spec):
    run_pipeline([FakeLabelmap(), MeshStep()], spec, config(tmp_path))
    second = run_pipeline([FakeLabelmap(), MeshStep()], spec, config(tmp_path))
    assert second.manifest.steps["mesh"].status == "cached"
    assert (second.ws.step_dir("mesh") / "bones" / "femur_r.stl").is_file()


def test_all_labels_empty_fails(tmp_path, spec):
    FakeLabelmap.empty_only = True
    with pytest.raises(PipelineError, match="all labels empty"):
        run_pipeline([FakeLabelmap(), MeshStep()], spec, config(tmp_path))


def test_missing_labelmap_fails(tmp_path, spec):
    with pytest.raises(PipelineError, match="Label table not found"):
        run_pipeline([MeshStep()], spec, config(tmp_path))


def _drop_one_face(monkeypatch):
    """Quadric meshes lose a triangle (the defect quadric decimation leaves on thin bone)."""
    from mskpipe.geometry import surface
    from mskpipe.io.mesh_io import TriMesh

    real = surface.mask_to_surface

    def opened(*args, decimation="quadric", **kwargs):
        result = real(*args, decimation=decimation, **kwargs)
        if decimation == "quadric":
            mesh = TriMesh(result.mesh.vertices, result.mesh.faces[1:])
            return surface.SurfaceResult(mesh, result.info)
        return result

    monkeypatch.setattr(surface, "mask_to_surface", opened)


def _index(result):
    out = result.ws.step_dir("mesh")
    data = json.loads((out / MESHES_FILE).read_text(encoding="utf-8"))
    return out, {s["name"]: s for s in data["structures"]}


def test_open_surfaces_are_repaired_in_the_mesh_step(tmp_path, spec, monkeypatch):
    """Open surfaces are repaired here (bones: topology-preserving decimation; muscles: the
    mask remeshing Muscle Wrapping needs), recorded in meshes.json and defects.json."""
    from mskpipe.qc.defects import DefectReport

    _drop_one_face(monkeypatch)
    result = run_pipeline([FakeLabelmap(), MeshStep()], spec, config(tmp_path))
    out, by_name = _index(result)
    femur, muscle = by_name["femur_r"], by_name["gluteus_maximus_r"]
    for e in (femur, muscle):
        assert e["valid"] and e["repaired"] and e["closed"] and e["problems"] == []
        assert e["decimation"] == "topology"
        assert e["repair"]["original_problems"] == ["boundary edges: 3"]
        assert e["repair"]["attempts"][0]["method"] == "mesh"  # cannot close a hole
        assert mesh_stats(read_mesh(out / e["file"]))["closed"]
    # the coarse test femur: the patch closes the hole but changes the volume too much
    patch = femur["repair"]["attempts"][1]
    assert patch["method"] == "patch" and not patch["accepted"]
    assert femur["repair"]["used"] == "topology-preserving decimation"
    metrics = result.manifest.steps["mesh"].metrics
    assert metrics["repaired"] == ["femur_r", "gluteus_maximus_r"]
    assert metrics["invalid"] == [] and metrics["not_closed"] == []
    (d,) = DefectReport.read(out).defects  # bones only; muscles are reported by export_mw2
    assert d.structure == "femur_r" and d.fixed and d.center_mm is not None
    assert "boundary edges: 3" in d.detail


def test_unrepairable_bone_stops_the_run(tmp_path, spec, monkeypatch):
    _drop_one_face(monkeypatch)
    cfg = config(tmp_path, "mesh.repair.enabled=false")
    with pytest.raises(PipelineError, match="mesh") as info:
        run_pipeline([FakeLabelmap(), MeshStep()], spec, cfg)
    assert "femur_r (boundary edges: 3)" in str(info.value.__cause__)
    _, by_name = _index(info.value)  # the index is written before failing
    assert not by_name["femur_r"]["valid"] and "repair" not in by_name["femur_r"]

    warn = config(tmp_path, "mesh.repair.enabled=false", "mesh.on_invalid=warn")
    result = run_pipeline([FakeLabelmap(), MeshStep()], spec, warn)
    assert result.manifest.steps["mesh"].metrics["invalid"] == ["femur_r", "gluteus_maximus_r"]


def test_old_export_repair_keys_are_moved():
    cfg = load_config(overrides=["export.repair=false", "export.repair_max_opening_mm=2"])
    assert cfg.mesh.repair.enabled is False and cfg.mesh.repair.max_opening_mm == 2
    assert not hasattr(cfg.export, "repair")
    new = load_config(
        overrides=["export.repair_max_closing_mm=3", "mesh.repair.max_closing_mm=4.5"]
    )
    assert new.mesh.repair.max_closing_mm == 4.5  # an explicit new key wins


def test_genus_check_reports_open_and_handled_bones():
    from mskpipe.geometry.surface import mask_to_surface
    from mskpipe.io.mesh_io import TriMesh
    from mskpipe.steps.mesh import _genus_check

    grid = np.indices((40, 40, 20)).astype(float)
    r = np.hypot(grid[0] - 20, grid[1] - 20)
    ring = (r > 6) & (r < 14) & (np.abs(grid[2] - 10) < 5)  # torus-like: genus 1
    params = load_config().mesh.bones
    mesh = mask_to_surface(ring, np.eye(4), params).mesh

    genera, defect = _genus_check("femur_r", mesh)  # femur: anatomical genus 0
    assert genera == [1] and defect.kind == "surface" and not defect.fixed
    assert np.allclose(defect.center_mm, [20, 20, 10], atol=1.0)
    assert "genus 1 > anatomical 0" in defect.detail

    genera, defect = _genus_check("femur_r", TriMesh(mesh.vertices, mesh.faces[1:]))
    assert genera == [None] and "not closed" in defect.detail
    assert _genus_check("sacrum", mesh)[1] is None  # sacrum: not checked
