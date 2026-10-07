# SPDX-License-Identifier: Apache-2.0
"""Step export_mw2 on a synthetic subject (fake upstream steps)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import vtk
from osim_models import (
    IMAGE_HIP_DEG,
    KNEE_OFF_AXIS_DEG,
    PELVIS_ORIGIN,
    staple_model_text,
    write_staple_model,
)
from vtk.util.numpy_support import vtk_to_numpy

from mskpipe.config import InputSpec, load_config
from mskpipe.config.schema import MeshParams
from mskpipe.core.runner import PipelineError, run_pipeline
from mskpipe.core.step import Step, StepContext
from mskpipe.geometry.attachment_guard import is_degenerate, mean_edge_length, outline_stats
from mskpipe.geometry.frames import RAS_TO_OPENSIM, rotate_points
from mskpipe.geometry.surface import mask_to_surface
from mskpipe.geometry.topology import check_surface
from mskpipe.io.attachment_vtk import read_points, write_points
from mskpipe.io.mesh_io import TriMesh, read_mesh, write_mesh
from mskpipe.io.osim import read_osim
from mskpipe.steps.export_mw2 import ExportMw2Step, read_export_index

# world (mm) of voxel (0, 0, 0); 1 mm voxels
ORIGIN = np.array([-80.0, -60.0, -460.0])
SHAPE = (160, 130, 450)

BONES = {
    "pelvis_no_sacrum": ((0, 0, 0), (100, 60, 60)),
    "sacrum": ((0, -80, 20), (25, 15, 35)),
    "femur_r": ((0, 0, -300), (25, 25, 200)),
    "tibia_r": ((0, 0, -700), (25, 25, 180)),
    "fibula_r": ((40, 0, -700), (8, 8, 160)),
}
# name: label, ellipsoids (center, radii), tunnel?, outlines (center, radius) or None
MUSCLES = {
    "iliacus_r": (11, [((0, 50, -90), (20, 15, 60))], False),
    "gracilis_r": (12, [((60, 40, -300), (12, 12, 100)), ((60, 40, -440), (4, 4, 4))], False),
    "adductor_longus_r": (13, [((-60, 40, -200), (12, 12, 40))], False),
    "sartorius_r": (14, [((-60, -40, -300), (10, 10, 60))], False),
    "pectineus_r": (15, [((50, -30, -95), (12, 12, 18))], True),
}
# torus (genus 1) with a hole wider than any closing, tube thicker than any opening
TORUS = ("obturator_internus_r", 16, (-40, -25, -400), 14.0, 5.0)
AREAS = {  # muscle: {kind: (ring centre, radius, body)}
    "iliacus_r": {"Ori": ((0, 50, -28), 8, "pelvis"), "Ins": ((0, 50, -152), 8, "femur_r")},
    "gracilis_r": {"Ori": ((60, 40, -210), 7, "pelvis"), "Ins": ((60, 40, -390), 7, "tibia_r")},
    "adductor_longus_r": {  # origin far above the muscle: projection collapses
        "Ori": ((-60, 40, -20), 8, "pelvis"),
        "Ins": ((-60, 40, -242), 7, "femur_r"),
    },
    "pectineus_r": {"Ori": ((50, -30, -80), 6, "pelvis"), "Ins": ((50, -30, -110), 6, None)},
    "obturator_internus_r": {
        "Ori": ((-54, -25, -400), 4, "pelvis"),
        "Ins": ((-26, -25, -400), 4, "femur_r"),
    },
}
STEMS = {"iliacus_r": "Iliacus", "gracilis_r": "Gracilis", "adductor_longus_r": "AddLong"}


def blob(center, radii) -> TriMesh:
    src = vtk.vtkSphereSource()
    src.SetThetaResolution(48)
    src.SetPhiResolution(32)
    src.Update()
    poly = src.GetOutput()
    v = vtk_to_numpy(poly.GetPoints().GetData()).astype(float) * 2.0
    f = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return TriMesh(v * np.asarray(radii) + np.asarray(center), f)


def ring(center, radius, n=12) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(center, float) + radius * np.c_[np.cos(a), np.sin(a), np.zeros(n)]


def labelmap() -> np.ndarray:
    data = np.zeros(SHAPE, dtype=np.uint8)
    for value, parts, tunnel in MUSCLES.values():
        for center, radii in parts:
            c = np.asarray(center, float) - ORIGIN
            r = np.asarray(radii, float)
            lo = np.floor(c - r - 1).astype(int)
            hi = np.ceil(c + r + 2).astype(int)
            box = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
            g = np.indices(hi - lo).astype(float) + lo[:, None, None, None]
            inside = (((g - c[:, None, None, None]) / r[:, None, None, None]) ** 2).sum(0) < 1
            if tunnel:  # thin hole through the muscle along y: genus 1
                inside &= (g[0] - c[0]) ** 2 + (g[2] - c[2]) ** 2 > 1.0
            data[box][inside] = value
    _, value, center, ring_r, tube_r = TORUS
    c = np.asarray(center, float) - ORIGIN
    lo = np.floor(c - ring_r - tube_r - 2).astype(int)
    hi = np.ceil(c + ring_r + tube_r + 2).astype(int)
    box = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    g = np.indices(hi - lo).astype(float) + lo[:, None, None, None]
    r = np.hypot(g[0] - c[0], g[1] - c[1])
    data[box][(r - ring_r) ** 2 + (g[2] - c[2]) ** 2 < tube_r**2] = value
    return data


class FakeLabelmap(Step):
    name = "labelmap"

    def run(self, ctx: StepContext) -> None:
        affine = np.eye(4)
        affine[:3, 3] = ORIGIN
        nib.save(nib.Nifti1Image(labelmap(), affine), ctx.out_dir / "labelmap.nii.gz")
        labels = [{"name": n, "value": v[0], "kind": "muscle"} for n, v in MUSCLES.items()]
        labels.append({"name": TORUS[0], "value": TORUS[1], "kind": "muscle"})
        index = {"format": "mskpipe.labels", "version": 1, "labels": labels}
        (ctx.out_dir / "labels.json").write_text(json.dumps(index), encoding="utf-8")


class FakeMesh(Step):
    name = "mesh"

    def run(self, ctx: StepContext) -> None:
        structures = []
        for name, (c, r) in BONES.items():
            rel = f"bones/{name}.stl"
            write_mesh(ctx.out_dir / rel, blob(c, r))
            structures.append({"name": name, "kind": "bone", "status": "ok", "file": rel})
        data = labelmap()
        affine = np.eye(4)
        affine[:3, 3] = ORIGIN
        values = {n: v[0] for n, v in MUSCLES.items()} | {TORUS[0]: TORUS[1]}
        for name, value in values.items():
            params = MeshParams(min_component_fraction=0.0)  # keep gracilis' 2nd part
            mesh = mask_to_surface(data == value, affine, params).mesh
            rel = f"muscles/{name}.obj"
            write_mesh(ctx.out_dir / rel, mesh)
            structures.append({"name": name, "kind": "muscle", "status": "ok", "file": rel})
        index = {"format": "mskpipe.meshes", "version": 1, "structures": structures}
        (ctx.out_dir / "meshes.json").write_text(json.dumps(index), encoding="utf-8")


class FakeSkeleton(Step):
    name = "skeleton"

    def run(self, ctx: StepContext) -> None:
        write_staple_model(ctx.out_dir / "bone_model.osim")
        index = {"format": "mskpipe.skeleton", "version": 1, "model": "bone_model.osim"}
        (ctx.out_dir / "skeleton.json").write_text(json.dumps(index), encoding="utf-8")


class FakeAttachments(Step):
    name = "attachments"

    def run(self, ctx: StepContext) -> None:
        files, areas = {}, []
        for muscle, kinds in AREAS.items():
            stem = STEMS.get(muscle, muscle.split("_")[0].capitalize())
            for kind, (center, radius, body) in kinds.items():
                rel = f"{muscle}/{stem}_{kind}.vtk"
                write_points(ctx.out_dir / rel, ring(center, radius))
                files.setdefault(muscle, []).append(rel)
                if body is not None:  # pectineus Ins: no record, body from nearest bone
                    areas.append({"muscle": muscle, "kind": kind, "status": "ok", "body": body})
        index = {
            "format": "mskpipe.attachments",
            "version": 1,
            "muscles": files,
            "areas": areas,
        }
        (ctx.out_dir / "attachments.json").write_text(json.dumps(index), encoding="utf-8")


STEPS = [FakeLabelmap(), FakeMesh(), FakeSkeleton(), FakeAttachments(), ExportMw2Step()]


@pytest.fixture
def spec(tmp_path: Path) -> InputSpec:
    image = tmp_path / "subject.nii.gz"
    image.write_bytes(b"fake volume")
    return InputSpec(image=image, modality="ct")


def config(tmp_path: Path, *overrides: str):
    return load_config(overrides=[f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}", *overrides])


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export")
    image = tmp / "subject.nii.gz"
    image.write_bytes(b"fake volume")
    spec = InputSpec(image=image, modality="ct")
    return run_pipeline(STEPS, spec, config(tmp))


def test_export_layout_and_setup(default_run):
    out = default_run.ws.step_dir("export_mw2")
    index = read_export_index(out)
    status = {m["name"]: m["status"] for m in index["muscles"]}
    assert status == {
        "adductor_longus_r": "ok",
        "gracilis_r": "ok",
        "iliacus_r": "ok",
        "obturator_internus_r": "excluded",
        "pectineus_r": "ok",
        "sartorius_r": "excluded",
    }
    setup = ET.parse(out / index["setup"]).getroot().find("MuscleGeneratorTool")
    gens = setup.findall("MuscleGeneratorSet/objects/MuscleGenerator")
    assert [g.get("name") for g in gens] == [
        "adductor_longus_r",
        "gracilis_r",
        "iliacus_r",
        "pectineus_r",
    ]
    for g in gens:  # every referenced file exists, relative to the setup XML
        files = [g.findtext("MuscleGeometry/Mesh/mesh_file")]
        files += [a.findtext("point_file") for a in g.iter("AttachmentArea")]
        assert all((out / f).is_file() for f in files)
    assert (out / setup.findtext("model_file")).is_file()
    assert (out / setup.findtext("motion_file")).is_file()
    assert (out / "analysis").is_dir() and not (out / "fibres").exists()
    iliacus = next(g for g in gens if g.get("name") == "iliacus_r")
    assert iliacus.findtext("MuscleGeometry/Mesh/mesh_file") == "muscles/iliacus_r/Iliacus.obj"
    bodies = [a.findtext("body") for a in iliacus.iter("AttachmentArea")]
    assert bodies == ["pelvis", "femur_r"]
    pect = next(g for g in gens if g.get("name") == "pectineus_r")
    assert [a.findtext("body") for a in pect.iter("AttachmentArea")] == ["pelvis", "femur_r"]


def test_exclusion_reasons_and_repairs(default_run):
    index = read_export_index(default_run.ws.step_dir("export_mw2"))
    rec = {m["name"]: m for m in index["muscles"]}
    assert rec["sartorius_r"]["problems"] == ["no origin area", "no insertion area"]
    # outline checks only warn: a far, collapsing attachment is still exported
    assert rec["adductor_longus_r"]["problems"] == []
    assert "origin collapsed" in rec["adductor_longus_r"]["warnings"]
    assert rec["adductor_longus_r"]["areas"]["Ori"]["gap_mean_mm"] > 100
    assert (default_run.ws.step_dir("export_mw2") / "muscles" / "adductor_longus_r").is_dir()

    gracilis = rec["gracilis_r"]["mesh"]
    assert gracilis["repaired"] and gracilis["repair"]["attempts"][0]["method"] == "mesh"
    pect = rec["pectineus_r"]["mesh"]
    used = pect["repair"]["attempts"][pect["repair"]["best"]]
    assert pect["repaired"] and used["method"] == "remesh" and used["accepted"]
    assert used["closing_mm"] > 0 and pect["repair"]["used"].startswith("closing")
    assert pect["repair"]["distance_to_original_mm"]["mean"] < 1.0
    assert abs(pect["repair"]["volume_ratio"] - 1) <= 0.05

    torus = rec["obturator_internus_r"]
    # excluded by export.exclude (default), still measured and repaired for the record
    assert torus["excluded_by_config"].startswith("the belly lies inside the pelvis")
    assert torus["problems"][0].startswith("export.exclude: the belly lies")
    assert not torus["mesh"]["repaired"] and not torus["mesh"]["repair"]["ok"]
    assert torus["mesh"]["check"]["genus"] == 1  # best attempt kept for the record
    assert any(p.startswith("mesh repair failed (best") for p in torus["problems"])
    assert "mesh genus: 1" in torus["problems"]
    assert all("opening_mm" in a for a in torus["mesh"]["repair"]["attempts"][1:])
    assert not rec["iliacus_r"]["mesh"]["repaired"]
    for name in ("gracilis_r", "iliacus_r", "pectineus_r"):
        mesh = read_mesh(default_run.ws.step_dir("export_mw2") / rec[name]["files"]["mesh"])
        assert check_surface(mesh)["ok"]


def test_rest_pose_motion_and_model(default_run):
    out = default_run.ws.step_dir("export_mw2")
    index = read_export_index(out)
    pose = index["rest_pose"]
    assert pose["aligned_child_frames"] == ["knee_r"]
    assert pose["joints"]["knee_r"]["rotation_deg"] == pytest.approx(KNEE_OFF_AXIS_DEG, abs=1e-5)
    assert pose["errors_before_alignment"]["tibia_r"]["max_displacement_mm"] > 10
    assert all(e["max_displacement_mm"] < 1e-6 for e in pose["errors"].values())

    model = read_osim(out / index["model"])
    assert model.bodies["pelvis"].mesh_files == ("Geometry/pelvis.obj",)
    hip = model.joints["hip_r"].default_values
    assert np.degrees(hip["hip_flexion_r"]) == pytest.approx(IMAGE_HIP_DEG[0], abs=1e-6)
    assert index["bodies"]["pelvis"]["sources"] == ["mesh:pelvis_no_sacrum", "mesh:sacrum"]
    assert index["bodies"]["tibia_r"]["sources"] == ["mesh:tibia_r", "mesh:fibula_r"]

    motion = index["motion"]
    assert motion["coordinate"] == "hip_flexion_r"
    assert motion["columns"] == [
        "hip_flexion_r",
        "hip_adduction_r",
        "hip_rotation_r",
        "knee_angle_r",
    ]
    lines = (out / motion["file"]).read_text(encoding="utf-8").splitlines()
    rows = np.array([[float(v) for v in line.split("\t")] for line in lines[7:]])
    np.testing.assert_allclose(rows[0, 1:4], IMAGE_HIP_DEG, atol=1e-6)
    assert np.all(np.diff(rows[:, 1]) > 0) and rows[-1, 1] == pytest.approx(90.0)
    assert rows[1, 1] == pytest.approx(12.0) and len(rows) == motion["n_frames"] == 41
    np.testing.assert_allclose(rows[:, 2:], np.tile(rows[0, 2:], (len(rows), 1)))


def test_metrics(default_run):
    m = default_run.manifest.steps["export_mw2"].metrics
    assert m["n_selected"] == 6 and m["n_exported"] == 4
    assert set(m["excluded"]) == {"obturator_internus_r", "sartorius_r"}
    assert "origin collapsed" in m["outline_warnings"]["adductor_longus_r"]
    assert m["excluded_by_config"] == ["obturator_internus_r"]
    assert set(m["repaired"]) == {"gracilis_r", "pectineus_r"}
    assert m["joint_residual_deg"]["knee_r"] == pytest.approx(KNEE_OFF_AXIS_DEG, abs=1e-3)
    assert m["luca_body_mismatch"] == ["adductor_longus_r:Ori", "gracilis_r:Ori", "gracilis_r:Ins"]
    assert m["tendon_gap_mean_mm"]["iliacus_r"]["Ori"] < 8


def test_keep_invalid_and_no_alignment(tmp_path, spec):
    cfg = config(
        tmp_path,
        "export.on_invalid=keep",
        "export.exclude=[]",
        "export.align_child_frames=false",
        "export.muscles=[obturator_internus_r, iliacus_r]",
        "export.fibre_export=true",
    )
    result = run_pipeline(STEPS, spec, cfg)
    out = result.ws.step_dir("export_mw2")
    index = read_export_index(out)
    assert [m["status"] for m in index["muscles"]] == ["kept_invalid", "ok"]
    assert index["rest_pose"]["aligned_child_frames"] == []
    assert index["rest_pose"]["errors"]["tibia_r"]["max_displacement_mm"] > 10
    setup = ET.parse(out / index["setup"]).getroot().find("MuscleGeneratorTool")
    assert setup.findtext("export_folder") == "fibres/" and (out / "fibres").is_dir()
    assert len(setup.findall("MuscleGeneratorSet/objects/MuscleGenerator")) == 2


def test_motion_end_below_rest_pose_fails(tmp_path, spec):
    with pytest.raises(PipelineError, match="end_deg"):
        run_pipeline(STEPS, spec, config(tmp_path, "export.motion.end_deg=5"))


def test_no_valid_muscle_fails(tmp_path, spec):
    cfg = config(tmp_path, "export.muscles=[sartorius_r]")
    with pytest.raises(PipelineError, match="No muscle passed"):
        run_pipeline(STEPS, spec, cfg)


def test_excluded_reason():
    from mskpipe.steps.export_mw2 import EXCLUSION_REASONS, excluded_reason

    default = load_config().export.exclude
    assert set(default) == set(EXCLUSION_REASONS)
    assert excluded_reason("tensor_fasciae_latae_l", default).startswith("inserts on the lateral")
    assert excluded_reason("piriformis_r", default) is not None
    assert excluded_reason("iliacus_r", default) is None
    assert excluded_reason("iliacus_r", ["iliacus"]) == "listed in export.exclude"
    assert excluded_reason("piriformis_r", []) is None


def test_exclude_all_muscles_fails(tmp_path, spec):
    cfg = config(
        tmp_path,
        "export.exclude=[adductor_longus, gracilis, iliacus, pectineus]",
        "export.muscles=[adductor_longus_r, gracilis_r, iliacus_r, pectineus_r]",
    )
    with pytest.raises(PipelineError, match="No muscle passed"):
        run_pipeline(STEPS, spec, cfg)


# ------------------------------------------------------------------- OpenSim frame, guard


def exported_points(out: Path, rec: dict, kind: str) -> np.ndarray:
    return read_points(out / rec["files"][kind])


def test_export_is_in_the_opensim_frame(default_run):
    out = default_run.ws.step_dir("export_mw2")
    index = read_export_index(out)
    assert index["coordinates"]["space"] == "opensim_yup"
    np.testing.assert_array_equal(index["coordinates"]["rotation_from_ras"], RAS_TO_OPENSIM)
    # the femur is long along z in RAS, along y (up) in the export
    femur = read_mesh(out / index["bodies"]["femur_r"]["file"])
    assert np.argmax(np.ptp(femur.vertices, axis=0)) == 1
    source = read_mesh(default_run.ws.step_dir("mesh") / "bones" / "femur_r.stl")
    np.testing.assert_allclose(
        femur.vertices, rotate_points(source.vertices, RAS_TO_OPENSIM), atol=1e-3
    )
    rec = {m["name"]: m for m in index["muscles"]}
    centre, radius, _ = AREAS["iliacus_r"]["Ori"]
    np.testing.assert_allclose(
        exported_points(out, rec["iliacus_r"], "Ori"),
        rotate_points(ring(centre, radius), RAS_TO_OPENSIM),
        atol=1e-5,
    )
    muscle = read_mesh(out / rec["iliacus_r"]["files"]["mesh"])
    assert np.argmax(np.ptp(muscle.vertices, axis=0)) == 1
    assert "inflated" not in rec["iliacus_r"]["areas"]["Ori"]
    # the steps before stay in RAS
    assert np.argmax(np.ptp(source.vertices, axis=0)) == 2
    model = (default_run.ws.step_dir("skeleton") / "bone_model.osim").read_text(encoding="utf-8")
    assert model == staple_model_text()


def test_collapsed_outline_is_inflated(default_run):
    out = default_run.ws.step_dir("export_mw2")
    rec = {m["name"]: m for m in read_export_index(out)["muscles"]}["adductor_longus_r"]
    assert rec["status"] == "ok"  # the muscle is kept
    assert "adductor_longus_r origin: area inflated" in rec["warnings"]
    mesh = read_mesh(out / rec["files"]["mesh"])
    points = exported_points(out, rec, "Ori")
    stats = outline_stats(points)
    assert stats.distinct >= 3
    assert not is_degenerate(stats, min_distinct=3, min_span_mm=2 * mean_edge_length(mesh))
    # on vertices of the muscle, not on the bone 140 mm above it
    d = np.linalg.norm(mesh.vertices[None] - points[:, None], axis=2).min(axis=1)
    assert d.max() < 1e-3

    inflated = rec["areas"]["Ori"]["inflated"]
    assert inflated["radius_mm"] == 5.0
    assert inflated["before"]["span_mm"] < 2 * mean_edge_length(mesh)
    assert inflated["after"]["distinct"] == stats.distinct == len(points)
    assert inflated["after"]["span_mm"] == pytest.approx(stats.span_mm, abs=1e-3)
    # the insertion lies on the muscle: written as it came (rotated)
    assert "inflated" not in rec["areas"]["Ins"]
    centre, radius, _ = AREAS["adductor_longus_r"]["Ins"]
    np.testing.assert_allclose(
        exported_points(out, rec, "Ins"),
        rotate_points(ring(centre, radius), RAS_TO_OPENSIM),
        atol=1e-5,
    )


def test_inflation_and_times_in_metrics(default_run):
    index = read_export_index(default_run.ws.step_dir("export_mw2"))
    rec = {m["name"]: m for m in index["muscles"]}
    m = default_run.manifest.steps["export_mw2"].metrics
    assert m["inflated"] == {
        "adductor_longus_r": {"Ori": rec["adductor_longus_r"]["areas"]["Ori"]["inflated"]}
    }
    assert m["inflated"]["adductor_longus_r"]["Ori"]["after"]["distinct"] >= 3
    assert "adductor_longus_r origin: area inflated" in m["outline_warnings"]["adductor_longus_r"]
    assert m["coordinates_space"] == "opensim_yup"
    assert m["times_s"] == index["times_s"]
    assert set(m["times_s"]) == {
        "model_rotation",
        "bodies",
        "rest_pose",
        "muscles",
        "inflate",
        "total",
    }
    assert m["times_s"]["total"] == m["export_time_s"] == index["time_s"]
    assert all("inflate_time_s" not in r for r in index["muscles"])


def test_inflate_radius_from_config(tmp_path, spec):
    cfg = config(tmp_path, "export.muscles=[adductor_longus_r]", "export.inflate_radius_mm=8")
    index = read_export_index(run_pipeline(STEPS, spec, cfg).ws.step_dir("export_mw2"))
    assert index["muscles"][0]["areas"]["Ori"]["inflated"]["radius_mm"] == 8.0


def test_guard_error_keeps_the_outline(tmp_path, spec, monkeypatch):
    from mskpipe.geometry import attachment_guard

    def fail(*args, **kwargs):
        raise attachment_guard.GuardError("no disc-shaped patch")

    monkeypatch.setattr(attachment_guard, "inflate_outline", fail)
    cfg = config(tmp_path, "export.muscles=[adductor_longus_r]")
    out = run_pipeline(STEPS, spec, cfg).ws.step_dir("export_mw2")
    rec = read_export_index(out)["muscles"][0]
    assert rec["status"] == "ok"
    assert rec["warnings"] == [
        "origin collapsed",
        "adductor_longus_r origin: area not inflated (no disc-shaped patch)",
    ]
    assert "inflated" not in rec["areas"]["Ori"]
    centre, radius, _ = AREAS["adductor_longus_r"]["Ori"]
    np.testing.assert_allclose(
        exported_points(out, rec, "Ori"),
        rotate_points(ring(centre, radius), RAS_TO_OPENSIM),
        atol=1e-5,
    )


class IsbPelvisSkeleton(FakeSkeleton):
    """Pelvis frame with the ISB axes (x anterior, y superior, z right), written in RAS."""

    def run(self, ctx: StepContext) -> None:
        super().run(ctx)
        write_staple_model(ctx.out_dir / "bone_model.osim", pelvis_rotation=RAS_TO_OPENSIM.T)


def test_isb_pelvis_stands_upright_in_the_rest_pose(tmp_path, spec, default_run):
    steps = [IsbPelvisSkeleton() if s.name == "skeleton" else s for s in STEPS]
    cfg = config(tmp_path, "export.muscles=[iliacus_r]")
    out = run_pipeline(steps, spec, cfg).ws.step_dir("export_mw2")
    pose = read_export_index(out)["rest_pose"]
    values = pose["values"]
    for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation"):
        assert values[name] == pytest.approx(0.0, abs=1e-6)
    # the translation of the ground joint is the pelvis origin, in the rotated frame
    np.testing.assert_allclose(
        [values[f"pelvis_t{a}"] for a in "xyz"], RAS_TO_OPENSIM @ PELVIS_ORIGIN, atol=1e-6
    )
    assert pose["joints"]["ground_pelvis"]["rotation_deg"] < 1e-6
    assert all(e["max_displacement_mm"] < 1e-6 for e in pose["errors"].values())
    model = read_osim(out / "bone_model.osim")
    defaults = model.joints["ground_pelvis"].default_values
    assert max(abs(defaults[n]) for n in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")) < 1e-8
    # the fixture pelvis of the other tests (90 deg about z in RAS) is not upright
    lying = read_export_index(default_run.ws.step_dir("export_mw2"))["rest_pose"]["values"]
    assert abs(lying["pelvis_list"]) == pytest.approx(90.0, abs=1e-6)
