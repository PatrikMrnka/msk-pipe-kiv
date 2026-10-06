# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from mskpipe.config import InputSpec, load_config
from mskpipe.core.runner import PipelineError, run_pipeline
from mskpipe.core.step import Step, StepContext
from mskpipe.geometry.compare import closest_on_surface
from mskpipe.geometry.fast_rnrr import (
    FAST_RNRR_ENV,
    FastRnrrError,
    MeshWarp,
    find_executable,
    run_fast_rnrr,
)
from mskpipe.geometry.register import (
    apply,
    fit_coherent_icp,
    fit_cpd,
    sample_surface,
    similarity_icp,
    umeyama,
)
from mskpipe.io.attachment_vtk import read_points, write_points
from mskpipe.io.mesh_io import TriMesh, read_mesh, write_mesh
from mskpipe.plugins.attachments.bone_registration import ATLAS_ENV, atlas_fingerprint
from mskpipe.steps.attachments import AttachmentsStep, read_attachments_index


def blob(center, radii, bumps=0.0) -> TriMesh:
    src = vtk.vtkSphereSource()
    src.SetThetaResolution(60)
    src.SetPhiResolution(40)
    src.Update()
    poly = src.GetOutput()
    v = vtk_to_numpy(poly.GetPoints().GetData()).astype(float) * 2.0  # unit radius
    f = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    # asymmetric shape: waves + one knob, so that the registration is unambiguous
    knob = np.exp(-np.sum((v - np.array([0.6, 0.6, 0.5])) ** 2, axis=1) / 0.1)
    shape = 1.0 + bumps * (
        0.08 * np.sin(3 * v[:, 0] + 0.5) * np.cos(2 * v[:, 1] + 0.3) + 0.3 * knob
    )
    v = v * shape[:, None] * np.asarray(radii)
    return TriMesh(v + np.asarray(center), f)


def similarity(deg: float, scale: float, t) -> np.ndarray:
    a = np.radians(deg)
    r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    m = np.eye(4)
    m[:3, :3] = scale * r
    m[:3, 3] = t
    return m


ATLAS_BONES = {
    "pelvis_no_sacrum": ((0, 0, 0), (120, 60, 80)),
    "femur_r": ((80, 0, -250), (25, 25, 200)),
    "tibia_r": ((80, 0, -650), (25, 25, 180)),
}
SACRUM = ((0, -95, 10), (25, 20, 40))
SUBJECT_T = similarity(4.0, 1.05, (10.0, -5.0, 20.0))


def outline(mesh: TriMesh, around, radius=15.0, n=12) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.asarray(around) + radius * np.c_[np.cos(ang), np.zeros(n), np.sin(ang)]
    return closest_on_surface(ring, mesh)[0]


@pytest.fixture
def atlas(tmp_path: Path) -> Path:
    folder = tmp_path / "atlas"
    meshes = {k: blob(c, r, bumps=1.0) for k, (c, r) in ATLAS_BONES.items()}
    for name, mesh in meshes.items():
        write_mesh(folder / "bones" / f"{name}.obj", mesh)
    ori = outline(meshes["pelvis_no_sacrum"], (60, -60, 0))
    ins = outline(meshes["femur_r"], (80, -25, -100), radius=8.0)
    write_points(folder / "muscles/iliacus_r/Iliacus_Ori.vtk", ori)
    write_points(folder / "muscles/iliacus_r/Iliacus_Ins.vtk", ins)
    index = {
        "frame": "ras",
        "units": "mm",
        "muscles": {
            "iliacus_r": {
                "stem": "Iliacus",
                "Ori": {
                    "file": "muscles/iliacus_r/Iliacus_Ori.vtk",
                    "bone": "pelvis_with_sacrum",
                    "body": "pelvis",
                },
                "Ins": {
                    "file": "muscles/iliacus_r/Iliacus_Ins.vtk",
                    "bone": "femur_r",
                    "body": "femur_r",
                },
            },
            "obturator_externus_r": {"stem": "ObturatorExternus", "Ori": {"status": "missing"}},
        },
    }
    (folder / "atlas.json").write_text(json.dumps(index), encoding="utf-8")
    return folder


class FakeMesh(Step):
    """Subject = atlas bones under SUBJECT_T, muscles iliacus_r and gracilis_r."""

    name = "mesh"
    sacrum = False

    def run(self, ctx: StepContext) -> None:
        structures = []
        bones = {**ATLAS_BONES, **({"sacrum": SACRUM} if self.sacrum else {})}
        muscles = ("iliacus_r", "gracilis_r", "iliacus_l", *(("piriformis_r",) * self.sacrum))
        for name, (c, r) in bones.items():
            parts = {"tibia_r": [("tibia_r", c, r), ("fibula_r", (115, 0, -650), (8, 8, 170))]}
            for part, pc, pr in parts.get(name, [(name, c, r)]):
                mesh = blob(pc, pr, bumps=1.0).transformed(SUBJECT_T)
                rel = f"bones/{part}.stl"
                write_mesh(ctx.out_dir / rel, mesh)
                structures.append({"name": part, "kind": "bone", "status": "ok", "file": rel})
        for m in muscles:
            rel = f"muscles/{m}.obj"
            write_mesh(ctx.out_dir / rel, blob((0, 0, 0), (5, 5, 5)))
            structures.append({"name": m, "kind": "muscle", "status": "ok", "file": rel})
        index = {"format": "mskpipe.meshes", "version": 1, "structures": structures}
        (ctx.out_dir / "meshes.json").write_text(json.dumps(index), encoding="utf-8")


@pytest.fixture
def spec(tmp_path: Path) -> InputSpec:
    image = tmp_path / "subject.nii.gz"
    image.write_bytes(b"fake volume")
    return InputSpec(image=image, modality="ct")


def config(tmp_path: Path, atlas: Path | None, *overrides: str):
    sets = [
        f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}",
        "attachments.params.n_points=1500",
        "attachments.params.nonrigid_points=500",
        *overrides,
    ]
    if atlas is not None:
        sets.append(f"attachments.params.atlas_dir={atlas.as_posix()}")
    return load_config(overrides=sets)


# --------------------------------------------------------------------------- io + geometry


def test_points_roundtrip_keeps_order(tmp_path):
    pts = np.random.default_rng(0).normal(size=(17, 3)) * 100
    path = write_points(tmp_path / "a.vtk", pts)
    text = path.read_text(encoding="ascii")
    assert "POINTS 17 float" in text and "VERTICES 17 34" in text and "SCALARS scalars bit" in text
    np.testing.assert_allclose(read_points(path), pts, atol=1e-6)


def test_umeyama_recovers_similarity():
    src = np.random.default_rng(1).normal(size=(50, 3))
    m = similarity(30.0, 1.2, (1, 2, 3))
    np.testing.assert_allclose(umeyama(src, apply(m, src)), m, atol=1e-9)


def test_point_to_plane_icp_beats_sparse_point_to_point():
    """Elongated bone, target sampled only ~7x denser than the source: point-to-point is
    biased by the sampling noise, the point-to-plane refinement is not."""
    rng = np.random.default_rng(3)
    atlas = blob((0, 0, 0), (25, 25, 200), bumps=1.0)
    subject = atlas.transformed(SUBJECT_T)
    src, src_n = sample_surface(atlas, 3000, rng, normals=True)
    dst, dst_n = sample_surface(subject, 20000, rng, normals=True)
    init = np.eye(4)
    init[:3, 3] = dst.mean(0) - src.mean(0)
    p2p = similarity_icp(src, dst, init=init)
    p2pl = similarity_icp(src, dst, init=p2p.matrix, src_normals=src_n, dst_normals=dst_n)

    def error(m):
        diff = apply(m, atlas.vertices) - subject.vertices
        return np.linalg.norm(diff, axis=1).mean()

    assert p2pl.iterations < 20
    assert p2pl.scale == pytest.approx(1.05, abs=0.005)
    assert error(p2pl.matrix) < 0.35 < error(p2p.matrix)


def test_point_to_plane_warp_keeps_correspondence():
    """Bent bone: both warps reach the surface, the point-to-plane one also keeps the
    points where they belong (no tangential sliding)."""
    rng = np.random.default_rng(0)
    atlas = blob((0, 0, 0), (25, 25, 200), bumps=1.0)

    def bend(p):
        return p + np.c_[np.sin(p[:, 2] / 80.0) * 3.0, np.zeros(len(p)), np.zeros(len(p))]

    truth = bend(apply(SUBJECT_T, atlas.vertices))
    subject = TriMesh(truth, atlas.faces)
    src, src_n = sample_surface(atlas, 3000, rng, normals=True)
    dst, dst_n = sample_surface(subject, 50000, rng, normals=True)
    init = np.eye(4)
    init[:3, 3] = dst.mean(0) - src.mean(0)
    rigid = similarity_icp(src, dst, init=init)
    rigid = similarity_icp(src, dst, init=rigid.matrix, src_normals=src_n, dst_normals=dst_n)
    assert rigid.scale == pytest.approx(1.05, abs=0.02)
    moved = apply(rigid.matrix, src)[:1500]
    verts = apply(rigid.matrix, atlas.vertices)
    plain, _ = fit_coherent_icp(moved, dst, beta=40.0, lam=2.0)
    plane, _ = fit_coherent_icp(moved, dst, dst_normals=dst_n, beta=40.0, lam=2.0)

    def corr(points):
        return np.linalg.norm(points - truth, axis=1).mean()

    before = closest_on_surface(verts, subject)[1].mean()
    after = closest_on_surface(plane(verts), subject)[1].mean()
    assert after < before and after < 0.3
    assert corr(plane(verts)) < min(0.6, corr(plain(verts)), corr(verts) / 2)


# ---------------------------------------------------------------------------------- step


def test_step_transfers_outlines(tmp_path, spec, atlas):
    result = run_pipeline([FakeMesh(), AttachmentsStep()], spec, config(tmp_path, atlas))
    out = result.ws.step_dir("attachments")
    index = read_attachments_index(out)
    assert index["muscles"] == {
        "iliacus_r": ["iliacus_r/Iliacus_Ins.vtk", "iliacus_r/Iliacus_Ori.vtk"]
    }
    assert index["not_in_atlas"] == ["gracilis_r"]
    assert index["not_meshed"] == ["obturator_externus_r"]
    assert set(index["bones"]) == {"pelvis", "femur"} and index["bones"]["pelvis"]["status"] == "ok"
    assert index["bones"]["femur"]["scale"] == pytest.approx(1.05, abs=0.02)

    # expected = atlas outline mapped by the true transform
    for kind in ("Ori", "Ins"):
        truth = apply(SUBJECT_T, read_points(atlas / f"muscles/iliacus_r/Iliacus_{kind}.vtk"))
        got = read_points(out / f"iliacus_r/Iliacus_{kind}.vtk")
        assert got.shape == truth.shape
        assert np.linalg.norm(got - truth, axis=1).mean() < 0.5
    assert (out / "registration/femur_r.stl").is_file()
    assert (out / "bodies/tibia_r.stl").is_file()

    rec = result.manifest.steps["attachments"]
    assert rec.status == "completed"
    assert rec.metrics["n_muscles"] == 1 and rec.metrics["n_areas"] == 2
    assert rec.metrics["min_snapped"] == 1.0
    assert "05_attachments/attachments.json" in rec.outputs


def test_missing_atlas_is_step_error(tmp_path, spec, monkeypatch):
    monkeypatch.delenv(ATLAS_ENV, raising=False)
    with pytest.raises(PipelineError) as info:
        run_pipeline([FakeMesh(), AttachmentsStep()], spec, config(tmp_path, None))
    assert "No atlas" in str(info.value.__cause__)


def test_atlas_from_environment(tmp_path, spec, atlas, monkeypatch):
    monkeypatch.setenv(ATLAS_ENV, str(atlas))
    result = run_pipeline([FakeMesh(), AttachmentsStep()], spec, config(tmp_path, None))
    assert result.manifest.steps["attachments"].metrics["n_areas"] == 2


def test_fingerprint_follows_atlas_content(atlas):
    cfg = load_config(overrides=[f"attachments.params.atlas_dir={atlas.as_posix()}"])
    step = AttachmentsStep()
    first = step.fingerprint_extra(cfg)
    assert first["plugin"]["plugin"] == "attachments/bone_registration"
    assert first["atlas_sha256"] == atlas_fingerprint(atlas) != "missing"
    write_points(atlas / "muscles/iliacus_r/Iliacus_Ins.vtk", np.zeros((3, 3)))
    assert step.fingerprint_extra(cfg)["atlas_sha256"] != first["atlas_sha256"]


# ------------------------------------------------------------------- non-rigid methods


def transfer_error(result, atlas) -> float:
    out = result.ws.step_dir("attachments")
    errors = []
    for kind in ("Ori", "Ins"):
        truth = apply(SUBJECT_T, read_points(atlas / f"muscles/iliacus_r/Iliacus_{kind}.vtk"))
        got = read_points(out / f"iliacus_r/Iliacus_{kind}.vtk")
        errors.append(np.linalg.norm(got - truth, axis=1).mean())
    return float(max(errors))


@pytest.mark.parametrize(
    ("method", "extra", "limit"),
    [
        ("none", [], 1.0),
        ("cpd", ["attachments.params.cpd_sigma_init_mm=5"], 3.0),
    ],
)
def test_step_nonrigid_methods(tmp_path, spec, atlas, method, extra, limit):
    sets = [f"attachments.params.nonrigid={method}", *extra]
    result = run_pipeline([FakeMesh(), AttachmentsStep()], spec, config(tmp_path, atlas, *sets))
    index = read_attachments_index(result.ws.step_dir("attachments"))
    assert {b["nonrigid_method"] for b in index["bones"].values()} == {method}
    assert transfer_error(result, atlas) < limit


def test_cpd_standard_start_drifts_on_elongated_bone():
    """Documents a property of CPD used in the paper: with the standard start (sigma^2 from
    all pairs) the points of an unchanged femur slide along it by centimetres; a small
    initial sigma keeps them within the sampling noise."""
    rng = np.random.default_rng(0)
    bone = blob((0, 0, 0), (25, 25, 200), bumps=1.0)
    src, dst = sample_surface(bone, 800, rng), sample_surface(bone, 2500, rng)
    standard = fit_cpd(src, dst, iterations=60)
    small = fit_cpd(src, dst, iterations=60, sigma_init=5.0)

    def drift(res):
        return np.linalg.norm(res.warp.displacement(src), axis=1).mean()

    assert drift(small) < 3.0 < 10.0 < drift(standard)
    scale = np.sqrt(np.mean(np.sum((dst - dst.mean(0)) ** 2, axis=1)))
    assert small.beta == pytest.approx(2.0 * scale)


def test_fast_rnrr_wrapper_and_mesh_warp(tmp_path):
    source = blob((0, 0, 0), (25, 25, 200), bumps=1.0)
    target = source.transformed(SUBJECT_T)
    affine = similarity(2.0, 1.02, (1.0, 2.0, 3.0))
    seen = {}

    def fake_run(argv):  # stands in for the executable: affine motion + one appended vertex
        src_path, tar_path, prefix = (Path(a) for a in argv[1:])
        seen["argv"] = argv
        mesh = read_mesh(src_path)
        moved = np.vstack([apply(affine, mesh.vertices), [[0.0, 0.0, 0.0]]])
        write_mesh(Path(f"{prefix}res.obj"), TriMesh(moved, mesh.faces))
        seen["target_centred"] = np.allclose(read_mesh(tar_path).vertices.mean(0), 0, atol=1e-6)

    shift = target.vertices.mean(0)
    deformed = run_fast_rnrr(Path("Fast_RNRR"), source, target, tmp_path, fake_run)
    assert seen["target_centred"] and str(seen["argv"][0]) == "Fast_RNRR"
    expected = apply(affine, source.vertices - shift) + shift
    np.testing.assert_allclose(deformed.vertices, expected, atol=1e-5)

    # barycentric transport reproduces an affine field for points on and near the surface
    field = MeshWarp(source, deformed.vertices)
    pts = sample_surface(source, 200, np.random.default_rng(4))
    want = apply(affine, pts - shift) + shift
    np.testing.assert_allclose(field(pts), want, atol=1e-4)


def test_fast_rnrr_missing_executable(monkeypatch):
    monkeypatch.delenv(FAST_RNRR_ENV, raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FastRnrrError, match="not distributed"):
        find_executable(None)


def test_step_fast_rnrr(tmp_path, spec, atlas, monkeypatch):
    """Plugin wiring with a stand-in for the executable (identity deformation)."""
    import mskpipe.geometry.fast_rnrr as fr

    exe = tmp_path / "Fast_RNRR.exe"
    exe.write_bytes(b"fake")

    def fake(executable, source, target, work_dir, run):
        assert executable == exe and work_dir.parts[-2] == "fast_rnrr"
        return source

    monkeypatch.setattr(fr, "run_fast_rnrr", fake)
    cfg = config(
        tmp_path,
        atlas,
        "attachments.params.nonrigid=fast_rnrr",
        f"attachments.params.fast_rnrr_exe={exe.as_posix()}",
    )
    result = run_pipeline([FakeMesh(), AttachmentsStep()], spec, cfg)
    index = read_attachments_index(result.ws.step_dir("attachments"))
    assert index["bones"]["femur"]["nonrigid_info"]["vertices"] > 0
    assert transfer_error(result, atlas) < 1.0
    fp = AttachmentsStep().fingerprint_extra(cfg)
    assert fp["fast_rnrr_sha256"] == hashlib.sha256(b"fake").hexdigest()


class FakeMeshSacrum(FakeMesh):
    sacrum = True


@pytest.fixture
def atlas_sacrum(atlas: Path) -> Path:
    """Atlas with pelvis_with_sacrum.obj and a piriformis origin on the sacrum."""
    pelvis = blob(*ATLAS_BONES["pelvis_no_sacrum"], bumps=1.0)
    sacrum = blob(*SACRUM, bumps=1.0)
    merged = TriMesh(
        np.vstack([pelvis.vertices, sacrum.vertices]),
        np.vstack([pelvis.faces, sacrum.faces + pelvis.n_vertices]),
    )
    write_mesh(atlas / "bones/pelvis_with_sacrum.obj", merged)
    ori = outline(sacrum, (0, -115, 10), radius=8.0)
    write_points(atlas / "muscles/piriformis_r/Piriformis_Ori.vtk", ori)
    index = json.loads((atlas / "atlas.json").read_text(encoding="utf-8"))
    index["muscles"]["piriformis_r"] = {
        "stem": "Piriformis",
        "Ori": {
            "file": "muscles/piriformis_r/Piriformis_Ori.vtk",
            "bone": "pelvis_with_sacrum",
            "body": "pelvis",
        },
    }
    (atlas / "atlas.json").write_text(json.dumps(index), encoding="utf-8")
    return atlas


def test_sacrum_joins_the_pelvis(tmp_path, spec, atlas_sacrum):
    """Subject sacrum + hip bones are registered as one pelvis to the atlas pelvis with
    sacrum, and sacral areas land on the subject's sacrum."""
    cfg = config(tmp_path, atlas_sacrum)
    result = run_pipeline([FakeMeshSacrum(), AttachmentsStep()], spec, cfg)
    out = result.ws.step_dir("attachments")
    index = read_attachments_index(out)
    assert index["body_geometry_sources"]["pelvis_with_sacrum"] == ["pelvis_no_sacrum", "sacrum"]
    assert index["bones"]["pelvis"]["atlas"] == "pelvis_with_sacrum.obj"
    assert index["bones"]["pelvis"]["subject"] == "pelvis_with_sacrum.stl"
    area = next(a for a in index["areas"] if a["muscle"] == "piriformis_r" and a["kind"] == "Ori")
    assert area["status"] == "ok" and area["snapped"] == 1.0
    truth = apply(SUBJECT_T, read_points(atlas_sacrum / "muscles/piriformis_r/Piriformis_Ori.vtk"))
    got = read_points(out / "piriformis_r/Piriformis_Ori.vtk")
    assert np.linalg.norm(got - truth, axis=1).mean() < 0.5


def test_subject_without_sacrum_prefers_atlas_without_sacrum(tmp_path, spec, atlas_sacrum):
    result = run_pipeline([FakeMesh(), AttachmentsStep()], spec, config(tmp_path, atlas_sacrum))
    index = read_attachments_index(result.ws.step_dir("attachments"))
    assert index["bones"]["pelvis"]["atlas"] == "pelvis_no_sacrum.obj"
    assert "pelvis_with_sacrum" not in index["body_geometry_sources"]
