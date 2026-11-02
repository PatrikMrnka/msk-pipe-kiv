# SPDX-License-Identifier: Apache-2.0
"""Method ``manual``: user-supplied outlines (files, names, frames, order, snapping)."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_attachments import SUBJECT_T, FakeMesh, blob

from mskpipe.config import InputSpec, load_config
from mskpipe.core.registry import default_registry
from mskpipe.core.runner import PipelineError, run_pipeline
from mskpipe.geometry.compare import closest_on_surface
from mskpipe.geometry.outline import angular_order, signed_turn
from mskpipe.io.attachment_vtk import read_points, write_points
from mskpipe.io.manual_areas import (
    MAPPING_FILE,
    default_stem,
    folder_fingerprint,
    guess_area,
    match_folder,
    read_outline,
    write_mapping,
)
from mskpipe.io.mesh_io import read_mesh
from mskpipe.plugins.attachments.manual import BodySurface, Manual, place_outline
from mskpipe.steps.attachments import AttachmentsStep, read_attachments_index

MUSCLES = ["adductor_brevis", "adductor_longus", "gluteus_medius", "gracilis", "iliacus"]


def ring(center, radius=10.0, n=10, normal_axis=1):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = radius * np.cos(a), radius * np.sin(a)
    cols = [c, s, np.zeros(n)]
    cols.insert(normal_axis, cols.pop())
    return np.asarray(center, float) + np.c_[cols[0], cols[1], cols[2]]


# --------------------------------------------------------------------------- names


@pytest.mark.parametrize(
    ("rel", "muscle", "kind", "stem", "rule"),
    [
        (
            "adductor_longus_r/AdductorLongus_Ins.vtk",
            "adductor_longus_r",
            "Ins",
            "AdductorLongus",
            "layout",
        ),
        (
            "adductor_longus_l/whatever_origin.txt",
            "adductor_longus_l",
            "Ori",
            "AdductorLongus",
            "layout",
        ),
        ("AdductorLongus_R_Ins.vtk", "adductor_longus_r", "Ins", "AdductorLongus", "name"),
        ("gracilis_l_Ori.csv", "gracilis_l", "Ori", "Gracilis", "name"),
        ("Insertion gracilis.txt", "gracilis_r", "Ins", "Gracilis", "name"),
        ("Ori_1_L_M102_Adduct_Brev.vtk", "adductor_brevis_l", "Ori", "AdductorBrevis", "code"),
    ],
)
def test_guess_area(rel, muscle, kind, stem, rule):
    area = guess_area(rel, "r", MUSCLES, {"m102": "adductor_brevis"})
    assert (area.muscle, area.kind, area.stem, area.rule) == (muscle, kind, stem, rule)


@pytest.mark.parametrize(
    "rel", ["notes.txt", "gracilis.vtk", "gracilis_ori_ins.vtk", "M999_Ori.vtk"]
)
def test_guess_area_unknown(rel):
    assert guess_area(rel, "r", MUSCLES, {"m102": "adductor_brevis"}) is None


def test_default_stem():
    assert default_stem("adductor_longus_r") == "AdductorLongus"
    assert default_stem("gracilis") == "Gracilis"


def test_match_folder_duplicates_and_unmatched(tmp_path):
    pts = ring((0, 0, 0))
    write_points(tmp_path / "gracilis_r" / "Gracilis_Ori.vtk", pts)
    write_points(tmp_path / "gracilis_r_origin.vtk", pts)
    write_points(tmp_path / "gracilis_r" / "Gracilis_Ins.vtk", pts)
    (tmp_path / "readme.txt").write_text("hello\n")
    m = match_folder(tmp_path, "r", MUSCLES, codes={})
    assert list(m.areas) == [("gracilis_r", "Ins")]
    assert set(m.duplicates) == {("gracilis_r", "Ori")}
    assert [p.name for p in m.unmatched] == ["readme.txt"]
    assert "more than one file" in m.problems()[0]


def test_mapping_csv_overrides_names(tmp_path):
    pts = ring((0, 0, 0))
    write_points(tmp_path / "a.vtk", pts)
    np.savetxt(tmp_path / "b.txt", pts)
    write_points(tmp_path / "gracilis_r_Ori.vtk", pts)  # not listed: ignored
    (tmp_path / MAPPING_FILE).write_text(
        "file,muscle,kind\na.vtk,iliacus,origin\nb.txt,iliacus_l,Ins\nc.vtk,iliacus_r,Ins\n"
        "a.vtk,biceps,Ori\n",
        encoding="utf-8",
    )
    m = match_folder(tmp_path, "r", MUSCLES, codes={})
    assert set(m.areas) == {("iliacus_r", "Ori"), ("iliacus_l", "Ins")}
    assert all(a.rule == "csv" for a in m.areas.values())
    assert [p.name for p in m.unmatched] == ["gracilis_r_Ori.vtk"]
    assert any("file not found: c.vtk" in e for e in m.errors)
    assert any("unknown muscle 'biceps'" in e for e in m.errors)
    # written back and read again
    write_mapping(tmp_path, m.areas.values())
    again = match_folder(tmp_path, "r", MUSCLES, codes={})
    assert set(again.areas) == set(m.areas) and not again.errors


def test_fingerprint_follows_content(tmp_path):
    write_points(tmp_path / "gracilis_r_Ori.vtk", ring((0, 0, 0)))
    first = folder_fingerprint(tmp_path)
    write_points(tmp_path / "gracilis_r_Ori.vtk", ring((0, 0, 1)))
    assert folder_fingerprint(tmp_path) != first
    assert folder_fingerprint(tmp_path / "missing") == "missing"


# --------------------------------------------------------------------------- files


def test_read_text_and_vtk_formats(tmp_path):
    pts = ring((1, 2, 3), n=6)
    (tmp_path / "a.txt").write_text(
        "# exported\nname x y z\n"
        + "".join(f"P{i} {x} {y} {z}\n" for i, (x, y, z) in enumerate(pts))
    )
    (tmp_path / "b.csv").write_text("x;y;z\n" + "".join(f"{x};{y};{z}\n" for x, y, z in pts))
    for name in ("a.txt", "b.csv"):
        assert np.allclose(read_outline(tmp_path / name), pts)
    # binary legacy VTK with a closed polyline in another order than the points
    import vtk

    order = [3, 4, 5, 0, 1, 2]
    poly = vtk.vtkPolyData()
    vp = vtk.vtkPoints()
    for p in pts:
        vp.InsertNextPoint(*p)
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(7)
    for i, idx in enumerate([*order, order[0]]):
        line.GetPointIds().SetId(i, idx)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    poly.SetPoints(vp)
    poly.SetLines(cells)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(tmp_path / "c.vtk"))
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    writer.Write()
    assert np.allclose(read_outline(tmp_path / "c.vtk"), pts[order], atol=1e-5)


def test_read_outline_needs_three_points(tmp_path):
    (tmp_path / "a.txt").write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="at least 3 points"):
        read_outline(tmp_path / "a.txt")


# --------------------------------------------------------------------------- geometry


def test_angular_order_is_clockwise_from_outside():
    pts = ring((0, 0, 0), n=8, normal_axis=2)  # plane z = 0
    shuffled = pts[[0, 5, 2, 7, 1, 3, 6, 4]]
    out = np.array([0.0, 0.0, 1.0])
    ordered = shuffled[angular_order(shuffled, out)]
    assert np.allclose(ordered[0], shuffled[0])  # starts at the first point
    assert signed_turn(ordered, out) < 0  # clockwise seen from +z
    steps = np.linalg.norm(np.diff(np.vstack([ordered, ordered[:1]]), axis=0), axis=1)
    assert np.allclose(steps, steps[0])  # neighbours on the circle


def test_place_outline_picks_nearest_body_and_snaps():
    pelvis = blob((0, 0, 0), (60, 40, 40))
    femur = blob((150, 0, 0), (20, 20, 60))
    surfaces = {"pelvis": BodySurface(pelvis), "femur_r": BodySurface(femur)}
    on_femur = closest_on_surface(ring((150, -25, 0), 8.0), femur)[0]
    lifted = on_femur + np.array([0, -3.0, 0])  # 3 mm off the bone
    final, rec = place_outline(lifted, surfaces, max_projection_mm=5.0)
    assert rec["body"] == "femur_r" and rec["snapped"] == 1.0
    assert closest_on_surface(final, femur)[1].max() < 1e-6
    _, rec = place_outline(lifted, surfaces, max_projection_mm=1.0)
    assert rec["snapped"] == 0.0 and rec["to_bone_median_mm"] == pytest.approx(3.0, abs=0.5)


# --------------------------------------------------------------------------- step


@pytest.fixture
def spec(tmp_path: Path) -> InputSpec:
    image = tmp_path / "subject.nii.gz"
    image.write_bytes(b"fake volume")
    return InputSpec(image=image, modality="ct")


def subject_mesh(tmp_path: Path, name: str):
    """Mesh of FakeMesh written once to find outlines on it."""
    from test_attachments import ATLAS_BONES

    from mskpipe.io.mesh_io import write_mesh

    c, r = ATLAS_BONES[name]
    mesh = blob(c, r, bumps=1.0).transformed(SUBJECT_T)
    return mesh, write_mesh(tmp_path / f"{name}.stl", mesh)


@pytest.fixture
def areas(tmp_path: Path) -> Path:
    folder = tmp_path / "manual"
    pelvis, _ = subject_mesh(tmp_path, "pelvis_no_sacrum")
    femur, _ = subject_mesh(tmp_path, "femur_r")
    ori = closest_on_surface(
        ring(pelvis.vertices.mean(0) + np.array([40.0, -60.0, 0.0]), 12.0), pelvis
    )[0]
    ins = closest_on_surface(
        ring(femur.vertices.mean(0) + np.array([0.0, -30.0, 60.0]), 8.0), femur
    )[0]
    write_points(folder / "iliacus_r" / "Iliacus_Ori.vtk", ori)
    np.savetxt(folder / "iliacus_r_insertion.txt", ins[::-1])
    write_points(folder / "gracilis_r_Ori.vtk", ori)  # origin only
    write_points(folder / "adductor_longus_r_Ori.vtk", ori)  # not meshed
    return folder


def manual_config(tmp_path: Path, folder: Path | None, *extra: str):
    sets = [
        f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}",
        "attachments.method=manual",
        *extra,
    ]
    if folder is not None:
        sets.append(f"attachments.params.areas_dir={folder.as_posix()}")
    return load_config(overrides=sets)


def test_registered_and_documented():
    reg = default_registry()
    assert reg.get("attachments", "manual") is Manual
    assert Manual.missing() == []


def test_step_uses_manual_outlines(tmp_path, spec, areas):
    cfg = manual_config(tmp_path, areas)
    assert Manual.preflight(cfg, "ct") == []
    warnings = Manual.preflight_warnings(cfg, "ct")
    assert any("adductor_longus_r, gracilis_r" in w for w in warnings)

    result = run_pipeline([FakeMesh(), AttachmentsStep()], spec, cfg)
    out = result.ws.step_dir("attachments")
    index = read_attachments_index(out)
    assert index["method"] == "manual"
    assert index["muscles"] == {
        "gracilis_r": ["gracilis_r/Gracilis_Ori.vtk"],
        "iliacus_r": ["iliacus_r/Iliacus_Ins.vtk", "iliacus_r/Iliacus_Ori.vtk"],
    }
    assert index["not_provided"] == []
    assert index["not_meshed"] == ["adductor_longus_r"]
    by_key = {(a["muscle"], a["kind"]): a for a in index["areas"]}
    assert by_key[("iliacus_r", "Ori")]["body"] == "pelvis"
    assert by_key[("iliacus_r", "Ins")]["body"] == "femur_r"
    assert by_key[("iliacus_r", "Ins")]["match"] == "name"
    assert by_key[("gracilis_r", "Ins")]["status"] == "not_provided"
    # file order kept, points on the bone
    ins = read_points(out / "iliacus_r/Iliacus_Ins.vtk")
    src = np.loadtxt(areas / "iliacus_r_insertion.txt")
    assert np.linalg.norm(ins - src, axis=1).max() < 1e-3
    rec = result.manifest.steps["attachments"]
    assert rec.metrics["n_areas"] == 3


def test_step_rejects_other_frame(tmp_path, spec, areas):
    moved = tmp_path / "moved"
    for name in ("iliacus_r/Iliacus_Ori.vtk", "gracilis_r_Ori.vtk"):
        write_points(moved / name, read_points(areas / name) + np.array([0.0, 250.0, 0.0]))
    cfg = manual_config(tmp_path, moved)
    with pytest.raises(PipelineError) as info:
        run_pipeline([FakeMesh(), AttachmentsStep()], spec, cfg)
    assert "another coordinate frame" in str(info.value.__cause__)


def test_preflight_errors(tmp_path):
    assert "areas_dir is not set" in Manual.preflight(manual_config(tmp_path, None), "ct")[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "No attachment outlines" in Manual.preflight(manual_config(tmp_path, empty), "ct")[0]
    (empty / "gracilis_r_Ori.txt").write_text("1 2 3\n")
    assert "at least 3 points" in Manual.preflight(manual_config(tmp_path, empty), "ct")[0]


def test_fingerprint_follows_folder(tmp_path, areas):
    cfg = manual_config(tmp_path, areas)
    first = Manual.cache_identity(cfg)
    write_points(areas / "gracilis_r_Ori.vtk", read_points(areas / "gracilis_r_Ori.vtk") + 1.0)
    assert Manual.cache_identity(cfg) != first


def test_transfer_json_lists_sources(tmp_path, spec, areas):
    result = run_pipeline([FakeMesh(), AttachmentsStep()], spec, manual_config(tmp_path, areas))
    transfer = json.loads((result.ws.step_dir("attachments") / "transfer.json").read_text())
    assert transfer["source"]["dir"] == str(areas)
    assert transfer["bodies"]["pelvis"] == ["pelvis_no_sacrum"]
    assert transfer["bodies"]["tibia_r"] == ["tibia_r"]  # merged by the step
    assert read_mesh(result.ws.step_dir("attachments") / "bodies/tibia_r.stl").n_vertices > 0


def test_lps_to_ras():
    from mskpipe.io.manual_areas import to_ras

    assert np.allclose(to_ras(np.array([[1.0, 2.0, 3.0]]), "lps"), [[-1.0, -2.0, 3.0]])
    with pytest.raises(ValueError):
        to_ras(np.zeros((1, 3)), "xyz")


def test_voxel_frame_uses_the_input_header():
    from mskpipe.io.manual_areas import to_ras

    affine = np.array([[-0.5, 0, 0, 10.0], [0, -0.5, 0, 20.0], [0, 0, 2.0, -100.0], [0, 0, 0, 1.0]])
    voxel_mm = np.array([[1.0, 2.0, 10.0]])  # index (2, 4, 5) x voxel size
    assert np.allclose(to_ras(voxel_mm, "voxel", affine), [[9.0, 18.0, -90.0]])
    with pytest.raises(ValueError, match="input image"):
        to_ras(voxel_mm, "voxel")


def test_step_voxel_frame(tmp_path, areas):
    import nibabel as nib

    from mskpipe.io.manual_areas import image_affine

    affine = np.array([[-1.0, 0, 0, 30.0], [0, 1.0, 0, -40.0], [0, 0, 1.0, -500.0], [0, 0, 0, 1.0]])
    image = tmp_path / "subject.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), affine), str(image))
    assert np.allclose(image_affine(image), affine)
    inv = np.linalg.inv(affine)
    voxel = tmp_path / "voxel"
    for name in ("iliacus_r/Iliacus_Ori.vtk", "gracilis_r_Ori.vtk"):
        ras = read_points(areas / name)
        write_points(voxel / name, ras @ inv[:3, :3].T + inv[:3, 3])  # spacing 1
    spec = InputSpec(image=image, modality="ct")
    cfg = manual_config(tmp_path, voxel, "attachments.params.frame=voxel")
    result = run_pipeline([FakeMesh(), AttachmentsStep()], spec, cfg)
    out = result.ws.step_dir("attachments")
    got = read_points(out / "iliacus_r/Iliacus_Ori.vtk")
    assert np.abs(got - read_points(areas / "iliacus_r/Iliacus_Ori.vtk")).max() < 1e-3
