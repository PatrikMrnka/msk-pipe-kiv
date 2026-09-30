# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from mskpipe.io.mesh_io import MeshIOError, TriMesh, read_mesh, write_mesh


def tetra() -> TriMesh:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) + 100.25
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return TriMesh(vertices, faces)


def same_geometry(a: TriMesh, b: TriMesh) -> bool:
    tri_a = np.sort(a.vertices[a.faces].reshape(len(a.faces), -1), axis=0)
    tri_b = np.sort(b.vertices[b.faces].reshape(len(b.faces), -1), axis=0)
    return a.n_faces == b.n_faces and np.allclose(tri_a, tri_b, atol=1e-5)


@pytest.mark.parametrize("suffix", [".stl", ".obj", ".STL"])
def test_roundtrip(tmp_path, suffix):
    mesh = tetra()
    path = write_mesh(tmp_path / f"m{suffix}", mesh)
    back = read_mesh(path)
    assert back.n_vertices == 4  # STL vertices are merged back into an indexed mesh
    assert same_geometry(mesh, back)


def test_non_ascii_path(tmp_path):
    path = tmp_path / "Příliš žluťoučký kůň" / "sval_ř.obj"
    write_mesh(path, tetra())
    assert same_geometry(read_mesh(path), tetra())


def test_obj_polygons_and_index_forms(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text(
        "# quad\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\nf 1/1/1 2//1 -2 4\n",
        encoding="utf-8",
    )
    mesh = read_mesh(path)
    assert mesh.faces.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_ascii_stl(tmp_path):
    path = tmp_path / "a.stl"
    path.write_text(
        "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n"
        "vertex 0 1 0\nendloop\nendfacet\nendsolid t\n",
        encoding="utf-8",
    )
    mesh = read_mesh(path)
    assert (mesh.n_vertices, mesh.n_faces) == (3, 1)


def test_errors(tmp_path):
    with pytest.raises(MeshIOError, match="Unsupported"):
        write_mesh(tmp_path / "m.ply", tetra())
    bad = tmp_path / "bad.stl"
    bad.write_bytes(b"\x00" * 90)
    with pytest.raises(MeshIOError, match="Not a valid STL"):
        read_mesh(bad)
    with pytest.raises(MeshIOError, match="out of range"):
        TriMesh(np.zeros((3, 3)), np.array([[0, 1, 3]]))


def test_transformed_keeps_orientation_under_mirroring():
    from mskpipe.geometry.metrics import signed_volume

    mesh = tetra()
    mirror = np.diag([-1.0, 1.0, 1.0, 1.0])
    assert signed_volume(mesh) == pytest.approx(1 / 6)
    assert signed_volume(mesh.transformed(mirror)) == pytest.approx(1 / 6)
