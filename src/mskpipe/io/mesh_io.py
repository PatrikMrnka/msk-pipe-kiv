# SPDX-License-Identifier: Apache-2.0
"""Triangle meshes in STL and OBJ.

Readers and writers are implemented with NumPy only: no VTK, full control over the
written precision, and no trouble with non-ASCII paths on Windows. Only geometry is
handled (vertices and triangles); normals, texture coordinates and groups in OBJ files
are ignored, polygons are fan-triangulated.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUFFIXES = (".stl", ".obj")
_STL_HEADER = b"mskpipe binary STL"
_STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


class MeshIOError(ValueError):
    """A mesh file cannot be read or written."""


@dataclass(frozen=True, eq=False)
class TriMesh:
    """Indexed triangle mesh: ``vertices`` (N, 3) float64, ``faces`` (M, 3) int64."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise MeshIOError(f"vertices must have shape (N, 3), got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise MeshIOError(f"faces must have shape (M, 3), got {faces.shape}")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise MeshIOError("face indices out of range")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def transformed(self, matrix: np.ndarray) -> TriMesh:
        """Apply a 4x4 affine; triangle winding is reversed for mirroring transforms."""
        matrix = np.asarray(matrix, dtype=np.float64)
        vertices = self.vertices @ matrix[:3, :3].T + matrix[:3, 3]
        faces = self.faces[:, ::-1] if np.linalg.det(matrix[:3, :3]) < 0 else self.faces
        return TriMesh(vertices, faces)


# ------------------------------------------------------------------------------ public API


def read_mesh(path: str | Path) -> TriMesh:
    """Read an ``.stl`` (binary or ASCII) or ``.obj`` file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return _read_stl(path)
    if suffix == ".obj":
        return _read_obj(path)
    raise MeshIOError(f"Unsupported mesh format '{path.suffix}' (expected {SUFFIXES})")


def write_mesh(path: str | Path, mesh: TriMesh) -> Path:
    """Write ``mesh`` as binary STL or OBJ, chosen by the file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".stl":
        _write_stl(path, mesh)
    elif suffix == ".obj":
        _write_obj(path, mesh)
    else:
        raise MeshIOError(f"Unsupported mesh format '{path.suffix}' (expected {SUFFIXES})")
    return path


# ------------------------------------------------------------------------------ STL


def _write_stl(path: Path, mesh: TriMesh) -> None:
    tri = mesh.vertices[mesh.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    records = np.zeros(mesh.n_faces, dtype=_STL_DTYPE)
    records["normal"] = normals
    records["vertices"] = tri
    with path.open("wb") as fh:
        fh.write(_STL_HEADER.ljust(80, b" "))
        fh.write(struct.pack("<I", mesh.n_faces))
        fh.write(records.tobytes())


def _read_stl(path: Path) -> TriMesh:
    data = path.read_bytes()
    if len(data) >= 84:
        (count,) = struct.unpack_from("<I", data, 80)
        if len(data) == 84 + count * _STL_DTYPE.itemsize:
            records = np.frombuffer(data, dtype=_STL_DTYPE, count=count, offset=84)
            return _merge_vertices(records["vertices"].reshape(-1, 3))
    if data.lstrip()[:5].lower() == b"solid":
        return _read_stl_ascii(data, path)
    raise MeshIOError(f"Not a valid STL file: {path}")


def _read_stl_ascii(data: bytes, path: Path) -> TriMesh:
    coords = [
        line.split()[1:4]
        for line in data.decode("ascii", errors="replace").splitlines()
        if line.strip().startswith("vertex")
    ]
    if not coords or len(coords) % 3:
        raise MeshIOError(f"Malformed ASCII STL: {path}")
    try:
        points = np.asarray(coords, dtype=np.float32)
    except ValueError as exc:
        raise MeshIOError(f"Malformed ASCII STL: {path}") from exc
    return _merge_vertices(points)


def _merge_vertices(points: np.ndarray) -> TriMesh:
    """STL stores three vertices per triangle; merge identical ones into an indexed mesh."""
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    return TriMesh(unique, inverse.reshape(-1, 3))


# ------------------------------------------------------------------------------ OBJ


def _write_obj(path: Path, mesh: TriMesh) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# mskpipe\n# {mesh.n_vertices} vertices, {mesh.n_faces} triangles\n")
        np.savetxt(fh, mesh.vertices, fmt="v %.9g %.9g %.9g")
        np.savetxt(fh, mesh.faces + 1, fmt="f %d %d %d")


def _read_obj(path: Path) -> TriMesh:
    vertices: list[list[str]] = []
    faces: list[list[int]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            parts = line.split()
            if not parts:
                continue
            try:
                if parts[0] == "v":
                    vertices.append(parts[1:4])
                elif parts[0] == "f":
                    idx = [_obj_index(p, len(vertices)) for p in parts[1:]]
                    faces.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))
            except (ValueError, IndexError) as exc:
                raise MeshIOError(f"Malformed OBJ {path} at line {lineno}") from exc
    try:
        points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    except ValueError as exc:
        raise MeshIOError(f"Malformed OBJ vertices in {path}") from exc
    return TriMesh(points, np.asarray(faces, dtype=np.int64).reshape(-1, 3))


def _obj_index(token: str, n_vertices: int) -> int:
    """``7``, ``7/1`` or ``7//3`` -> 0-based vertex index; negative indices are relative."""
    value = int(token.split("/", 1)[0])
    if value == 0:
        raise ValueError("OBJ indices start at 1")
    return value - 1 if value > 0 else n_vertices + value
