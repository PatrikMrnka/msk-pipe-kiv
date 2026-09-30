# SPDX-License-Identifier: Apache-2.0
"""Mesh quality metrics (NumPy only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mskpipe.io.mesh_io import TriMesh


def surface_area(mesh: TriMesh) -> float:
    tri = mesh.vertices[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def signed_volume(mesh: TriMesh) -> float:
    """Enclosed volume; positive for a closed mesh with outward-facing triangles."""
    tri = mesh.vertices[mesh.faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def edge_topology(mesh: TriMesh) -> dict[str, int]:
    """Counts of boundary, non-manifold and inconsistently oriented edges."""
    directed = mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    undirected = np.sort(directed, axis=1)
    _, counts = np.unique(undirected, axis=0, return_counts=True)
    _, directed_counts = np.unique(directed, axis=0, return_counts=True)
    return {
        "boundary_edges": int((counts == 1).sum()),
        "non_manifold_edges": int((counts > 2).sum()),
        "misoriented_edges": int((directed_counts > 1).sum()),
    }


def mesh_stats(mesh: TriMesh) -> dict[str, Any]:
    """Size, area, volume and topology of a mesh (lengths in the mesh units).

    ``volume`` is the divergence-theorem volume; it is exact only for ``closed`` meshes
    (small defects, e.g. a few non-manifold edges after decimation, change it negligibly).
    """
    topo = edge_topology(mesh) if mesh.n_faces else {}
    closed = bool(mesh.n_faces) and all(v == 0 for v in topo.values())
    return {
        "n_vertices": mesh.n_vertices,
        "n_faces": mesh.n_faces,
        "area": surface_area(mesh),
        "volume": signed_volume(mesh),
        "closed": closed,
        **topo,
    }
