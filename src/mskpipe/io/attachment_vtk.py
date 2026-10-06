# SPDX-License-Identifier: Apache-2.0
"""Attachment areas as legacy ASCII VTK point sets, the format Muscle Wrapping 2.x reads.

An area is a closed outline: the order of the points matters and is kept. Layout written
(same as the LHDL files)::

    # vtk DataFile Version 3.0 / vtk output / ASCII / DATASET POLYDATA
    POINTS n float          coordinates in mm
    VERTICES n 2n           one vertex cell per point
    POINT_DATA n            SCALARS scalars bit, all 1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class AttachmentFileError(ValueError):
    """The file is not a readable legacy VTK point set."""


def read_points(path: str | Path) -> np.ndarray:
    """Points (N, 3) of a legacy ASCII VTK file, in file order."""
    path = Path(path)
    tokens = path.read_text(encoding="ascii", errors="replace").split()
    try:
        i = tokens.index("POINTS")
        n = int(tokens[i + 1])
        values = np.asarray(tokens[i + 3 : i + 3 + 3 * n], dtype=np.float64)
    except (ValueError, IndexError) as exc:
        raise AttachmentFileError(f"{path}: no ASCII POINTS section") from exc
    if values.size != 3 * n:
        raise AttachmentFileError(f"{path}: expected {n} points")
    return values.reshape(n, 3)


def write_points(path: str | Path, points: np.ndarray) -> Path:
    """Write ``points`` (mm) in the Muscle Wrapping 2.x layout; returns the path."""
    path = Path(path)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    lines = ["# vtk DataFile Version 3.0", "vtk output", "ASCII", "DATASET POLYDATA"]
    lines.append(f"POINTS {n} float")
    flat = [f"{v:.9g}" for v in pts.ravel()]
    lines += [" ".join(flat[i : i + 9]) + " " for i in range(0, len(flat), 9)]
    lines.append(f"VERTICES {n} {2 * n}")
    lines += [f"1 {i} " for i in range(n)]
    lines += ["", f"POINT_DATA {n}", "SCALARS scalars bit", "LOOKUP_TABLE default"]
    ones = ["1"] * n
    lines += [" ".join(ones[i : i + 8]) for i in range(0, n, 8)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


__all__ = ["AttachmentFileError", "read_points", "write_points"]
