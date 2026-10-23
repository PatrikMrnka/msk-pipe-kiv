# SPDX-License-Identifier: Apache-2.0
"""Label maps over the slices of the image viewer (no Qt here).

A label layer is any integer NIfTI: the unified label map of a run
(``02_labelmap/labelmap.nii.gz`` with ``labels.json``), a raw tool output
(``01_segment/``, names from ``segmentation.json``) or the defect repairs
(``02_labelmap/patches.nii.gz``, names from ``defects.json``). It does not have to share
the grid of the image: every screen pixel takes the label voxel nearest to its world
position (exact voxels when the grids match), so no label map is resampled in memory.
"""

from __future__ import annotations

import colorsys
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "LabelInfo",
    "LabelLayer",
    "LayerSource",
    "compose",
    "label_at",
    "label_plane",
    "load_layer",
    "run_sources",
    "source_for_file",
    "to_canonical",
]

_SIDE = re.compile(r"_(l|r|left|right)$")
BONE_WORDS = (
    "bone", "clavicle", "femur", "fibula", "hip", "humerus", "ilium", "patella",
    "pelvis", "rib", "sacrum", "scapula", "skull", "sternum", "tibia", "vertebra",
)  # fmt: skip
BONE_COLORS = (  # light, low-saturation tones: bones read as bones, muscles stand out
    (241, 214, 171), (200, 220, 240), (232, 190, 160), (214, 206, 240),
    (196, 230, 205), (240, 236, 180), (236, 200, 214), (190, 212, 200),
)  # fmt: skip
MAX_LABEL = 65535


@dataclass(frozen=True)
class LayerSource:
    """A label map to offer in the viewer and how to name its values."""

    path: Path
    title: str
    names: dict[int, str] | None = None  # value -> structure name
    kinds: dict[int, str] | None = None  # value -> bone / muscle / patch


@dataclass(frozen=True)
class LabelInfo:
    value: int
    name: str
    kind: str  # bone, muscle, patch or other
    voxels: int
    volume_ml: float
    centroid: tuple[float, float, float]  # voxel of the label file (float)
    color: tuple[int, int, int]


@dataclass(frozen=True, eq=False)
class LabelLayer:
    source: LayerSource
    data: np.ndarray  # integer labels in the voxel order of the file
    affine: np.ndarray  # voxel of the file -> world RAS [mm]
    labels: dict[int, LabelInfo]  # present values, bones first, then by name
    slice_counts: dict[int, np.ndarray] = field(repr=False)  # value -> voxels per file k
    notes: tuple[str, ...] = ()

    @property
    def max_value(self) -> int:
        return max(self.labels, default=0)

    def same_grid(self, shape: Sequence[int], affine: np.ndarray) -> bool:
        return tuple(self.data.shape) == tuple(shape[:3]) and np.allclose(
            self.affine, affine, atol=1e-4
        )

    def locate(self, value: int) -> tuple[int, int, int]:
        """A voxel of ``value`` near its centroid (the centroid of a curved structure such
        as the pelvis may lie outside it)."""
        info = self.labels[value]
        counts = self.slice_counts[value]
        filled = np.flatnonzero(counts)
        k = int(filled[np.argmin(np.abs(filled - info.centroid[2]))])
        ii, jj = np.nonzero(self.data[:, :, k] == value)
        best = int(np.argmin((ii - info.centroid[0]) ** 2 + (jj - info.centroid[1]) ** 2))
        return int(ii[best]), int(jj[best]), k


# --------------------------------------------------------------------------- sources


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _table_names(table: Path) -> tuple[dict[int, str], dict[int, str]] | None:
    """Names and kinds from a ``labels.json`` (mskpipe.labels)."""
    from mskpipe.io.labels import LabelTable, LabelTableError

    try:
        t = LabelTable.load(table)
    except LabelTableError:
        return None
    return {lab.value: lab.name for lab in t}, {lab.value: str(lab.kind) for lab in t}


def _tool_title(tool: str, task: str) -> str:
    pretty = {"totalsegmentator": "TotalSegmentator", "musclemap": "MuscleMap"}.get(tool, tool)
    return f"{pretty} {task}".strip() if task and task != "default" else pretty


def _segment_sources(folder: Path, step: str) -> list[LayerSource]:
    from mskpipe.io.segmentation import (
        SEGMENTATION_FILE,
        SegmentationIndex,
        SegmentationIndexError,
    )

    try:
        index = SegmentationIndex.load(folder / SEGMENTATION_FILE)
    except SegmentationIndexError:
        return []
    out = []
    for o in index:
        path = folder / o.file
        if path.is_file():
            names = {v: n for n, v in o.labels.items()}
            out.append(LayerSource(path, f"{_tool_title(o.tool, o.task)} ({step})", names))
    return out


def _patch_source(path: Path, step: str) -> LayerSource:
    from mskpipe.qc.defects import DefectReport

    names: dict[int, str] = {}
    try:
        defects = DefectReport.read(path.parent).defects
    except (OSError, ValueError, TypeError):
        defects = []
    for i, d in enumerate(defects, start=1):
        names[i] = f"{d.kind} {d.structure}" + ("" if d.fixed else " (not fixed)")
    return LayerSource(path, f"Defect repairs ({step})", names, dict.fromkeys(names, "patch"))


def run_sources(run_dir: str | Path) -> tuple[Path | None, list[LayerSource]]:
    """Input volume of a run folder and its label maps (unified map first)."""
    from mskpipe.core.workspace import INPUT_FILE, STEP_DIRS
    from mskpipe.io.labels import LABELMAP_FILE, LABELS_FILE
    from mskpipe.qc.defects import PATCHES_FILE

    root = Path(run_dir)
    found = sorted((root / STEP_DIRS["prepare"]).glob("input.nii*"))
    image = found[0] if found else None
    if image is None:  # input copy deleted: the original, if it is still there
        record = _read_json(root / INPUT_FILE) or {}
        original = Path(record["image"]) if record.get("image") else None
        image = original if original is not None and original.is_file() else None

    sources: list[LayerSource] = []
    lm_dir = root / STEP_DIRS["labelmap"]
    labelmap = lm_dir / LABELMAP_FILE
    if labelmap.is_file():
        named = _table_names(lm_dir / LABELS_FILE)
        names, kinds = named if named else (None, None)
        sources.append(LayerSource(labelmap, f"Label map ({lm_dir.name})", names, kinds))
    seg_dir = root / STEP_DIRS["segment"]
    sources += _segment_sources(seg_dir, seg_dir.name)
    if (lm_dir / PATCHES_FILE).is_file():
        sources.append(_patch_source(lm_dir / PATCHES_FILE, lm_dir.name))
    return image, sources


def source_for_file(path: str | Path) -> LayerSource:
    """A label file chosen by the user, named from the run files next to it if possible."""
    from mskpipe.io.labels import LABELS_FILE
    from mskpipe.qc.defects import PATCHES_FILE

    path = Path(path).resolve()
    folder = path.parent
    if path.name == PATCHES_FILE:
        return _patch_source(path, folder.name)
    for src in _segment_sources(folder, folder.name):
        if src.path.resolve() == path:
            return src
    named = _table_names(folder / LABELS_FILE) if (folder / LABELS_FILE).is_file() else None
    names, kinds = named if named else (None, None)
    return LayerSource(path, path.name, names, kinds)


# --------------------------------------------------------------------------- loading


def _labels_array(img) -> tuple[np.ndarray, list[str]]:
    import nibabel as nib

    notes: list[str] = []
    proxy = img.dataobj
    slope = getattr(proxy, "slope", 1.0) if nib.is_proxy(proxy) else 1.0
    inter = getattr(proxy, "inter", 0.0) if nib.is_proxy(proxy) else 0.0
    unscaled = (math.isnan(slope) or slope == 1.0) and (math.isnan(inter) or inter == 0.0)
    if nib.is_proxy(proxy) and unscaled:
        data = np.asanyarray(proxy.get_unscaled())
    else:
        data = np.asanyarray(proxy)
    if data.ndim > 3:
        data = data[(slice(None),) * 3 + (0,) * (data.ndim - 3)]
        notes.append("first volume of a 4D label map")
    while data.ndim < 3:
        data = data[..., None]
    if data.dtype == bool:
        data = data.view(np.uint8)
    if not np.issubdtype(data.dtype, np.integer):
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError(f"not a label map: voxel type {data.dtype}")
        data = np.nan_to_num(np.rint(data), nan=0.0)
        if data.size and float(data.min()) < 0:
            raise ValueError("not a label map: negative values")
        top = float(data.max()) if data.size else 0.0
        data = data.astype(np.uint8 if top < 256 else np.uint16 if top <= MAX_LABEL else np.int64)
    if data.size and int(data.min()) < 0:
        raise ValueError("not a label map: negative values")
    if data.size and int(data.max()) > MAX_LABEL:
        raise ValueError(f"not a label map: values above {MAX_LABEL}")
    return data, notes


def _kind(name: str, kind: str | None) -> str:
    if kind:
        return kind
    base = _SIDE.sub("", name)
    return "bone" if any(word in base for word in BONE_WORDS) else "muscle"


def _colors(infos: Iterable[tuple[int, str, str]]) -> dict[int, tuple[int, int, int]]:
    """Bones: light tones; other structures: hues spread by the golden angle. Both sides
    of a structure share the hue (the left one is paler)."""
    infos = list(infos)
    bones = sorted({_SIDE.sub("", n) for _, n, k in infos if k == "bone"})
    others = sorted({_SIDE.sub("", n) for _, n, k in infos if k != "bone"})
    colors: dict[int, tuple[int, int, int]] = {}
    for value, name, kind in infos:
        base = _SIDE.sub("", name)
        left = name.endswith(("_l", "_left"))
        if kind == "bone":
            colors[value] = BONE_COLORS[bones.index(base) % len(BONE_COLORS)]
            continue
        hue = (others.index(base) * 0.618033988749895 + 0.02) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.45 if left else 0.85, 0.95)
        colors[value] = (round(r * 255), round(g * 255), round(b * 255))
    return colors


def load_layer(source: LayerSource) -> LabelLayer:
    """Read a label map and its per-structure voxel counts and centroids."""
    import nibabel as nib

    img = nib.load(str(source.path))
    data, notes = _labels_array(img)
    affine = np.asarray(img.affine, float)
    n_values = int(data.max()) + 1 if data.size else 1
    n_k = data.shape[2]
    grid_i, grid_j = (g.astype(np.float64).ravel() for g in np.indices(data.shape[:2]))
    total = np.zeros(n_values, np.int64)
    sum_i = np.zeros(n_values)
    sum_j = np.zeros(n_values)
    sum_k = np.zeros(n_values)
    per_k: list[tuple[int, np.ndarray, np.ndarray]] = []
    for k in range(n_k):  # one slice at a time: no voxel-sized temporary arrays
        flat = np.ascontiguousarray(data[:, :, k]).ravel()
        counts = np.bincount(flat, minlength=n_values)
        counts[0] = 0
        present = np.flatnonzero(counts)
        if present.size == 0:
            continue
        total += counts
        sum_i += np.bincount(flat, weights=grid_i, minlength=n_values)
        sum_j += np.bincount(flat, weights=grid_j, minlength=n_values)
        sum_k += counts * k
        per_k.append((k, present, counts[present]))

    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
    present = [int(v) for v in np.flatnonzero(total)]
    names = source.names or {}
    kinds = source.kinds or {}
    named = []
    for v in present:
        if v in names:
            named.append((v, names[v], _kind(names[v], kinds.get(v))))
        else:  # no name to guess the kind from
            named.append((v, f"label {v}", kinds.get(v, "other")))
    colors = _colors(named)
    order = sorted(named, key=lambda item: (item[2] != "bone", item[1]))
    labels = {
        v: LabelInfo(
            value=v,
            name=n,
            kind=k,
            voxels=int(total[v]),
            volume_ml=float(total[v]) * voxel_ml,
            centroid=(sum_i[v] / total[v], sum_j[v] / total[v], sum_k[v] / total[v]),
            color=colors[v],
        )
        for v, n, k in order
    }
    slice_counts = {v: np.zeros(n_k, np.int64) for v in present}
    for k, values, counts in per_k:
        for v, c in zip(values.tolist(), counts.tolist(), strict=True):
            slice_counts[v][k] = c
    unknown = [v for v in present if v not in names]
    if source.names is not None and unknown:
        notes.append(f"{len(unknown)} values without a name")
    return LabelLayer(source, data, affine, labels, slice_counts, tuple(notes))


# --------------------------------------------------------------------------- slices


def _mapping(volume_affine: np.ndarray, layer: LabelLayer) -> np.ndarray:
    """Canonical voxel of the image -> voxel of the label file."""
    return np.linalg.solve(layer.affine, volume_affine)


def label_plane(
    layer: LabelLayer,
    volume_affine: np.ndarray,
    volume_shape: Sequence[int],
    axes: tuple[int, int, int],
    index: int,
) -> np.ndarray:
    """Labels under the screen pixels of one view (rows top to bottom, as
    :func:`mskpipe.gui.slice_viewer.plane`); 0 outside the label map."""
    s_axis, h_axis, v_axis = axes
    m = _mapping(volume_affine, layer)
    cols = np.arange(volume_shape[h_axis] - 1, -1, -1, dtype=np.float64)
    rows = np.arange(volume_shape[v_axis] - 1, -1, -1, dtype=np.float64)
    base = m[:3, 3] + m[:3, s_axis] * index
    shape = layer.data.shape
    take = []
    inside = np.ones((rows.size, cols.size), bool)
    for a in range(3):
        coord = np.rint(base[a] + m[a, h_axis] * cols[None, :] + m[a, v_axis] * rows[:, None])
        idx = coord.astype(np.intp)
        inside &= (idx >= 0) & (idx < shape[a])
        take.append(idx)
    out = np.zeros((rows.size, cols.size), layer.data.dtype)
    out[inside] = layer.data[take[0][inside], take[1][inside], take[2][inside]]
    return out


def label_at(layer: LabelLayer, volume_affine: np.ndarray, ijk: Sequence[int]) -> int:
    m = _mapping(volume_affine, layer)
    idx = np.rint(m @ np.r_[np.asarray(ijk, float), 1.0])[:3].astype(int)
    if np.any(idx < 0) or np.any(idx >= layer.data.shape):
        return 0
    return int(layer.data[tuple(idx)])


def to_canonical(
    layer: LabelLayer, volume_affine: np.ndarray, volume_shape: Sequence[int], file_ijk
) -> list[int]:
    """Voxel of the label file -> nearest canonical voxel of the image (clamped)."""
    idx = np.linalg.solve(volume_affine, layer.affine @ np.r_[np.asarray(file_ijk, float), 1.0])
    return [max(0, min(int(np.rint(c)), n - 1)) for c, n in zip(idx[:3], volume_shape, strict=True)]


def compose(
    gray: np.ndarray,
    labels: np.ndarray,
    lut: np.ndarray,
    alpha: np.ndarray,
    opacity: float,
    outline: bool,
) -> np.ndarray:
    """Grey slice with the labels blended in (RGB uint8, C-contiguous).

    ``lut``: (n, 3) uint8 colour per label value, ``alpha``: (n,) 1 = shown, 0 = hidden
    (value 0 must be 0). ``outline`` draws only the label borders, opaque.
    """
    a = alpha[labels]
    if outline:
        edge = np.zeros(labels.shape, bool)
        edge[1:, :] |= labels[1:, :] != labels[:-1, :]
        edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
        edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        a = a * edge
    else:
        a = a * np.float32(opacity)
    a = a[..., None]
    rgb = gray[..., None].astype(np.float32) * (1.0 - a) + lut[labels].astype(np.float32) * a
    return np.ascontiguousarray(np.clip(rgb + 0.5, 0, 255).astype(np.uint8))
