# SPDX-License-Identifier: Apache-2.0
"""Tunnels through muscles in the assembled label map (genus 0 for Muscle Wrapping).

Muscle Wrapping 2.x needs every muscle surface to be one closed part of genus 0. A muscle
of the label map can get a tunnel in two ways, and each needs a different repair:

* **another muscle passes through it** - MuscleMap labels a strip of one muscle through a
  thin wall of its neighbour (LHDL CT: a 108 mm strip of vastus intermedius through the
  vastus lateralis; msk-pipe does not model the vastus intermedius, so there the strip is
  background). The tunnel is filled: its voxels are *transferred* to the muscle,
  background voxels added. Candidates are voxels enclosed by the muscle in a 2D slice
  along x, y or z (the tunnel is closed in some slice, a notch is not); voxels of bone are
  never taken. A transfer is accepted only if the genus drops and the donor keeps its
  topology (no new tunnel, still one part; pieces it loses contact with go to background).
* **the muscle is a ring around something it may not take** (bone, a strip of another
  muscle that is not enclosed in any slice) - the ring is *cut* at its thinnest section:
  candidates are the parts an opening of growing radius removes (up to
  ``max_cut_radius_mm``), each trimmed to the thinnest slab (2-6 voxels along its own
  principal axes and x, y, z) that still removes the tunnel. Cut voxels become background.

Global closings and openings, used by the surface repair of step ``mesh``, fail on both:
a closing wide enough for a 5 mm thick wall creates new handles, an opening that cuts it
erodes the whole muscle. Here every change is local and checked on the genus of the Flying
Edges surface (:mod:`mskpipe.labelmap.topology`, the surface step ``mesh`` builds).

The changed voxels of one muscle (added + removed) may not exceed ``max_volume_change`` of
its volume; a tunnel that needs more is left to step ``mesh`` and reported there.
Muscles whose surface has several parts (genus undefined) are skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from mskpipe.labelmap.topology import STRUCT26, euler_number, surface_topology
from mskpipe.qc.defects import Defect

CUT_RADII_MM: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
SLAB_WIDTHS: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0)  # voxels
MARGIN = 3  # voxels around a muscle box


@dataclass
class MuscleTunnelRepair:
    """Result for one muscle (``labelmap.json`` entry ``topology``)."""

    structure: str
    genus_before: int | None
    genus_after: int | None
    filled: int = 0  # tunnels filled by a transfer
    cut: int = 0  # rings cut
    added_voxels: int = 0
    removed_voxels: int = 0
    taken_from: dict[str, int] = field(default_factory=dict)  # donor -> voxels
    defects: list[Defect] = field(default_factory=list)
    patches: list[np.ndarray | None] = field(default_factory=list)  # full grid, per defect

    def summary(self) -> dict[str, Any]:
        return {
            "genus_before": self.genus_before,
            "genus_after": self.genus_after,
            "tunnels_filled": self.filled,
            "rings_cut": self.cut,
            "added_voxels": self.added_voxels,
            "removed_voxels": self.removed_voxels,
            "taken_from": dict(self.taken_from),
        }


def _box(mask: np.ndarray, margin: int = MARGIN) -> tuple[slice, ...] | None:
    found = ndimage.find_objects(mask.astype(np.int8))
    if not found or found[0] is None:
        return None
    return _grow(found[0], margin, mask.shape)


def _grow(box: tuple[slice, ...], n: int, shape: Sequence[int]) -> tuple[slice, ...]:
    return tuple(
        slice(max(s.start - n, 0), min(s.stop + n, d)) for s, d in zip(box, shape, strict=True)
    )


def _points_box(pts: np.ndarray) -> tuple[slice, ...]:
    lo, hi = pts.min(axis=0), pts.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))


def _union(a: tuple[slice, ...], b: tuple[slice, ...]) -> tuple[slice, ...]:
    return tuple(
        slice(min(x.start, y.start), max(x.stop, y.stop)) for x, y in zip(a, b, strict=True)
    )


def _topology(mask: np.ndarray) -> tuple[int, int]:
    """(outer surfaces, genus) of a cropped mask."""
    outer, _cavities, handles = surface_topology(mask)
    return outer, handles


def _delta(
    mask: np.ndarray, change: np.ndarray, add: bool, box: tuple[slice, ...] | None = None
) -> int:
    """Exact change of the Euler number when ``change`` is added to / removed from ``mask``
    (computed on ``box``, the box of ``change``, grown by one voxel)."""
    box = _box(change, 0) if box is None else box
    if box is None:
        return 0
    box = _grow(box, 1, mask.shape)
    before = mask[box]
    after = before | change[box] if add else before & ~change[box]
    return euler_number(after) - euler_number(before)


def enclosed_in_slices(mask: np.ndarray) -> np.ndarray:
    """Voxels outside ``mask`` but enclosed by it in a 2D slice along x, y or z."""
    out = np.zeros_like(mask, dtype=bool)
    for axis in range(3):
        m = np.moveaxis(mask, axis, 0)
        o = np.moveaxis(out, axis, 0)
        for i in np.flatnonzero(m.any(axis=(1, 2))):
            o[i] |= ndimage.binary_fill_holes(m[i]) & ~m[i]
    return out


def _center(patch: np.ndarray, origin: Sequence[int], affine: np.ndarray):
    ijk = np.argwhere(patch).mean(axis=0) + np.asarray(origin)
    xyz = affine[:3, :3] @ ijk + affine[:3, 3]
    return [round(float(v), 2) for v in ijk], [round(float(v), 2) for v in xyz]


def _extent_mm(patch: np.ndarray, spacing: np.ndarray) -> float:
    pts = np.argwhere(patch) * spacing
    if len(pts) < 2:
        return float(spacing.max())
    c = pts - pts.mean(axis=0)
    ext = np.ptp(c @ np.linalg.svd(c, full_matrices=False)[2].T, axis=0)
    return float(ext.max() + spacing.min())


class _Muscles:
    """The label map with the muscle values; boxes are found on demand."""

    def __init__(self, data: np.ndarray, values: dict[str, int], protected: Sequence[int]):
        self.data = data
        self.values = values
        self.names = {v: k for k, v in values.items()}
        self.movable = np.zeros(256, dtype=bool)  # values a transfer may take
        self.movable[0] = True
        for v in values.values():
            self.movable[v] = True
        for v in protected:
            self.movable[v] = False
        self._boxes = ndimage.find_objects(data)  # one pass; boxes only shrink later

    def box(self, value: int, margin: int = MARGIN) -> tuple[slice, ...] | None:
        found = self._boxes[value - 1] if 0 < value <= len(self._boxes) else None
        return None if found is None else _grow(found, margin, self.data.shape)


def _try_transfer(
    lm: _Muscles, value: int, box: tuple[slice, ...], m: np.ndarray, comp: np.ndarray,
    comp_box: tuple[slice, ...], g: int,
) -> tuple[int, dict[int, np.ndarray]] | None:  # fmt: skip
    """Genus after adding ``comp`` (box coordinates) and the donor voxels lost to
    background, or None if the muscle or a donor would get worse."""
    if _delta(m, comp, add=True, box=comp_box) <= 0:
        return None  # no tunnel closed (a notch, or a new handle)
    outer, g2 = _topology(m | comp)
    if outer != 1 or g2 >= g:
        return None
    sub = lm.data[box]
    orphans: dict[int, np.ndarray] = {}
    for donor in np.unique(sub[comp]):
        donor = int(donor)
        if donor in (0, value):
            continue
        dbox = lm.box(donor)
        if dbox is None:
            continue
        dbox = _union(dbox, box)
        taken = np.zeros(lm.data.shape, dtype=bool)
        taken[box] = comp & (sub == donor)
        before = lm.data[dbox] == donor
        after = before & ~taken[dbox]
        labels, n = ndimage.label(after, structure=STRUCT26)
        lost = np.zeros_like(after)
        if n > 1:
            sizes = np.bincount(labels.ravel())[1:]
            lost = (labels > 0) & (labels != int(np.argmax(sizes)) + 1)
            after = after & ~lost
        if not after.any():
            return None
        outer_b, g_b = _topology(before)
        outer_a, g_a = _topology(after)
        if outer_a > max(outer_b, 1) or g_a > g_b:
            return None
        if lost.any():
            full_lost = np.zeros(lm.data.shape, dtype=bool)
            full_lost[dbox] = lost
            orphans[donor] = full_lost
    return g2, orphans


def _slab(m: np.ndarray, piece: np.ndarray, g: int) -> np.ndarray:
    """Thinnest slab of ``piece`` whose removal still lowers the genus (else ``piece``)."""
    pts = np.argwhere(piece)
    if len(pts) < 4:
        return piece
    centred = pts - pts.mean(axis=0)
    dirs = [*np.linalg.svd(centred.astype(float), full_matrices=False)[2], *np.eye(3)]
    for width in SLAB_WIDTHS:
        for direction in dirs:
            t = centred @ direction
            bins = np.floor((t - t.min()) / width).astype(int)
            counts = np.bincount(bins)
            for b in np.argsort(counts, kind="stable"):
                if counts[b] == 0:
                    continue
                sel = pts[bins == b]
                slab = np.zeros_like(piece)
                slab[tuple(sel.T)] = True
                if _delta(m, slab, add=False, box=_points_box(sel)) <= 0:
                    continue
                outer, g2 = _topology(m & ~slab)
                if outer == 1 and g2 < g:
                    return slab
    return piece


def _find_cut(
    m: np.ndarray, g: int, spacing: np.ndarray, max_radius_mm: float, budget: int
) -> np.ndarray | None:
    from mskpipe.labelmap.clean import element_radius, structuring_element

    for r in (r for r in CUT_RADII_MM if r <= max_radius_mm + 1e-9):
        pad = int(np.max(element_radius(r, spacing))) + 1
        opened = ndimage.binary_opening(np.pad(m, pad), structure=structuring_element(r, spacing))
        removed = m & ~opened[pad:-pad, pad:-pad, pad:-pad]
        labels, n = ndimage.label(removed, structure=STRUCT26)
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel())[1:]
        boxes = ndimage.find_objects(labels)
        for idx in np.argsort(sizes, kind="stable"):
            piece = np.zeros_like(m)
            piece[boxes[idx]] = labels[boxes[idx]] == idx + 1
            if _delta(m, piece, add=False, box=boxes[idx]) <= 0:
                continue
            outer, g2 = _topology(m & ~piece)
            if outer != 1 or g2 >= g:
                continue
            cut = _slab(m, piece, g)
            if cut.sum() <= budget:
                return cut
    return None


def repair_muscle_tunnels(
    data: np.ndarray,
    affine: np.ndarray,
    muscles: dict[str, int],
    protected: Sequence[int],
    *,
    max_volume_change: float = 0.05,
    max_cut_radius_mm: float = 5.0,
    log: Callable[..., Any] | None = None,
) -> list[MuscleTunnelRepair]:
    """Remove tunnels of the ``muscles`` (name -> value) in ``data`` (changed in place).

    ``protected`` values (bones) are never taken. Returns one result per muscle that had a
    tunnel; defects carry full-grid voxel coordinates.
    """
    affine = np.asarray(affine, dtype=float)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    lm = _Muscles(data, muscles, protected)
    results: list[MuscleTunnelRepair] = []
    for name, value in muscles.items():
        box = lm.box(value)
        if box is None:
            continue
        m = data[box] == value
        outer, g = _topology(m)
        if outer != 1 or g == 0:
            continue
        res = MuscleTunnelRepair(name, g, g)
        budget = int(max_volume_change * m.sum())
        origin = [s.start for s in box]

        # 1. fill: tunnels closed in a slice (voxels of other muscles transferred)
        while g > 0:
            sub = data[box]
            cand = enclosed_in_slices(m) & lm.movable[sub]
            labels, n = ndimage.label(cand, structure=STRUCT26)
            if n == 0:
                break
            sizes = np.bincount(labels.ravel())[1:]
            boxes = ndimage.find_objects(labels)
            done = False
            for idx in np.argsort(sizes, kind="stable"):
                used = res.added_voxels + res.removed_voxels
                if used + sizes[idx] > budget:
                    break
                comp = np.zeros_like(m)
                comp[boxes[idx]] = labels[boxes[idx]] == idx + 1
                trial = _try_transfer(lm, value, box, m, comp, boxes[idx], g)
                if trial is None:
                    continue
                g2, orphans = trial
                values, counts = np.unique(sub[comp], return_counts=True)
                taken = {
                    lm.names[int(v)]: int(c)
                    for v, c in zip(values, counts, strict=True)
                    if v not in (0, value)
                }
                patch = np.zeros(data.shape, dtype=bool)
                patch[box] = comp
                sub[comp] = value
                for lost in orphans.values():
                    data[lost] = 0
                m = sub == value
                for k, v in taken.items():
                    res.taken_from[k] = res.taken_from.get(k, 0) + v
                res.added_voxels += int(comp.sum())
                res.filled += 1
                ijk, xyz = _center(comp, origin, affine)
                parts = [f"{v} voxels from {k}" for k, v in taken.items()]
                parts.append(f"{int(comp.sum()) - sum(taken.values())} from background")
                parts += [
                    f"{int(lost.sum())} detached voxels of {lm.names[d]} set to background"
                    for d, lost in orphans.items()
                ]
                res.defects.append(
                    Defect(
                        structure=name,
                        kind="hole",
                        center_ijk=ijk,
                        center_mm=xyz,
                        size_mm=round(_extent_mm(comp, spacing), 1),
                        fixed=True,
                        method="voxel transfer",
                        voxels=int(comp.sum()),
                        detail=f"tunnel through the muscle filled, genus {g} -> {g2}; "
                        + ", ".join(parts),
                    )
                )
                res.patches.append(patch)
                g, done = g2, True
                break
            if not done:
                break

        # 2. cut: rings around voxels that cannot be taken
        while g > 0:
            budget_left = budget - res.added_voxels - res.removed_voxels
            cut = _find_cut(m, g, spacing, max_cut_radius_mm, budget_left)
            if cut is None:
                break
            outer, g2 = _topology(m & ~cut)
            sub = data[box]
            sub[cut] = 0
            m = sub == value
            res.removed_voxels += int(cut.sum())
            res.cut += 1
            ijk, xyz = _center(cut, origin, affine)
            res.defects.append(
                Defect(
                    structure=name,
                    kind="hole",
                    center_ijk=ijk,
                    center_mm=xyz,
                    size_mm=round(_extent_mm(cut, spacing), 1),
                    fixed=True,
                    method="ring cut",
                    voxels=0,
                    detail=f"ring of the muscle cut at its thinnest section, genus {g} -> {g2}; "
                    f"{int(cut.sum())} voxels set to background",
                )
            )
            res.patches.append(None)
            g = g2

        res.genus_after = g
        if log is not None:
            if g == 0:
                log("info", "%s: genus %d -> 0 (%d tunnel(s) filled, %d ring(s) cut; "
                    "+%d / -%d voxels)", name, res.genus_before, res.filled, res.cut,
                    res.added_voxels, res.removed_voxels)  # fmt: skip
            else:
                log("warning", "%s: genus %d left in the label map (limit %.0f %% of the volume"
                    " or no tunnel found); step mesh tries to repair the surface", name, g,
                    100 * max_volume_change)  # fmt: skip
        results.append(res)
    return results


__all__ = ["MuscleTunnelRepair", "enclosed_in_slices", "repair_muscle_tunnels"]
