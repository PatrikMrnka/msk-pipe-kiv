# SPDX-License-Identifier: Apache-2.0
"""Topology repair of bone masks: through-holes (handles) and gaps (broken bones).

Thin bone (iliac wing, fibular shaft) is often segmented with defects, above all in MRI:

* **hole** - a tunnel through a thin plate. The surface gets an extra handle (genus): the
  hip bone has genus 1 (obturator foramen), every wing hole adds one. Muscle Wrapping and
  STAPLE need the anatomical topology.
* **gap** - a long bone split into fragments (part of the shaft missing). Keeping only the
  largest component then silently shortens the bone.

Holes are found and filled on the voxel mask with the Euler characteristic of the voxel
set (26-connected foreground, closed unit cubes; ``chi = components - tunnels +
cavities``). Candidate fillers are (1) voxels removed by the opening of the cleaning,
(2) voxels added by closings of growing radius. A candidate is accepted only if it removes
a tunnel (``chi`` rises; computed locally, chi is additive), smallest first, until the
expected genus is reached. Concavities filled by the closing do not change ``chi`` and are
never added, so notches and the acetabulum keep their shape; the obturator foramen is
protected because filling stops at the expected genus and smaller holes come first.

Gaps are bridged by a tapered cylinder between the facing ends of consecutive fragments
along the principal axis of the bone (``max_gap_mm``). The bridge is interpolated, not
measured: it is always reported.

Every defect is returned as a :class:`~mskpipe.qc.defects.Defect` (location in voxel and
world coordinates) for the QC report and the 3D viewer.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from mskpipe.qc.defects import Defect

STRUCT26 = np.ones((3, 3, 3), dtype=bool)

# Anatomical genus of a bone part (closed surface). Only these parts are checked: the
# sacrum has foramina whose number in a segmentation varies (LHDL CT: genus 7), so it is not.
EXPECTED_GENUS: dict[str, int] = {
    "hip_left": 1,  # obturator foramen
    "hip_right": 1,
    "ilium_l": 1,  # MuscleMap hip bone
    "ilium_r": 1,
    "femur_left": 0,
    "femur_right": 0,
    "femur_l": 0,
    "femur_r": 0,
    "tibia": 0,
    "tibia_l": 0,
    "tibia_r": 0,
    "fibula": 0,
    "fibula_l": 0,
    "fibula_r": 0,
    "patella": 0,
}
# Long bones whose fragments are bridged (source label names of all tools).
LONG_BONES: frozenset[str] = frozenset(
    {"femur_left", "femur_right", "femur_l", "femur_r", "tibia", "tibia_l", "tibia_r",
     "fibula", "fibula_l", "fibula_r"}
)  # fmt: skip
# The same bone in another tool's segmentation (holes are filled from it first).
DONORS: dict[str, str] = {
    "hip_left": "ilium_l",
    "hip_right": "ilium_r",
    "ilium_l": "hip_left",
    "ilium_r": "hip_right",
}
DONOR_DEPTH_MM: tuple[float, ...] = (4.0, 3.0, 2.0)  # mm, see _donor_trim
DONOR_TRIM_MM: tuple[float, ...] = (8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0)
DEFAULT_CLOSING_MM: tuple[float, ...] = (1.5, 3.0, 4.5, 6.0, 8.0, 10.0, 12.0)
LOCATE_ONLY_MM: tuple[float, ...] = (16.0, 20.0, 25.0, 30.0)  # holes too wide to fill


# --------------------------------------------------------------------------- Euler number


def euler_number(mask: np.ndarray) -> int:
    """Euler characteristic of the union of closed voxel cubes (26-connected foreground).

    ``V - E + F - C`` of the cubical complex; a solid ball gives 1, a solid ring 0.
    """
    p = np.pad(np.asarray(mask, dtype=bool), 1)
    a, b = slice(None, -1), slice(1, None)
    full = slice(None)
    verts = np.zeros(tuple(n - 1 for n in p.shape), dtype=bool)
    for i in (a, b):
        for j in (a, b):
            for k in (a, b):
                verts |= p[i, j, k]
    edges = 0
    faces = 0
    for axis in range(3):
        e = None
        f = None
        for s1 in (a, b):
            for s2 in (a, b):
                idx = [s1, s2]
                idx.insert(axis, full)
                e = p[tuple(idx)] if e is None else e | p[tuple(idx)]
        for s in (a, b):
            idx = [full, full]
            idx.insert(axis, s)
            f = p[tuple(idx)] if f is None else f | p[tuple(idx)]
        edges += int(e.sum())
        faces += int(f.sum())
    return int(verts.sum()) - edges + faces - int(p.sum())


def _local_delta(mask: np.ndarray, add: np.ndarray, box: tuple[slice, ...]) -> int:
    """Change of the Euler number when ``add`` (inside ``box``) joins ``mask``.

    Exact: the Euler number counts local cells, and cells outside ``box`` grown by one voxel
    do not touch the added voxels.
    """
    grown = tuple(
        slice(max(s.start - 1, 0), min(s.stop + 1, n)) for s, n in zip(box, mask.shape, strict=True)
    )
    before = mask[grown]
    after = before | add[grown]
    return euler_number(after) - euler_number(before)


def genus(mask: np.ndarray) -> int | None:
    """Number of tunnels of a mask with one 26-connected component and no cavities."""
    _, n = ndimage.label(mask, structure=STRUCT26)
    if n != 1:
        return None
    filled = ndimage.binary_fill_holes(mask)
    cavities = ndimage.label(filled & ~mask, structure=STRUCT26)[1]
    return 1 + cavities - euler_number(mask)


# --------------------------------------------------------------------------- helpers


def _dilate(mask: np.ndarray, r: float, spacing: Sequence[float]) -> np.ndarray:
    return ndimage.distance_transform_edt(~mask, sampling=spacing) <= r


def _erode(mask: np.ndarray, r: float, spacing: Sequence[float]) -> np.ndarray:
    return ndimage.distance_transform_edt(mask, sampling=spacing) > r


def _hole_pool(mask: np.ndarray, radius_mm: float, spacing: Sequence[float]) -> np.ndarray:
    """Voxels that may span holes up to about ``radius_mm`` wide.

    A closing alone does not close a hole in a thin plate (a ball enters the hole from both
    sides), so the plate is first thickened by ``radius/2``, closed with ``radius``, and the
    filled core is grown back to the original plate (``radius/2 + 1 voxel``).
    """
    t = radius_mm / 2
    pad = int(np.ceil((radius_mm + t + 1) / min(spacing))) + 1
    m = np.pad(mask, pad)
    thick = _dilate(m, t, spacing)
    core = _erode(_dilate(thick, radius_mm, spacing), radius_mm, spacing) & ~thick
    pool = _dilate(core, t + min(spacing), spacing) & ~m
    return pool[tuple(slice(pad, -pad) for _ in range(3))]


def _plate_trim(
    mask: np.ndarray, comp: np.ndarray, spacing: np.ndarray, ring_mm: float = 4.0
) -> tuple[np.ndarray, float]:
    """``comp`` cut to the local plate around it (best-fit plane of the bone within
    ``ring_mm``, thickness from the spread across it); also returns the half thickness."""
    box = _grow(_bbox(comp), int(np.ceil(ring_mm / spacing.min())) + 1, comp.shape)
    near = _dilate(comp[box], ring_mm, spacing) & mask[box]
    pts = np.argwhere(near) * spacing
    if len(pts) < 10:
        return comp, 0.0
    c = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    half = max(s[2] / np.sqrt(len(pts)) * np.sqrt(3.0), 0.75 * spacing.min())
    q = np.argwhere(comp[box])
    off = np.abs((q * spacing - c) @ vt[2])
    keep = q[off <= half + 0.5 * spacing.min()]
    out = np.zeros_like(comp)
    sub = out[box]
    sub[tuple(keep.T)] = True
    return out, float(half)


def _grow(box: tuple[slice, ...], n: int, shape: Sequence[int]) -> tuple[slice, ...]:
    return tuple(
        slice(max(s.start - n, 0), min(s.stop + n, d)) for s, d in zip(box, shape, strict=True)
    )


def _diameter(patch: np.ndarray, spacing: np.ndarray) -> float:
    """Width of a hole patch in mm: mean of its two largest extents along principal axes."""
    pts = np.argwhere(patch) * spacing
    if len(pts) < 3:
        return float(spacing.max())
    c = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ext = np.sort(np.ptp(c @ vt.T, axis=0))[::-1] + spacing.min()
    return float((ext[0] + ext[1]) / 2)


def _bbox(component: np.ndarray) -> tuple[slice, ...]:
    return ndimage.find_objects(component.astype(np.int8))[0]


def _center(component: np.ndarray, affine: np.ndarray) -> tuple[list[float], list[float]]:
    ijk = np.argwhere(component).mean(axis=0)
    xyz = affine[:3, :3] @ ijk + affine[:3, 3]
    return [round(float(v), 2) for v in ijk], [round(float(v), 2) for v in xyz]


# --------------------------------------------------------------------------- holes


@dataclass
class HoleRepair:
    mask: np.ndarray
    genus_before: int | None
    genus_after: int | None
    expected: int
    defects: list[Defect] = field(default_factory=list)
    added_voxels: int = 0
    patches: list[np.ndarray | None] = field(default_factory=list)  # added voxels per defect


@dataclass
class _Hole:
    patch: np.ndarray
    delta: int  # tunnels it closes
    diameter_mm: float
    source: str


def find_holes(
    mask: np.ndarray,
    spacing: Sequence[float],
    *,
    removed_by_opening: np.ndarray | None = None,
    closing_mm: Sequence[float] = DEFAULT_CLOSING_MM,
    blocked: np.ndarray | None = None,
    max_tunnels: int | None = None,
    donor: np.ndarray | None = None,
    donor_name: str = "donor",
) -> list[_Hole]:
    """Patches that each close exactly one tunnel of ``mask``, from the smallest scale up.

    Pools: voxels the opening removed, then :func:`_hole_pool` at growing scales. A
    candidate (first cut to the local plate, :func:`_plate_trim`, then as it is) is
    accepted only if the genus of the mask with all accepted patches drops by exactly one
    and the mask stays one component (a patch closing two tunnels at once, e.g. a wing
    hole together with the obturator foramen, is rejected). Local Euler changes are used
    as a cheap pre-filter. Patches never overlap; ``max_tunnels`` stops the search.

    ``donor`` = the same bone from another segmentation (MuscleMap hip bone for the
    TotalSegmentator one). Its voxels are tried first: they are measured, not interpolated,
    and fill holes of any size. A donor part is cut to the hole (:func:`_hole_pool` with the
    smallest radius of ``DONOR_TRIM_MM`` that still closes the tunnel), so the donor's
    different outline elsewhere on the bone is not added. A donor part may close several
    tunnels at once (one wide hole split by a strand); it never touches the last
    ``expected`` tunnels, see :func:`repair_holes`.
    """
    spacing = np.asarray(spacing, dtype=float)
    work = np.asarray(mask, dtype=bool).copy()
    current = genus(work)
    holes: list[_Hole] = []
    if current is None or current == 0:
        return holes
    if donor is not None:
        current = _donor_holes(work, current, donor, spacing, blocked, holes, donor_name,
                               max_tunnels)  # fmt: skip
    pools: list[tuple[str, float]] = []
    if removed_by_opening is not None and removed_by_opening.any():
        pools.append(("opening", 0.0))
    pools += [("closing", float(r)) for r in closing_mm]
    for source, radius in pools:
        if current == 0 or (max_tunnels is not None and sum(h.delta for h in holes) >= max_tunnels):
            break
        pool = (
            removed_by_opening.copy() if source == "opening" else _hole_pool(work, radius, spacing)
        )
        pool &= ~work
        if blocked is not None:
            pool &= ~blocked
        labels, n = ndimage.label(pool, structure=STRUCT26)
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel())[1:]
        boxes = ndimage.find_objects(labels)
        for idx in np.argsort(sizes, kind="stable"):
            if current == 0 or (
                max_tunnels is not None and sum(h.delta for h in holes) >= max_tunnels
            ):
                break
            box = boxes[idx]
            comp = np.zeros_like(work)
            comp[box] = labels[box] == idx + 1
            options = [comp]
            if source == "closing":
                trimmed, _half = _plate_trim(work, comp, spacing)
                if trimmed.any():
                    options.insert(0, trimmed)
            for patch in options:
                if _local_delta(work, patch, _bbox(patch)) == 0:
                    continue  # pre-filter: no topological change at all
                g = genus(work | patch)
                if g is not None and g == current - 1:
                    break
            else:
                continue
            holes.append(_Hole(patch, 1, _diameter(patch, spacing), source))
            work |= patch
            current = g
    return holes


def _donor_trim(work, comp, g, spacing, trims) -> np.ndarray:
    """Smallest part of the donor component ``comp`` that still gives genus ``g``.

    The donor outline differs from the mask everywhere a little (a layer of 1-2 voxels on
    the plate); only the part inside the hole is wanted. Candidates: voxels deeper than
    ``d`` mm from the mask, grown back by ``d`` inside the donor so that they reach the rim
    (layers thinner than ``d`` vanish), and the cut by :func:`_hole_pool`.
    """
    dist = ndimage.distance_transform_edt(~work, sampling=spacing)
    best = comp
    for d in DONOR_DEPTH_MM:
        core = comp & (dist > d)
        if not core.any():
            continue
        grown = comp & _dilate(core, d + spacing.max(), spacing)
        if grown.sum() < best.sum() and genus(work | grown) == g:
            best = grown
    for r in DONOR_TRIM_MM:
        if r not in trims:
            trims[r] = _hole_pool(work, r, spacing)
        cut = best & trims[r]
        if cut.any() and cut.sum() < best.sum() and genus(work | cut) == g:
            return cut
    return best


def _donor_holes(work, current, donor, spacing, blocked, holes, name, max_tunnels) -> int:
    """Donor parts that close tunnels (``work`` and ``holes`` updated in place)."""
    pool = np.asarray(donor, dtype=bool) & ~work
    if blocked is not None:
        pool &= ~blocked
    labels, n = ndimage.label(pool, structure=STRUCT26)
    if n == 0:
        return current
    sizes = np.bincount(labels.ravel())[1:]
    trims: dict[float, np.ndarray] = {}
    for idx in np.argsort(sizes, kind="stable"):
        if current == 0 or (max_tunnels is not None and sum(h.delta for h in holes) >= max_tunnels):
            break
        comp = labels == idx + 1
        if _local_delta(work, comp, _bbox(comp)) == 0:
            continue
        g = genus(work | comp)
        if g is None or g >= current:
            continue
        patch = _donor_trim(work, comp, g, spacing, trims)
        holes.append(_Hole(patch, current - g, _diameter(patch, spacing), name))
        work |= patch
        current = g
    return current


def _too_big(h: _Hole, max_hole_mm: float) -> bool:
    measured = h.source not in ("opening", "closing")  # donor voxels, any size
    return not measured and h.diameter_mm > max_hole_mm


def prune_patches(
    mask: np.ndarray, repaired: np.ndarray, patches: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Patches without the pieces the genus of ``repaired`` does not need.

    A donor patch is cut to the hole only roughly and may carry the donor's different
    outline elsewhere on the bone (TLEM 2.0 MRI: ~30 % of the added voxels more than 40 mm
    from any hole). Every 26-connected piece of a patch is dropped, smallest first, if the
    genus stays the same without it.
    """
    target = genus(repaired)
    work = np.asarray(repaired, dtype=bool).copy()
    out: list[np.ndarray] = []
    for patch in patches:
        added = patch & ~mask
        labels, n = ndimage.label(added, structure=STRUCT26)
        kept = patch.copy()
        if n > 1:
            sizes = np.bincount(labels.ravel())[1:]
            for idx in np.argsort(sizes, kind="stable"):
                piece = labels == idx + 1
                trial = work & ~piece
                if genus(trial) == target:
                    work = trial
                    kept &= ~piece
        out.append(kept)
    return out


def repair_holes(
    mask: np.ndarray,
    affine: np.ndarray,
    structure: str,
    *,
    expected_genus: int = 0,
    removed_by_opening: np.ndarray | None = None,
    closing_mm: Sequence[float] = DEFAULT_CLOSING_MM,
    max_hole_mm: float = 40.0,
    blocked: np.ndarray | None = None,
    repair: bool = True,
    donor: np.ndarray | None = None,
    donor_name: str = "donor",
) -> HoleRepair:
    """Fill through-holes of ``mask`` so that its genus equals ``expected_genus``.

    The ``expected_genus`` largest tunnels are anatomical (obturator foramen) and stay
    open; the smaller ones are filled. Tunnels not found up to the largest scale count as
    the largest. ``mask`` is one cleaned part (one component, cavities filled); ``blocked``
    voxels (other structures) are never added. ``repair=False`` only locates the holes.
    """
    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(affine, dtype=float)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    g0 = genus(mask)
    result = HoleRepair(mask, g0, g0, expected_genus)
    if g0 is None or g0 <= expected_genus:
        return result

    holes = find_holes(
        mask, spacing, removed_by_opening=removed_by_opening, closing_mm=closing_mm,
        blocked=blocked, max_tunnels=g0 - expected_genus, donor=donor, donor_name=donor_name,
    )  # fmt: skip
    unfound = g0 - sum(h.delta for h in holes)
    keep_open = max(expected_genus - unfound, 0)  # unfound tunnels are the widest ones
    by_size = sorted(holes, key=lambda h: h.diameter_mm)
    to_fill = by_size[: len(by_size) - keep_open]

    fill = [h for h in to_fill if repair and not _too_big(h, max_hole_mm)]
    work = mask.copy()
    for h in fill:
        work |= h.patch
    pruned = prune_patches(mask, work, [h.patch for h in fill])
    for h, patch in zip(fill, pruned, strict=True):
        h.patch, h.diameter_mm = patch, _diameter(patch, spacing)
    work = mask.copy()
    for h in fill:
        work |= h.patch

    for h in to_fill:
        too_big = _too_big(h, max_hole_mm)
        ijk, xyz = _center(h.patch, affine)
        fix = repair and not too_big
        result.defects.append(
            Defect(
                structure=structure,
                kind="hole",
                center_ijk=ijk,
                center_mm=xyz,
                size_mm=round(h.diameter_mm, 1),
                fixed=fix,
                method=({"opening": "opening", "closing": "plate patch"}.get(h.source, h.source))
                if fix
                else ("too large" if too_big else "located only"),
                voxels=int(h.patch.sum()) if fix else 0,
                detail=f"through-hole, genus {g0} > expected {expected_genus}"
                + (
                    f"; filled from {h.source} ({h.delta} tunnel(s))"
                    if h.source not in ("opening", "closing")
                    else ""
                )
                + (f"; larger than {max_hole_mm:g} mm" if too_big else ""),
            )
        )
        result.patches.append(h.patch if fix else None)
        if fix:
            result.added_voxels += int(h.patch.sum())

    excess_unfound = unfound - expected_genus
    if excess_unfound > 0:  # wider holes: locate them (never filled automatically)
        wide = find_holes(
            work, spacing, closing_mm=LOCATE_ONLY_MM, blocked=blocked, max_tunnels=unfound
        )
        for h in sorted(wide, key=lambda h: h.diameter_mm)[:excess_unfound]:
            ijk, xyz = _center(h.patch, affine)
            result.defects.append(
                Defect(
                    structure=structure,
                    kind="hole",
                    center_ijk=ijk,
                    center_mm=xyz,
                    size_mm=round(h.diameter_mm, 1),
                    fixed=False,
                    method="too large",
                    detail=f"through-hole about {h.diameter_mm:.0f} mm wide, too large to fill "
                    f"automatically (genus {g0} > expected {expected_genus})",
                )
            )
            result.patches.append(None)
            excess_unfound -= h.delta
    if excess_unfound > 0:
        ijk, xyz = _center(mask, affine)
        result.defects.append(
            Defect(
                structure=structure,
                kind="hole",
                center_ijk=ijk,
                center_mm=xyz,
                size_mm=None,
                fixed=False,
                method="not found",
                detail=f"{excess_unfound} through-hole(s) wider than {max(LOCATE_ONLY_MM):g} mm "
                "or blocked; location unknown",
            )
        )
        result.patches.append(None)
    result.mask = work
    result.genus_after = genus(work)
    return result


# --------------------------------------------------------------------------- gaps


@dataclass
class GapRepair:
    mask: np.ndarray
    fragments: int
    defects: list[Defect] = field(default_factory=list)
    added_voxels: int = 0
    patches: list[np.ndarray | None] = field(default_factory=list)  # bridge per defect


def _end(points: np.ndarray, proj: np.ndarray, at_max: bool, slab_mm: float):
    """Centre and radius (mm) of the fragment end (``slab_mm`` thick slab)."""
    edge = proj.max() if at_max else proj.min()
    sel = proj >= edge - slab_mm if at_max else proj <= edge + slab_mm
    return points[sel].mean(axis=0), sel.sum()


def repair_gaps(
    mask: np.ndarray,
    affine: np.ndarray,
    structure: str,
    *,
    max_gap_mm: float = 40.0,
    max_angle_deg: float = 30.0,
    min_fragment_fraction: float = 0.02,
    blocked: np.ndarray | None = None,
    repair: bool = True,
    slab_mm: float = 4.0,
) -> GapRepair:
    """Join fragments of an elongated bone lying one after another along its axis."""
    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(affine, dtype=float)
    labels, n = ndimage.label(mask, structure=STRUCT26)
    result = GapRepair(mask, n)
    if n < 2:
        return result
    sizes = np.bincount(labels.ravel())[1:]
    keep = [i + 1 for i in range(n) if sizes[i] >= min_fragment_fraction * sizes.max()]
    if len(keep) < 2:
        return result

    world = lambda ijk: ijk @ affine[:3, :3].T + affine[:3, 3]  # noqa: E731
    pts_all = world(np.argwhere(np.isin(labels, keep)).astype(float))
    centre = pts_all.mean(axis=0)
    axis = np.linalg.svd(pts_all - centre, full_matrices=False)[2][0]
    voxel_volume = abs(np.linalg.det(affine[:3, :3]))

    frags = []
    for lab in keep:
        pts = world(np.argwhere(labels == lab).astype(float))
        proj = (pts - centre) @ axis
        frags.append((proj.min(), proj.max(), lab, pts, proj))
    frags.sort(key=lambda f: f[0])

    work = mask.copy()
    inv = np.linalg.inv(affine)
    for (_lo_a, hi_a, _lab_a, pts_a, pr_a), (
        lo_b,
        _hi_b,
        _lab_b,
        pts_b,
        pr_b,
    ) in itertools.pairwise(frags):
        if lo_b < hi_a - slab_mm:  # side by side, not one after another
            continue
        pa, na = _end(pts_a, pr_a, True, slab_mm)
        pb, nb = _end(pts_b, pr_b, False, slab_mm)
        gap_vec = pb - pa
        gap = float(max(lo_b - hi_a, 0.0))  # along the bone axis, tip to tip
        angle = float(
            np.degrees(
                np.arccos(min(1.0, abs(gap_vec @ axis) / max(float(np.linalg.norm(gap_vec)), 1e-9)))
            )
        )
        ra = np.sqrt(na * voxel_volume / slab_mm / np.pi)
        rb = np.sqrt(nb * voxel_volume / slab_mm / np.pi)
        mid = (pa + pb) / 2
        ok = gap <= max_gap_mm and angle <= max_angle_deg
        defect = Defect(
            structure=structure,
            kind="gap",
            center_ijk=[round(float(v), 2) for v in inv[:3, :3] @ mid + inv[:3, 3]],
            center_mm=[round(float(v), 2) for v in mid],
            size_mm=round(gap, 1),
            fixed=bool(ok and repair),
            method="bridge (interpolated cylinder)" if ok else "not bridged",
            voxels=0,
            detail=f"bone split into fragments, gap {gap:.1f} mm, ends {2 * ra:.1f}/"
            f"{2 * rb:.1f} mm wide, {angle:.0f} deg off the bone axis"
            + ("" if ok else f" (limits {max_gap_mm:g} mm, {max_angle_deg:g} deg)"),
            end_a_mm=[round(float(v), 2) for v in pa],
            end_b_mm=[round(float(v), 2) for v in pb],
        )
        if ok and repair:
            bridge = _cone(work.shape, affine, pa, pb, ra, rb, overlap_mm=slab_mm)
            if blocked is not None:
                bridge &= ~blocked
            bridge &= ~work
            defect.voxels = int(bridge.sum())
            work |= bridge
            result.added_voxels += defect.voxels
            result.patches.append(bridge)
        else:
            result.patches.append(None)
        result.defects.append(defect)
    result.mask = work
    return result


def _cone(shape, affine, pa, pb, ra, rb, overlap_mm: float) -> np.ndarray:
    """Voxels inside the truncated cone pa(ra) -> pb(rb), extended ``overlap_mm`` into the
    fragments so that the bridge joins them."""
    d = pb - pa
    length = float(np.linalg.norm(d))
    u = d / length
    a = pa - u * overlap_mm
    b = pb + u * overlap_mm
    rmax = max(ra, rb)
    inv = np.linalg.inv(affine)
    corners = np.array([a - rmax, a + rmax, b - rmax, b + rmax])
    ijk = corners @ inv[:3, :3].T + inv[:3, 3]
    lo = np.clip(np.floor(ijk.min(axis=0)).astype(int) - 1, 0, None)
    hi = np.minimum(np.ceil(ijk.max(axis=0)).astype(int) + 2, shape)
    out = np.zeros(shape, dtype=bool)
    if np.any(hi <= lo):
        return out
    grid = np.stack(
        np.meshgrid(*(np.arange(a, b) for a, b in zip(lo, hi, strict=True)), indexing="ij"), -1
    ).astype(float)
    xyz = grid @ affine[:3, :3].T + affine[:3, 3]
    rel = xyz - a
    t = rel @ u
    full = length + 2 * overlap_mm
    radial = np.linalg.norm(rel - t[..., None] * u, axis=-1)
    frac = np.clip((t - overlap_mm) / length, 0.0, 1.0)
    inside = (t >= 0) & (t <= full) & (radial <= ra + (rb - ra) * frac)
    out[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = inside
    return out


__all__ = [
    "DEFAULT_CLOSING_MM",
    "DONORS",
    "EXPECTED_GENUS",
    "LONG_BONES",
    "GapRepair",
    "HoleRepair",
    "euler_number",
    "find_holes",
    "genus",
    "prune_patches",
    "repair_gaps",
    "repair_holes",
]
