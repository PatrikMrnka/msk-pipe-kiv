# SPDX-License-Identifier: Apache-2.0
"""Attachment outlines supplied by the user (method ``manual``): files, names, layout.

A folder of outlines, e.g. drawn in NMSBuilder on the bone meshes of a run or edited in
the GUI. Which muscle and which area (origin ``Ori`` / insertion ``Ins``) a file belongs
to is taken from, in this order:

1. ``areas.csv`` in the folder (``file,muscle,kind``), when present - nothing else is
   guessed then;
2. the layout of the atlas and of the pipeline outputs, ``<muscle>/<Stem>_Ori.vtk``: the
   muscle is the sub-folder, the area the suffix;
3. the file name: ``adductor_longus_r_Ins.txt``, ``AdductorLongus_R_Ins.vtk``,
   ``Ori_1_R_M102_Adduct_Brev.vtk`` (LHDL landmark clouds: muscle code ``M102``, mapped
   by the codes of the installed atlas). Case and separators do not matter; ``Ori`` /
   ``Origin`` and ``Ins`` / ``Insertion`` anywhere in the name; a missing side is
   ``skeleton.side``.

Point files: legacy VTK (ASCII or binary; the order of a polyline is used when there is
one), VTK XML ``.vtp``, or text (``.txt``, ``.csv``, ``.xyz``, ``.pts``) with the last three
numbers of each line as x, y, z (header and comment lines are skipped, so ``name x y z``
lines work). Coordinates in mm.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

MAPPING_FILE = "areas.csv"
INDEX_FILE = "manual.json"  # written by the GUI editor (sources of the areas)
KINDS = ("Ori", "Ins")
POINT_SUFFIXES = (".vtk", ".vtp", ".txt", ".csv", ".xyz", ".pts")
MAX_DEPTH = 2  # <folder>/<muscle>/<file>

_KIND_TOKENS = {"ori": "Ori", "origin": "Ori", "ins": "Ins", "insertion": "Ins"}
_SIDE_TOKENS = {"r": "r", "right": "r", "l": "l", "left": "l"}
_CODE = re.compile(r"^m\d{3}$")
_STEM = re.compile(r"^([A-Za-z][A-Za-z0-9]*)_(Ori|Ins)$")
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


class ManualAreasError(ValueError):
    """The outline folder or a file in it cannot be used; the message is for end users."""


@dataclass(frozen=True)
class AreaFile:
    """One outline file and the area it was assigned to."""

    path: Path
    muscle: str  # with side, e.g. adductor_longus_r
    kind: str  # Ori | Ins
    stem: str  # Muscle Wrapping file stem, e.g. AdductorLongus
    rule: str  # csv | layout | name | code

    @property
    def key(self) -> tuple[str, str]:
        return self.muscle, self.kind


@dataclass
class Matching:
    """Result of assigning the files of a folder to muscles and areas."""

    folder: Path
    areas: dict[tuple[str, str], AreaFile] = field(default_factory=dict)
    unmatched: list[Path] = field(default_factory=list)
    duplicates: dict[tuple[str, str], list[Path]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def of_side(self, side: str) -> dict[tuple[str, str], AreaFile]:
        return {k: a for k, a in self.areas.items() if k[0].endswith(f"_{side}")}

    def problems(self) -> list[str]:
        """Errors that make the folder unusable (duplicates, bad areas.csv)."""
        out = list(self.errors)
        for (muscle, kind), paths in sorted(self.duplicates.items()):
            names = ", ".join(self.rel(p) for p in paths)
            out.append(f"{muscle} {kind}: more than one file ({names})")
        return out

    def rel(self, path: Path) -> str:
        try:
            return path.relative_to(self.folder).as_posix()
        except ValueError:
            return str(path)


# --------------------------------------------------------------------------- names


def tokens(text: str) -> list[str]:
    """Lower-case words of a name: CamelCase, separators and digits split."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def default_stem(muscle: str) -> str:
    """Muscle Wrapping stem of a muscle: ``adductor_longus_r`` -> ``AdductorLongus``."""
    base = re.sub(r"_[lr]$", "", muscle)
    return "".join(part.capitalize() for part in base.split("_"))


def known_muscles() -> list[str]:
    """Muscle names (without side) of the unified label scheme."""
    from mskpipe.labelmap.scheme import load_scheme

    names: list[str] = []
    for name, structure in load_scheme().structures.items():
        base = re.sub(r"_[lr]$", "", name)
        if structure.kind == "muscle" and base not in names:
            names.append(base)
    return names


def atlas_codes(atlas_dir: Path | None = None) -> dict[str, str]:
    """LHDL muscle codes (``m102``) -> muscle name without side, from an atlas index
    (``atlas_dir`` or the installed atlas); empty when there is none."""
    if atlas_dir is None:
        from mskpipe.resources.atlas import resolve_atlas_dir

        atlas_dir = resolve_atlas_dir(None)[0]
    if atlas_dir is None:
        return {}
    try:
        atlas = json.loads((Path(atlas_dir) / "atlas.json").read_text(encoding="utf-8"))
        entries = dict(atlas["muscles"])
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    codes: dict[str, str] = {}
    for name, entry in entries.items():
        for code in (entry or {}).get("codes", []) or []:
            codes.setdefault(str(code).lower(), re.sub(r"_[lr]$", "", name))
    return codes


def _find_muscle(words: Sequence[str], muscles: Sequence[tuple[str, list[str]]]) -> str | None:
    """Muscle whose words occur in ``words`` as a run; the longest match wins."""
    best, best_len = None, 0
    for name, parts in muscles:
        n = len(parts)
        if n > best_len and any(list(words[i : i + n]) == parts for i in range(len(words) - n + 1)):
            best, best_len = name, n
    return best


def _side(words: Sequence[str]) -> str | None:
    found = {_SIDE_TOKENS[w] for w in words if w in _SIDE_TOKENS}
    return found.pop() if len(found) == 1 else None


def _kind(words: Sequence[str]) -> str | None:
    found = {_KIND_TOKENS[w] for w in words if w in _KIND_TOKENS}
    return found.pop() if len(found) == 1 else None


def guess_area(
    rel: PurePosixPath | str,
    side: str,
    muscles: Sequence[str],
    codes: Mapping[str, str] | None = None,
) -> AreaFile | None:
    """Muscle and area of a file from its path relative to the folder (rules 2 and 3 of
    the module docstring); ``None`` if the name does not tell."""
    rel = PurePosixPath(rel)
    file_words = tokens(rel.stem)
    folder_words = tokens(rel.parts[-2]) if len(rel.parts) > 1 else []
    kind = _kind(file_words)
    if kind is None:
        return None
    table = [(m, m.split("_")) for m in muscles]
    muscle, rule = _find_muscle(folder_words, table), "layout"
    if muscle is None:
        muscle, rule = _find_muscle(file_words, table), "name"
    if muscle is None and codes:
        hits = {codes[w] for w in file_words if _CODE.match(w) and w in codes}
        if len(hits) == 1:
            muscle, rule = hits.pop(), "code"
    if muscle is None:
        return None
    file_side = _side([w for w in file_words if w not in muscle.split("_")])
    folder_side = _side(folder_words[-1:]) if rule == "layout" else None
    full = f"{muscle}_{folder_side or file_side or side}"
    m = _STEM.match(rel.stem)
    stem = m.group(1) if m and tokens(m.group(1)) == muscle.split("_") else default_stem(full)
    return AreaFile(Path(rel), full, kind, stem, rule)


# --------------------------------------------------------------------------- folder


def area_files(folder: Path) -> list[Path]:
    """Point files of the folder (and its sub-folders, depth ``MAX_DEPTH``), sorted."""
    folder = Path(folder)
    out = []
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder)
        if (
            path.is_file()
            and path.suffix.lower() in POINT_SUFFIXES
            and len(rel.parts) <= MAX_DEPTH
            and path.name != MAPPING_FILE
            and not any(p.startswith(".") for p in rel.parts)
        ):
            out.append(path)
    return out


def _normal_kind(value: str) -> str | None:
    return _KIND_TOKENS.get(value.strip().lower())


def read_mapping(
    folder: Path, side: str, muscles: Sequence[str]
) -> tuple[list[AreaFile], list[str]]:
    """Rows of ``areas.csv`` as areas, and the errors found in it."""
    path = Path(folder) / MAPPING_FILE
    areas: list[AreaFile] = []
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return [], [f"{MAPPING_FILE}: {exc}"]
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"file", "muscle", "kind"} <= set(reader.fieldnames):
        return [], [f"{MAPPING_FILE}: columns file, muscle, kind are required"]
    for i, row in enumerate(reader, start=2):
        name = (row.get("file") or "").strip()
        muscle = (row.get("muscle") or "").strip().lower()
        kind = _normal_kind(row.get("kind") or "")
        if not name and not muscle:
            continue
        base = re.sub(r"_[lr]$", "", muscle)
        if base not in muscles:
            errors.append(f"{MAPPING_FILE} line {i}: unknown muscle '{muscle}'")
            continue
        if kind is None:
            errors.append(f"{MAPPING_FILE} line {i}: kind must be Ori or Ins")
            continue
        file = Path(folder) / name
        if not file.is_file():
            errors.append(f"{MAPPING_FILE} line {i}: file not found: {name}")
            continue
        full = muscle if re.search(r"_[lr]$", muscle) else f"{muscle}_{side}"
        m = _STEM.match(file.stem)
        stem = m.group(1) if m and tokens(m.group(1)) == base.split("_") else default_stem(full)
        areas.append(AreaFile(file, full, kind, stem, "csv"))
    return areas, errors


def match_folder(
    folder: str | Path,
    side: str,
    muscles: Sequence[str] | None = None,
    codes: Mapping[str, str] | None = None,
) -> Matching:
    """Assign the files of ``folder`` to muscles and areas (see the module docstring)."""
    folder = Path(folder)
    result = Matching(folder)
    if not folder.is_dir():
        result.errors.append(f"Folder of the attachment outlines not found: {folder}")
        return result
    muscles = list(muscles) if muscles is not None else known_muscles()
    files = area_files(folder)
    if (folder / MAPPING_FILE).is_file():
        found, result.errors = read_mapping(folder, side, muscles)
        listed = {a.path.resolve() for a in found}
        result.unmatched = [p for p in files if p.resolve() not in listed]
    else:
        codes = atlas_codes() if codes is None else codes
        found = []
        for path in files:
            guess = guess_area(path.relative_to(folder).as_posix(), side, muscles, codes)
            if guess is None:
                result.unmatched.append(path)
            else:
                found.append(AreaFile(path, guess.muscle, guess.kind, guess.stem, guess.rule))
    by_key: dict[tuple[str, str], list[AreaFile]] = {}
    for area in found:
        by_key.setdefault(area.key, []).append(area)
    for key, group in sorted(by_key.items()):
        if len(group) == 1:
            result.areas[key] = group[0]
        else:
            result.duplicates[key] = [a.path for a in group]
    return result


def write_mapping(folder: Path, areas: Iterable[AreaFile]) -> Path:
    """``areas.csv`` of the assignment (file names relative to ``folder``)."""
    folder = Path(folder)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["file", "muscle", "kind"])
    for a in sorted(areas, key=lambda a: (a.muscle, KINDS.index(a.kind))):
        try:
            name = Path(a.path).relative_to(folder).as_posix()
        except ValueError:
            name = Path(a.path).as_posix()
        writer.writerow([name, a.muscle, a.kind])
    path = folder / MAPPING_FILE
    path.write_text(out.getvalue(), encoding="utf-8")
    return path


def folder_fingerprint(folder: str | Path | None) -> str:
    """sha256 over the point files and ``areas.csv`` (names and contents)."""
    if folder is None or not Path(folder).is_dir():
        return "missing"
    folder = Path(folder)
    files = area_files(folder)
    if (folder / MAPPING_FILE).is_file():
        files.append(folder / MAPPING_FILE)
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(folder).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def layout_path(folder: Path, muscle: str, kind: str, stem: str | None = None) -> Path:
    """Where an area is written in the standard layout."""
    return Path(folder) / muscle / f"{stem or default_stem(muscle)}_{kind}.vtk"


# --------------------------------------------------------------------------- points


def read_outline(path: str | Path) -> np.ndarray:
    """Points (N, 3) of an outline file, in file (or polyline) order."""
    import numpy as np

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".vtk", ".vtp"):
        points = _read_vtk(path)
    elif suffix in POINT_SUFFIXES:
        points = _read_text(path)
    else:
        raise ManualAreasError(f"{path.name}: unsupported file type ({suffix})")
    if len(points) < 3:
        raise ManualAreasError(
            f"{path.name}: an outline needs at least 3 points, got {len(points)}"
        )
    if not np.isfinite(points).all():
        raise ManualAreasError(f"{path.name}: coordinates are not finite")
    return points


def _read_text(path: Path) -> np.ndarray:
    import numpy as np

    rows = []
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ManualAreasError(f"{path.name}: {exc}") from exc
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//", "%")):
            continue
        numbers = [t for t in re.split(r"[\s,;]+", line) if _NUMBER.match(t)]
        if len(numbers) >= 3:
            rows.append([float(v) for v in numbers[-3:]])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def _read_vtk(path: Path) -> np.ndarray:
    import numpy as np

    from mskpipe.io.attachment_vtk import AttachmentFileError, read_points

    if path.suffix.lower() == ".vtk":
        try:
            return read_points(path)
        except (AttachmentFileError, UnicodeDecodeError):
            pass  # binary or another dataset: read with VTK
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkGenericDataObjectReader
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

    if path.suffix.lower() == ".vtp":
        reader = vtkXMLPolyDataReader()
    else:
        reader = vtkGenericDataObjectReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    if data is None or not hasattr(data, "GetPoints") or data.GetPoints() is None:
        raise ManualAreasError(f"{path.name}: no points in the file")
    points = vtk_to_numpy(data.GetPoints().GetData()).astype(np.float64).reshape(-1, 3)
    lines = data.GetLines() if hasattr(data, "GetLines") else None
    if lines is not None and lines.GetNumberOfCells() == 1:
        from vtkmodules.vtkCommonCore import vtkIdList

        ids = vtkIdList()
        lines.InitTraversal()
        lines.GetNextCell(ids)
        order = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
        if len(order) > 1 and order[0] == order[-1]:
            order = order[:-1]  # closed polyline
        if len(order) >= 3:
            points = points[order]
    return points


FRAMES = ("ras", "lps", "voxel")


def to_ras(points: np.ndarray, frame: str, affine: np.ndarray | None = None) -> np.ndarray:
    """Points in RAS from ``frame``.

    * ``ras``: unchanged (the meshes of a run);
    * ``lps``: x and y flipped (DICOM patient coordinates);
    * ``voxel``: voxel index times voxel size of the run's input image, without its origin
      and orientation (meshes made from the voxel array alone, as MAF/NMSBuilder do with
      a raw volume); ``affine`` = the input image's voxel -> RAS matrix.
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3).copy()
    if frame == "lps":
        pts[:, :2] *= -1.0
    elif frame == "voxel":
        if affine is None:
            raise ManualAreasError("frame voxel needs the input image of the run")
        a = np.asarray(affine, dtype=np.float64)
        spacing = np.linalg.norm(a[:3, :3], axis=0)
        pts = (pts / spacing) @ a[:3, :3].T + a[:3, 3]
    elif frame != "ras":
        raise ManualAreasError(f"frame must be one of {', '.join(FRAMES)}, got {frame}")
    return pts


def image_affine(path: str | Path) -> np.ndarray:
    """Voxel -> RAS matrix of a NIfTI image (header only)."""
    import nibabel as nib

    return nib.load(str(path)).affine


# --------------------------------------------------------------------------- frame guess

FRAME_LABELS = {"ras": "RAS", "lps": "LPS", "voxel": "voxel"}
SAME_MM = 1.0  # frames whose converted outlines differ less than this are the same choice
INSIDE_MARGIN_MM = 10.0  # outlines may reach a little beyond the image (bone surface)
INSIDE_SHARE = 0.95  # share of the points inside the image for a frame to fit
MAX_GUESS_FILES = 200


@dataclass(frozen=True)
class FrameGuess:
    """Coordinate frame of an outline folder, guessed from where its points lie.

    ``basis`` ``bones``: ``scores`` = median distance of the outlines to the bone meshes
    of a run [mm] per frame; ``image``: share of the points inside the input image.
    ``frame`` is None when no frame fits or several fit (e.g. RAS and LPS of an image
    centred on the scanner axis). ``equivalent``: the frames giving the same points as
    ``frame`` (``frame`` included), e.g. voxel and RAS of an image at the origin."""

    frame: str | None
    basis: str
    scores: dict[str, float]
    fits: tuple[str, ...]
    equivalent: tuple[str, ...] = ()

    @property
    def note(self) -> str:
        def score(f: str) -> str:
            v = self.scores[f]
            return f"{v:.1f} mm" if self.basis == "bones" else f"{v:.0%}"

        what = (
            "median distance to the bones of the run"
            if self.basis == "bones"
            else "points inside the input image"
        )
        listed = ", ".join(f"{FRAME_LABELS[f]} {score(f)}" for f in self.scores)
        if self.frame is not None:
            return f"{FRAME_LABELS[self.frame]} ({what}: {listed})"
        if len(self.fits) > 1:
            names = " and ".join(FRAME_LABELS[f] for f in self.fits)
            hint = (
                "; run up to step mesh once to tell them apart by the bones"
                if self.basis == "image"
                else ""
            )
            return f"not detected: {names} fit equally ({what}: {listed}){hint}"
        return f"not detected: no frame fits ({what}: {listed})"


def image_box(affine: np.ndarray, shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper RAS corner [mm] of the axis-aligned box around an image grid."""
    import itertools

    import numpy as np

    n = [int(v) for v in shape[:3]]
    corners = np.array(list(itertools.product(*[(-0.5, k - 0.5) for k in n])), dtype=float)
    a = np.asarray(affine, dtype=np.float64)
    world = corners @ a[:3, :3].T + a[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def guess_frame(
    outlines: Sequence[np.ndarray],
    *,
    affine: np.ndarray | None = None,
    shape: Sequence[int] | None = None,
    bones: Sequence[np.ndarray] | None = None,
    max_distance_mm: float = 20.0,
) -> FrameGuess | None:
    """Which ``frame`` puts ``outlines`` where they belong.

    With ``bones`` (vertex arrays of the bone meshes of a run of the same image): the
    frame whose outlines lie closest to the bones, within ``max_distance_mm``. Without:
    the frame that puts (nearly) all points inside the input image (``affine``,
    ``shape``). Frame ``voxel`` is tried only with ``affine``. Frames giving the same
    points (e.g. voxel and RAS of an image at the origin without rotation) count once,
    RAS first. None when there is nothing to compare with."""
    import numpy as np

    outlines = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in outlines]
    outlines = [p for p in outlines if len(p)]
    has_bones = bones is not None and any(len(b) for b in bones)
    if not outlines or (not has_bones and (affine is None or shape is None)):
        return None
    frames = [f for f in FRAMES if f != "voxel" or affine is not None]
    converted = {f: [to_ras(p, f, affine) for p in outlines] for f in frames}
    if has_bones:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.vstack([np.asarray(b, dtype=np.float64).reshape(-1, 3) for b in bones]))
        scores = {
            f: float(np.median([np.median(tree.query(p)[0]) for p in pts]))
            for f, pts in converted.items()
        }
        fits = [f for f in frames if scores[f] <= max_distance_mm]
        basis = "bones"
    else:
        lo, hi = image_box(affine, shape)  # type: ignore[arg-type]
        lo, hi = lo - INSIDE_MARGIN_MM, hi + INSIDE_MARGIN_MM
        scores = {}
        for f, pts in converted.items():
            allpts = np.vstack(pts)
            scores[f] = float(np.mean(np.all((allpts >= lo) & (allpts <= hi), axis=1)))
        fits = [f for f in frames if scores[f] >= INSIDE_SHARE]
        basis = "image"
    stacked = {f: np.vstack(pts) for f, pts in converted.items()}

    def same(f: str, g: str) -> bool:
        return bool(np.abs(stacked[f] - stacked[g]).max() < SAME_MM)

    distinct: list[str] = []
    for f in fits:
        if not any(same(f, g) for g in distinct):
            distinct.append(f)
    frame = distinct[0] if len(distinct) == 1 else None
    if frame is None and basis == "bones" and len(distinct) > 1:
        ranked = sorted(distinct, key=scores.__getitem__)
        if scores[ranked[0]] * 3.0 < scores[ranked[1]]:  # clearly closer
            frame = ranked[0]
    equivalent = tuple(f for f in frames if frame is not None and same(f, frame))
    return FrameGuess(frame, basis, scores, tuple(distinct), equivalent)


def folder_outlines(folder: str | Path, limit: int = MAX_GUESS_FILES) -> list[np.ndarray]:
    """Points of the readable outline files of a folder (for :func:`guess_frame`)."""
    out = []
    folder = Path(folder)
    if not folder.is_dir():
        return out
    for path in area_files(folder)[:limit]:
        try:
            out.append(read_outline(path))
        except (ManualAreasError, OSError, ValueError):
            continue
    return out


def read_index(folder: Path) -> dict[str, Any]:
    """``manual.json`` of the GUI editor, or an empty dict."""
    try:
        data = json.loads((Path(folder) / INDEX_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


__all__ = [
    "FRAMES",
    "FRAME_LABELS",
    "INDEX_FILE",
    "KINDS",
    "MAPPING_FILE",
    "POINT_SUFFIXES",
    "AreaFile",
    "FrameGuess",
    "ManualAreasError",
    "Matching",
    "area_files",
    "atlas_codes",
    "default_stem",
    "folder_fingerprint",
    "folder_outlines",
    "guess_area",
    "guess_frame",
    "image_affine",
    "image_box",
    "known_muscles",
    "layout_path",
    "match_folder",
    "read_index",
    "read_mapping",
    "read_outline",
    "to_ras",
    "tokens",
    "write_mapping",
]
