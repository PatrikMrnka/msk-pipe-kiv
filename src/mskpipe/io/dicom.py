# SPDX-License-Identifier: Apache-2.0
"""DICOM input: a folder of DICOM files converted to NIfTI with dcm2niix.

``dcm2niix`` (MIT, conda-forge) converts every series of the folder, corrects gantry tilt
and irregular slice spacing, and writes a BIDS JSON sidecar with the modality but without
the patient's name, ID or birth date (``-ba y``). Each output volume is checked here:

* only CT and MR series can be used;
* the volume must be 3D (no localizer, single slice or 4D / multi-volume series);
* irregular slice spacing (missing slices) is accepted only when dcm2niix could resample
  the series to equidistant slices (``<series>_Eq_1``); a gantry-tilt corrected volume
  (``<series>_Tilt_1``) replaces the uncorrected one.

A folder with exactly one usable volume needs no choice; with several, the volume is
chosen by its id (``3``, ``9_e2``: the dcm2niix output name, starting with the DICOM
series number).

Conversions are cached in ``<MSKPIPE_HOME>/cache/dicom/<key>`` (key: dcm2niix version
and the paths, sizes and modification times of the files), so the GUI check, the pre-run
checks and the run itself convert a folder once.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DICOM_SUFFIXES",
    "DicomError",
    "DicomScan",
    "DicomVolume",
    "cache_root",
    "dcm2niix_executable",
    "dcm2niix_version",
    "scan",
]

DICOM_SUFFIXES = (".dcm", ".ima", ".dicom")
CACHE_KEEP = 3  # cached folders kept (a CT conversion can take a few hundred MB)
CACHE_FORMAT = 2  # bump when the cached scan.json or the checks change
TIMEOUT_S = 1800

_CONVERT = re.compile(r"^Convert (\d+) DICOM as (.+?) \((\d+(?:x\d+)+)\)\s*$")
_CORRECTED = re.compile(r"^(?P<base>.+)_(?P<kind>Eq|Tilt)_\d+$")
_SERIES_NUMBER = re.compile(r"^(\d+)")
# dcm2niix warnings that do not concern the use of the volume here
_BENIGN = (
    "Patient Position (0018,5100) not specified",
    "Unable to determine manufacturer",
    "conversion is not tuned for vendor",
    "Instance Number (0020,0013) order is not spatial",
    "Slices not stacked",  # echoes etc. become separate volumes, listed as such
    "Empty protocol name",
    "slice timing",  # fMRI slice timing (e.g. Siemens MoCo)
)
_NOTE_PREFIXES = ("Warning:", "Gantry Tilt", "Slices not stacked", "Only able to make")
_AFTER_CONVERT = ("Only able to make",)  # printed after the Convert line of its series
_VARYING = "Interslice distance varies"
_MODALITIES = {"CT": "ct", "MR": "mri"}


class DicomError(ValueError):
    """A DICOM folder cannot be used as input."""


@dataclass(frozen=True)
class DicomVolume:
    """One volume written by dcm2niix (one series, echo, or corrected copy)."""

    id: str  # dcm2niix output name: series number (+ _e2, _Eq_1, _Tilt_1, a, ...)
    nifti: str
    sidecar: str | None
    series_number: int | None
    description: str
    dicom_modality: str  # DICOM value: CT, MR, PT, ...
    image_type: tuple[str, ...]
    files: int
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    warnings: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()  # why the volume cannot be used
    replaced_by: str | None = None  # id of the corrected copy (Tilt, Eq) used instead

    @property
    def modality(self) -> str | None:
        """``ct`` or ``mri`` (pipeline modality), None for other modalities."""
        return _MODALITIES.get(self.dicom_modality.upper())

    @property
    def usable(self) -> bool:
        return not self.problems and self.replaced_by is None

    @property
    def nifti_path(self) -> Path:
        return Path(self.nifti)

    def label(self) -> str:
        """One line: id, description, modality, voxels, spacing, number of files."""
        desc = f" '{self.description}'" if self.description else ""
        shape = " x ".join(str(n) for n in self.shape[:3])
        if len(self.shape) > 3 and self.shape[3] > 1:
            shape += f" x {self.shape[3]}"
        spacing = " x ".join(f"{s:.4g}" for s in self.spacing[:3])
        files = f"{self.files} file{'s' if self.files != 1 else ''}"
        return (
            f"Series {self.id}{desc}: {self.dicom_modality or '?'}, {shape} voxels, "
            f"{spacing} mm, {files}"
        )

    def status(self) -> str:
        """Why the volume is not usable ('' if it is)."""
        if self.replaced_by:
            return f"replaced by the corrected series {self.replaced_by}"
        return "; ".join(self.problems)


@dataclass(frozen=True)
class DicomScan:
    """Every volume dcm2niix made from a folder."""

    folder: str
    volumes: tuple[DicomVolume, ...]
    version: str
    log: tuple[str, ...] = field(default=(), repr=False)

    @property
    def usable(self) -> list[DicomVolume]:
        return [v for v in self.volumes if v.usable]

    def get(self, series: str) -> DicomVolume | None:
        return next((v for v in self.volumes if v.id == str(series)), None)

    def select(self, series: str | None = None) -> DicomVolume:
        """The volume to process: ``series`` (an id), or the only usable volume.

        Raises :class:`DicomError` with the list of series when the choice is not clear.
        """
        name = Path(self.folder).name
        if series is not None and str(series).strip():
            series = str(series).strip()
            vol = self.get(series)
            if vol is None:
                ids = [v.id for v in self.volumes if v.id.split("_")[0] == series]
                usable = [v for v in self.usable if v.id in ids]
                if len(usable) == 1:
                    return usable[0]
                hint = f"; choose one of {', '.join(ids)}" if ids else ""
                raise DicomError(f"No series {series} in {name}{hint}.\n{self.describe()}")
            if vol.replaced_by:
                return self.select(vol.replaced_by)
            if vol.problems:
                raise DicomError(f"Series {vol.id} cannot be used: {vol.status()}")
            return vol
        usable = self.usable
        if len(usable) == 1:
            return usable[0]
        if not usable:
            raise DicomError(f"No usable 3D CT or MR series in {name}.\n{self.describe()}")
        raise DicomError(
            f"{len(usable)} usable series in {name}; choose one "
            f"({', '.join(v.id for v in usable)}).\n{self.describe()}"
        )

    def describe(self) -> str:
        """Every volume on its own line, with the reason when it cannot be used."""
        lines = []
        for v in self.volumes:
            status = v.status()
            lines.append(f"  {v.label()}" + (f" - not usable: {status}" if status else ""))
        return "\n".join(lines)


# ---------------------------------------------------------------------------- dcm2niix


def dcm2niix_executable() -> str | None:
    """dcm2niix of the environment (next to python.exe / in Library\\bin), else from PATH."""
    exe = "dcm2niix.exe" if sys.platform == "win32" else "dcm2niix"
    prefix = Path(sys.prefix)
    for folder in (prefix / "Library" / "bin", prefix / "Scripts", prefix / "bin"):
        if (folder / exe).is_file():
            return str(folder / exe)
    return shutil.which("dcm2niix")


def _require_exe() -> str:
    exe = dcm2niix_executable()
    if exe is None:
        raise DicomError(
            "dcm2niix was not found; it is part of the msk-pipe environment "
            "(reinstall the release, or run `pixi install`)"
        )
    return exe


def dcm2niix_version() -> str:
    """Version of dcm2niix, e.g. ``v1.0.20260724``."""
    exe = _require_exe()
    try:
        out = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=30, **_no_window()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DicomError(f"Cannot run dcm2niix: {exc}") from None
    text = out.stdout + out.stderr
    match = re.search(r"v\d+\.\d+\.\d+\w*", text)
    return match.group(0) if match else text.strip().splitlines()[-1] if text.strip() else "?"


def _no_window() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
    return {}


# ---------------------------------------------------------------------------- scan


def cache_root() -> Path:
    from mskpipe.resources.paths import home

    return home() / "cache" / "dicom"


def scan(folder: str | Path, *, cache_dir: Path | None = None) -> DicomScan:
    """Convert every series of ``folder`` (cached) and check the volumes.

    Raises :class:`DicomError` if the folder has no DICOM images or dcm2niix fails.
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise DicomError(f"DICOM folder not found: {folder}")
    files = _list_files(folder)
    if not files:
        raise DicomError(f"The folder is empty: {folder}")
    version = dcm2niix_version()
    root = cache_dir or cache_root()
    key = _cache_key(folder, files, version)
    target = root / key[:24]
    cached = _load_cached(target)
    if cached is not None:
        _touch(target)
        return cached

    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="convert-", dir=root))
    try:
        log = _run(folder, work / "out")
        result = _collect(folder, work / "out", log, version, target / "out")
        (work / "scan.json").write_text(json.dumps(_to_json(result), indent=1), encoding="utf-8")
        try:
            work.replace(target)  # another process may have finished first
        except OSError:
            if _load_cached(target) is None:
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _prune(root, keep=target)
    cached = _load_cached(target)
    if cached is None:
        raise DicomError(f"Cannot cache the conversion of {folder} in {root}")
    return cached


def _list_files(folder: Path, depth: int = 5) -> list[tuple[str, int, int]]:
    """(relative path, size, mtime_ns) of the files dcm2niix reads (5 levels deep)."""
    out = []
    base = len(folder.parts)
    for current, dirs, names in os.walk(folder):
        level = len(Path(current).parts) - base
        if level >= depth:
            dirs[:] = []
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            if name.startswith("."):
                continue
            path = Path(current) / name
            try:
                st = path.stat()
            except OSError:
                continue
            out.append((path.relative_to(folder).as_posix(), st.st_size, st.st_mtime_ns))
    return out


def _cache_key(folder: Path, files: list[tuple[str, int, int]], version: str) -> str:
    blob = json.dumps(
        {"format": CACHE_FORMAT, "dcm2niix": version, "folder": str(folder), "files": files},
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _run(folder: Path, out: Path) -> list[str]:
    out.mkdir(parents=True)
    argv = [_require_exe(), "-z", "y", "-b", "y", "-ba", "y", "-f", "%s", "-w", "2"]
    argv += ["-o", _native_path(out), _native_path(folder)]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_S,
            **_no_window(),
        )
    except subprocess.TimeoutExpired:
        raise DicomError(f"dcm2niix did not finish within {TIMEOUT_S // 60} min") from None
    except OSError as exc:
        raise DicomError(f"Cannot run dcm2niix: {exc}") from None
    log = [line.rstrip() for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]
    if not any(out.glob("*.nii*")):
        errors = [line.removeprefix("Error: ") for line in log if line.startswith("Error")]
        if any("Unable to find any DICOM" in e for e in errors):
            raise DicomError(f"No DICOM images in {folder} (or its subfolders)")
        detail = "; ".join(errors) or (log[-1] if log else f"exit code {proc.returncode}")
        raise DicomError(f"dcm2niix could not convert {folder.name}: {detail}")
    return log


def _native_path(path: Path) -> str:
    """Path as dcm2niix can open it. On Windows it reads its arguments in the ANSI code
    page, so a path with other characters (e.g. a user name with diacritics on an English
    Windows) is passed as its 8.3 short name, which is ASCII when short names exist."""
    text = str(path)
    if sys.platform != "win32" or text.isascii():
        return text
    import ctypes

    size = ctypes.windll.kernel32.GetShortPathNameW(text, None, 0)  # type: ignore[attr-defined]
    if size == 0:
        return text
    buffer = ctypes.create_unicode_buffer(size)
    ctypes.windll.kernel32.GetShortPathNameW(text, buffer, size)  # type: ignore[attr-defined]
    return buffer.value or text


def _notes_per_output(log: list[str]) -> dict[str, list[str]]:
    """dcm2niix messages per output name (a message precedes the Convert line of its series,
    except the ones in ``_AFTER_CONVERT``)."""
    notes: dict[str, list[str]] = {}
    pending: list[str] = []
    last: str | None = None
    for line in log:
        match = _CONVERT.match(line)
        if match:
            last = Path(match.group(2)).name
            notes.setdefault(last, []).extend(pending)
            pending = []
            continue
        if not line.startswith(_NOTE_PREFIXES) or any(b.lower() in line.lower() for b in _BENIGN):
            continue
        if line.startswith(_AFTER_CONVERT) and last is not None:
            notes[last].append(line)
        else:
            pending.append(line)
    if pending and last is not None:
        notes[last].extend(pending)
    return notes


def _collect(folder: Path, out: Path, log: list[str], version: str, final_out: Path) -> DicomScan:
    import nibabel as nib
    import numpy as np

    counts = {Path(m.group(2)).name: int(m.group(1)) for m in map(_CONVERT.match, log) if m}
    notes = _notes_per_output(log)
    names = sorted((p.name[: -len(".nii.gz")] for p in out.glob("*.nii.gz")), key=_natural)
    corrected = {}
    for name in names:
        m = _CORRECTED.match(name)
        if m and m.group("base") in names:
            corrected[m.group("base")] = (name, m.group("kind"))

    volumes = []
    for name in names:
        m = _CORRECTED.match(name)
        base = m.group("base") if m and m.group("base") in names else name
        meta = _read_meta(out / f"{name}.json") or _read_meta(out / f"{base}.json") or {}
        sidecar = name if (out / f"{name}.json").is_file() else base
        if not (out / f"{sidecar}.json").is_file():
            sidecar = None
        warnings = [_clean(n) for n in notes.get(base, [])]
        problems: list[str] = []
        try:
            img = nib.load(str(out / f"{name}.nii.gz"))
            shape = tuple(int(n) for n in img.shape)
            spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
            affine = np.asarray(img.affine, dtype=float)
            if not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-9:
                problems.append("invalid geometry (orientation / spacing)")
        except Exception as exc:  # nibabel raises many types for broken files
            shape, spacing = (), ()
            problems.append(f"unreadable output ({exc})")

        dicom_modality = str(meta.get("Modality", "")).upper()
        image_type = tuple(str(t).upper() for t in meta.get("ImageType", []) or [])
        if not dicom_modality:
            problems.append("modality unknown (no DICOM metadata)")
        elif dicom_modality not in _MODALITIES:
            problems.append(f"modality {dicom_modality} (only CT and MR can be processed)")
        if "LOCALIZER" in image_type:
            problems.append("localizer (scout) image")
        if shape:
            if len(shape) > 3 and any(n > 1 for n in shape[3:]):
                problems.append(f"{int(np.prod(shape[3:]))} volumes in one series (4D)")
            elif len(shape) < 3 or min(shape[:3]) < 2:
                problems.append("not a 3D volume (single slice)")
            if spacing and min(spacing) <= 0:
                problems.append("voxel spacing is zero")
        kind = m.group("kind") if m and base != name else None
        if kind == "Eq":
            warnings.append("irregular slice spacing: resampled to equidistant slices")
        elif kind == "Tilt":
            warnings.append("gantry tilt corrected")
        else:
            fix = corrected.get(name, ("", ""))[1]
            if any(_VARYING in w for w in warnings) and fix != "Eq":
                problems.append("irregular slice spacing (missing slices?), not correctable")
            if any("gantry tilt" in w.lower() for w in warnings) and fix != "Tilt":
                problems.append("gantry tilt that cannot be corrected")
        number = meta.get("SeriesNumber")
        if not isinstance(number, int):
            sn = _SERIES_NUMBER.match(name)
            number = int(sn.group(1)) if sn else None
        volumes.append(
            DicomVolume(
                id=name,
                nifti=str(final_out / f"{name}.nii.gz"),
                sidecar=str(final_out / f"{sidecar}.json") if sidecar else None,
                series_number=number,
                description=str(meta.get("SeriesDescription") or meta.get("ProtocolName") or ""),
                dicom_modality=dicom_modality,
                image_type=image_type,
                files=counts.get(base, 0),
                shape=shape,
                spacing=spacing,
                warnings=tuple(dict.fromkeys(warnings)),
                problems=tuple(problems),
                replaced_by=corrected.get(name, (None,))[0],
            )
        )
    return DicomScan(str(folder), tuple(volumes), version, tuple(log))


def _read_meta(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _clean(note: str) -> str:
    return note.removeprefix("Warning:").strip()


def _natural(name: str) -> list[Any]:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


# ---------------------------------------------------------------------------- cache


def _to_json(result: DicomScan) -> dict[str, Any]:
    return {"format": CACHE_FORMAT, **asdict(result)}


def _load_cached(target: Path) -> DicomScan | None:
    data = _read_meta(target / "scan.json")
    if not data or data.get("format") != CACHE_FORMAT:
        return None
    try:
        volumes = tuple(
            DicomVolume(
                **{
                    **v,
                    "image_type": tuple(v["image_type"]),
                    "shape": tuple(v["shape"]),
                    "spacing": tuple(v["spacing"]),
                    "warnings": tuple(v["warnings"]),
                    "problems": tuple(v["problems"]),
                }
            )
            for v in data["volumes"]
        )
    except (KeyError, TypeError):
        return None
    if not all(Path(v.nifti).is_file() for v in volumes):
        return None
    return DicomScan(data["folder"], volumes, data["version"], tuple(data.get("log", ())))


def _touch(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.utime(path)


def _prune(root: Path, keep: Path) -> None:
    """Keep the ``CACHE_KEEP`` most recently used conversions (and unfinished ones)."""
    done = [
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("convert-") and (p / "scan.json").is_file()
    ]
    done.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in done[CACHE_KEEP:]:
        if old != keep:
            shutil.rmtree(old, ignore_errors=True)
