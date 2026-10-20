# SPDX-License-Identifier: Apache-2.0
"""Attachment atlas location and installation (``mskpipe setup atlas``).

The atlas used by ``attachments/atlas_registration`` is looked up in this order:

1. ``attachments.params.atlas_dir`` (config),
2. the ``MSKPIPE_ATLAS_DIR`` environment variable (set by the Windows release launcher
   to the atlas shipped with the release),
3. ``<MSKPIPE_HOME>/atlas/lhdl`` (installed with ``mskpipe setup atlas``).

The LHDL atlas is distributed separately from the code under its own licence
(CC BY-NC-SA 2.0 BE, see the LICENSE file inside the atlas).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Literal

from mskpipe.resources.download import download, file_sha256
from mskpipe.resources.paths import atlas_root
from mskpipe.resources.tool import Log

ATLAS_ENV = "MSKPIPE_ATLAS_DIR"
ATLAS_INDEX = "atlas.json"
DEFAULT_ATLAS = "lhdl"

Source = Literal["config", "env", "home", "none"]


class AtlasError(RuntimeError):
    """The atlas cannot be found or installed."""


def default_atlas_dir() -> Path:
    return atlas_root() / DEFAULT_ATLAS


def resolve_atlas_dir(value: str | os.PathLike[str] | None = None) -> tuple[Path | None, Source]:
    """Atlas folder and where it came from (config value, environment, per-user folder)."""
    if value:
        return Path(value), "config"
    env = os.environ.get(ATLAS_ENV)
    if env:
        return Path(env), "env"
    folder = default_atlas_dir()
    if (folder / ATLAS_INDEX).is_file():
        return folder, "home"
    return None, "none"


def find_atlas_root(folder: Path) -> Path:
    """The folder with ``atlas.json``: ``folder`` itself or its only subfolder (a zip
    usually wraps the atlas in one top-level folder)."""
    if (folder / ATLAS_INDEX).is_file():
        return folder
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / ATLAS_INDEX).is_file():
        return subdirs[0]
    raise AtlasError(f"No {ATLAS_INDEX} in {folder} or in its only subfolder")


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip, refusing absolute paths and '..' (zip slip)."""
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or ":" in info.filename:
                raise AtlasError(f"Unsafe path in {archive.name}: {info.filename}")
        zf.extractall(dest)


def install_atlas(
    source: str,
    log: Log,
    *,
    name: str = DEFAULT_ATLAS,
    sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Install an atlas from a folder, a ``.zip`` file or an ``http(s)://``/``file://`` URL
    of a zip into ``<MSKPIPE_HOME>/atlas/<name>``. ``sha256`` checks the zip."""
    if not name or any(c in name for c in '\\/:*?"<>|') or name in (".", ".."):
        raise AtlasError(f"Invalid atlas name '{name}'")
    dest = atlas_root() / name
    if dest.exists() and not force:
        raise AtlasError(f"{dest} exists (use --force to replace it)")
    root_dir = atlas_root()
    root_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root_dir, prefix=".install-") as tmp:
        work = Path(tmp)
        if "://" in source:
            archive = work / "atlas.zip"
            log(f"Downloading {source}")
            download(source, archive, sha256=sha256)
            source_path = archive
        else:
            source_path = Path(source).expanduser()
            if not source_path.exists():
                raise AtlasError(f"{source_path} does not exist")
            if source_path.is_file() and sha256:
                digest = file_sha256(source_path)
                if digest != sha256.lower():
                    raise AtlasError(f"Checksum mismatch: expected {sha256}, got {digest}")
        staged = work / "atlas"
        if source_path.is_dir():
            shutil.copytree(find_atlas_root(source_path), staged)
        elif zipfile.is_zipfile(source_path):
            unpacked = work / "unpacked"
            try:
                safe_extract(source_path, unpacked)
            except zipfile.BadZipFile as exc:
                raise AtlasError(f"Invalid zip {source_path}: {exc}") from exc
            shutil.move(str(find_atlas_root(unpacked)), staged)
        else:
            raise AtlasError(f"{source_path} is neither a folder nor a zip file")
        if dest.exists():
            old = work / "old"
            dest.replace(old)
        staged.replace(dest)
    log(f"Atlas installed: {dest}")
    return dest
