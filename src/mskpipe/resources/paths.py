# SPDX-License-Identifier: Apache-2.0
"""Per-user data folder of msk-pipe (``MSKPIPE_HOME``).

Layout (shared by all msk-pipe versions and environments of the user)::

    <home>/
        weights/musclemap/<region>/v<version>/   MuscleMap model files (store)
        atlas/<name>/                            attachment atlases (``mskpipe setup atlas``)

TotalSegmentator keeps its own folder (``TOTALSEG_HOME_DIR`` or ``~/.totalsegmentator``:
licence key in ``config.json``, weights in ``nnunet/results``) so that an existing
installation is reused.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOME_ENV = "MSKPIPE_HOME"
APP_DIR = "msk-pipe"
TOTALSEG_HOME_ENV = "TOTALSEG_HOME_DIR"


def home() -> Path:
    """``$MSKPIPE_HOME``, else ``%LOCALAPPDATA%\\msk-pipe`` (Windows) or
    ``$XDG_DATA_HOME/msk-pipe`` (``~/.local/share/msk-pipe``)."""
    value = os.environ.get(HOME_ENV)
    if value:
        return Path(value).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / APP_DIR
        return Path.home() / "AppData" / "Local" / APP_DIR
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home() / ".local" / "share") / APP_DIR


def weights_dir() -> Path:
    return home() / "weights"


def atlas_root() -> Path:
    return home() / "atlas"


def totalseg_home() -> Path:
    """TotalSegmentator's folder (same rule as ``totalsegmentator.config.get_totalseg_dir``)."""
    value = os.environ.get(TOTALSEG_HOME_ENV)
    return Path(value) if value else Path.home() / ".totalsegmentator"


def totalseg_weights_dir() -> Path:
    """TotalSegmentator's weights (``TOTALSEG_WEIGHTS_PATH`` or ``<home>/nnunet/results``)."""
    value = os.environ.get("TOTALSEG_WEIGHTS_PATH")
    return Path(value) if value else totalseg_home() / "nnunet" / "results"
