# SPDX-License-Identifier: Apache-2.0
"""TotalSegmentator licence key (needed by ``appendicular_bones``, free for
non-commercial use: https://backend.totalsegmentator.com/license-academic/).

The key lives in TotalSegmentator's own ``config.json``; it is read here directly
because ``totalsegmentator.config`` imports torch, and written by TotalSegmentator's
``totalseg_set_license`` (which also validates it online).
"""

from __future__ import annotations

import json
import sys

from mskpipe.resources.paths import totalseg_home
from mskpipe.resources.tool import Log, run_tool

LICENCE_PREFIX = "aca_"
LICENCE_LENGTH = 18  # TotalSegmentator's own offline check (has_valid_license_offline)
LICENCE_URL = "https://backend.totalsegmentator.com/license-academic/"
SET_LICENCE_MODULE = "totalsegmentator.bin.totalseg_set_license"


class LicenceError(RuntimeError):
    """The licence key is malformed or could not be stored."""


def licence_number() -> str | None:
    """Stored key (``$TOTALSEG_HOME_DIR`` or ``~/.totalsegmentator``, ``config.json``)."""
    try:
        data = json.loads((totalseg_home() / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    key = data.get("license_number") if isinstance(data, dict) else None
    return key if isinstance(key, str) and key else None


def format_problem(key: str) -> str | None:
    """Why ``key`` cannot be a TotalSegmentator key, or None."""
    if not key.startswith(LICENCE_PREFIX):
        return f"The key must start with '{LICENCE_PREFIX}'"
    if len(key) != LICENCE_LENGTH:
        return f"The key must have {LICENCE_LENGTH} characters, got {len(key)}"
    return None


def set_licence_command(key: str, *, skip_validation: bool = False) -> list[str]:
    argv = [sys.executable, "-m", SET_LICENCE_MODULE, "-l", key]
    return [*argv, "-sv"] if skip_validation else argv


def set_licence(key: str, log: Log, *, skip_validation: bool = False) -> None:
    """Validate the format, store the key with TotalSegmentator, check it was stored."""
    key = key.strip()
    problem = format_problem(key)
    if problem:
        raise LicenceError(problem)
    code = run_tool(set_licence_command(key, skip_validation=skip_validation), log)
    if code != 0:
        raise LicenceError(f"totalseg_set_license failed (exit code {code})")
    if licence_number() != key:
        raise LicenceError(f"The key was not stored in {totalseg_home() / 'config.json'}")
