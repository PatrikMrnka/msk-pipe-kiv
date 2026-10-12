# SPDX-License-Identifier: Apache-2.0
"""Build information baked into release wheels (``mskpipe/build_info.json``).

Packages built into wheels for the Windows release (pixi-pack cannot pack git or source
packages) lose the git commit that ``pip``/``uv`` record in ``direct_url.json``, and the
installed release has no git checkout of msk-pipe. ``packaging/build_release.py wheels``
therefore writes this file before building the mskpipe wheel::

    {"commit": "<msk-pipe commit>", "packages": {"pystaple": "<commit>", "scripts": "..."}}

The manifest and the skeleton parity check fall back to it. Source checkouts have no such
file and use git and ``direct_url.json`` as before.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

BUILD_FILE = "build_info.json"


@cache
def load() -> dict[str, Any]:
    try:
        text = files("mskpipe").joinpath(BUILD_FILE).read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def commit() -> str | None:
    value = load().get("commit")
    return value if isinstance(value, str) and value else None


def package_commit(dist_name: str) -> str | None:
    packages = load().get("packages") or {}
    value = packages.get(dist_name.lower()) if isinstance(packages, dict) else None
    return value if isinstance(value, str) and value else None
