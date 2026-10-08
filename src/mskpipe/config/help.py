# SPDX-License-Identifier: Apache-2.0
"""Explanations of the configuration for users (``help.yaml``); the GUI shows them behind
a '?' next to each setting.

Keys are dotted config paths; plugin parameters use ``plugins.<kind>.<plugin>.<param>``
and GUI controls ``gui.<control>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources

import yaml

__all__ = ["HELP_FILE", "Help", "help_for", "load_help", "plugin_help"]

HELP_FILE = "help.yaml"


@dataclass(frozen=True)
class Help:
    text: str
    example: str | None = None


@cache
def load_help() -> dict[str, Help]:
    text = resources.files("mskpipe.config").joinpath(HELP_FILE).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    out: dict[str, Help] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("help"), str):
            raise ValueError(f"{HELP_FILE}: '{key}' needs a 'help' text")
        example = entry.get("example")
        out[key] = Help(" ".join(entry["help"].split()), None if example is None else str(example))
    return out


def help_for(path: str | tuple[str, ...]) -> Help | None:
    key = path if isinstance(path, str) else ".".join(path)
    return load_help().get(key)


def plugin_help(kind: str, plugin: str, param: str) -> Help | None:
    return load_help().get(f"plugins.{kind}.{plugin}.{param}")
