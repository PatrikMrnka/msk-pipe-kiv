# SPDX-License-Identifier: Apache-2.0
"""Unified label scheme (``config/unified_labels.yaml``).

Maps the label names of each segmentation source (TotalSegmentator ``total``/``total_mr``,
TotalSegmentator ``appendicular_bones``, MuscleMap) to the unified msk-pipe structures
(``pelvis_no_sacrum``, ``femur_r``, ``gluteus_maximus_l``, ...). Label IDs of the tools are
never used here: tools are matched by label names only.
"""

from __future__ import annotations

import hashlib
from functools import cache
from importlib.resources import files
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mskpipe.config import PipelineConfig

SCHEME_FILE = "unified_labels.yaml"
Group = Literal["bones", "tibia_fibula", "muscles"]
GROUPS: tuple[Group, ...] = ("bones", "tibia_fibula", "muscles")
_NAME = r"^[a-z][a-z0-9_]*$"


class SchemeError(ValueError):
    """The unified label scheme is invalid."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Structure(_Model):
    value: int = Field(ge=1, le=255)
    kind: Literal["bone", "muscle"]
    group: Group


class Source(_Model):
    tool: str = Field(pattern=_NAME)
    tasks: tuple[str, ...] = Field(
        default=(), description="Tool tasks providing these labels; empty = any task."
    )
    labels: dict[str, tuple[str, ...]]

    def accepts(self, tool: str, task: str) -> bool:
        return tool == self.tool and (not self.tasks or task in self.tasks)

    def split_names(self) -> set[str]:
        """Source labels covering both legs (mapped to both ``x_l`` and ``x_r``)."""
        out: set[str] = set()
        for name, labels in self.labels.items():
            if name.endswith("_r"):
                twin = self.labels.get(name[:-2] + "_l", ())
                out |= set(labels) & set(twin)
        return out


class Scheme(_Model):
    structures: dict[str, Structure]
    sources: dict[str, Source]

    @model_validator(mode="after")
    def _check(self) -> Scheme:
        values = [s.value for s in self.structures.values()]
        if len(values) != len(set(values)):
            raise ValueError("structure values must be unique")
        for src_name, src in self.sources.items():
            unknown = set(src.labels) - set(self.structures)
            if unknown:
                raise ValueError(f"source '{src_name}' maps unknown structures {sorted(unknown)}")
            for name, labels in src.labels.items():
                if not labels:
                    raise ValueError(f"source '{src_name}': no labels for '{name}'")
            shared = _shared_names(src)
            for label, targets in shared.items():
                bases = {t[:-2] for t in targets if t.endswith(("_l", "_r"))}
                if len(targets) != 2 or len(bases) != 1:
                    raise ValueError(
                        f"source '{src_name}': label '{label}' may be shared only by one "
                        f"_l/_r pair, got {sorted(targets)}"
                    )
        return self

    def by_value(self) -> dict[int, str]:
        return {s.value: name for name, s in self.structures.items()}

    def selection(self, config: PipelineConfig) -> dict[Group, str]:
        """Source chosen for each group by ``config.segmentation``."""
        seg = config.segmentation
        chosen: dict[Group, str] = {
            "bones": str(seg.bones),
            "tibia_fibula": str(seg.tibia_fibula),
            "muscles": str(seg.muscles),
        }
        for group, source in chosen.items():
            if source not in self.sources:
                raise SchemeError(f"segmentation source '{source}' ({group}) not in the scheme")
        return chosen

    def plan(
        self, config: PipelineConfig, sides: tuple[str, ...] | None = None
    ) -> dict[str, tuple[str, tuple[str, ...]]]:
        """Structure -> (source, source label names) for the configured sources.

        Only structures of ``sides`` (default: ``skeleton.side``) and sideless ones
        (``pelvis_no_sacrum``) are planned. Structures a chosen source does not provide are
        left out (e.g. TotalSegmentator has only the gluteal muscles).
        """
        sides = sides or (config.skeleton.side,)
        chosen = self.selection(config)
        plan: dict[str, tuple[str, tuple[str, ...]]] = {}
        for name, struct in self.structures.items():
            side = structure_side(name)
            if side is not None and side not in sides:
                continue
            source = chosen[struct.group]
            labels = self.sources[source].labels.get(name)
            if labels:
                plan[name] = (source, labels)
        return plan


def structure_side(name: str) -> str | None:
    """``r``/``l`` for ``*_r``/``*_l`` structures, ``None`` for sideless ones."""
    return name[-1] if name.endswith(("_r", "_l")) else None


def _shared_names(src: Source) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for name, labels in src.labels.items():
        for label in labels:
            owners.setdefault(label, []).append(name)
    return {label: names for label, names in owners.items() if len(names) > 1}


def scheme_text() -> str:
    return files("mskpipe.config").joinpath(SCHEME_FILE).read_text(encoding="utf-8")


def scheme_sha256() -> str:
    """Hash of the bundled scheme file (part of the labelmap step's cache key)."""
    return hashlib.sha256(scheme_text().encode("utf-8")).hexdigest()


def parse_scheme(text: str) -> Scheme:
    try:
        return Scheme.model_validate(yaml.safe_load(text))
    except (yaml.YAMLError, ValidationError) as exc:
        raise SchemeError(f"Invalid label scheme: {exc}") from exc


@cache
def load_scheme() -> Scheme:
    """The bundled unified label scheme."""
    return parse_scheme(scheme_text())
