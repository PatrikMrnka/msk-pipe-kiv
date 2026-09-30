# SPDX-License-Identifier: Apache-2.0
"""Label table of the unified multi-label map (``02_labelmap/labels.json``).

The ``labelmap`` step writes one integer label map (``labelmap.nii.gz``) and this
table, which maps each structure name to its label value and kind. Downstream steps
look structures up by name only; label values are never hard-coded.

Example::

    {
      "format": "mskpipe.labels",
      "version": 1,
      "labels": [
        {"name": "femur_r", "value": 1, "kind": "bone"},
        {"name": "gluteus_maximus_r", "value": 21, "kind": "muscle"}
      ]
    }
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

LABELMAP_FILE = "labelmap.nii.gz"
LABELS_FILE = "labels.json"
FORMAT = "mskpipe.labels"
VERSION = 1


class LabelTableError(ValueError):
    """The label table is missing or invalid."""


class LabelKind(StrEnum):
    BONE = "bone"
    MUSCLE = "muscle"


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    value: int = Field(ge=1, le=65535)
    kind: LabelKind


class LabelTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: str = FORMAT
    version: int = VERSION
    labels: tuple[Label, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> LabelTable:
        if self.format != FORMAT or self.version != VERSION:
            raise ValueError(f"expected format '{FORMAT}' version {VERSION}")
        for attr in ("name", "value"):
            seen = [getattr(label, attr) for label in self.labels]
            dupes = sorted({v for v in seen if seen.count(v) > 1}, key=str)
            if dupes:
                raise ValueError(f"duplicate label {attr}s: {dupes}")
        return self

    def __iter__(self) -> Iterator[Label]:  # type: ignore[override]
        return iter(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def get(self, name: str) -> Label:
        for label in self.labels:
            if label.name == name:
                return label
        raise KeyError(name)

    def of_kind(self, kind: LabelKind | str) -> tuple[Label, ...]:
        return tuple(label for label in self.labels if label.kind == kind)

    @classmethod
    def load(cls, path: str | Path) -> LabelTable:
        path = Path(path)
        try:
            return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise LabelTableError(f"Label table not found: {path}") from None
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LabelTableError(f"Invalid label table {path}: {exc}") from exc

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path
