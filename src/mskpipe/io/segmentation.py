# SPDX-License-Identifier: Apache-2.0
"""Index of the raw segmentations (``01_segment/segmentation.json``).

The ``segment`` step keeps every tool output as written by the tool (one multi-label image
per tool task, in the tool's own label scheme, on the input image grid) and lists them here
with the tool's name -> ID map. The ``labelmap`` step maps them to the unified scheme by
names (:mod:`mskpipe.labelmap.scheme`).

Example::

    {
      "format": "mskpipe.segmentation",
      "version": 1,
      "outputs": [
        {"tool": "totalsegmentator", "task": "total", "file": "totalsegmentator_total.nii.gz",
         "labels": {"femur_left": 75, "femur_right": 76, "hip_left": 77, "hip_right": 78}},
        {"tool": "musclemap", "task": "default", "file": "musclemap.nii.gz",
         "labels": {"gluteus_maximus_l": 6121, "gluteus_maximus_r": 6122}}
      ]
    }
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SEGMENTATION_FILE = "segmentation.json"
FORMAT = "mskpipe.segmentation"
VERSION = 1


class SegmentationIndexError(ValueError):
    """The segmentation index is missing or invalid."""


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    task: str = Field("", description="Tool task (e.g. total, total_mr, appendicular_bones).")
    file: str = Field(description="Path relative to the index (POSIX separators).")
    labels: dict[str, int] = Field(description="Tool label name -> label value.")

    @model_validator(mode="after")
    def _check(self) -> ToolOutput:
        bad = {name: v for name, v in self.labels.items() if not 1 <= v <= 65535}
        if bad:
            raise ValueError(f"label values must be in 1..65535: {bad}")
        return self


class SegmentationIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: str = FORMAT
    version: int = VERSION
    outputs: tuple[ToolOutput, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> SegmentationIndex:
        if self.format != FORMAT or self.version != VERSION:
            raise ValueError(f"expected format '{FORMAT}' version {VERSION}")
        files = [o.file for o in self.outputs]
        if len(files) != len(set(files)):
            raise ValueError("duplicate output files")
        return self

    def __iter__(self) -> Iterator[ToolOutput]:  # type: ignore[override]
        return iter(self.outputs)

    def __len__(self) -> int:
        return len(self.outputs)

    @classmethod
    def load(cls, path: str | Path) -> SegmentationIndex:
        path = Path(path)
        try:
            return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise SegmentationIndexError(f"Segmentation index not found: {path}") from None
        except (json.JSONDecodeError, ValidationError) as exc:
            raise SegmentationIndexError(f"Invalid segmentation index {path}: {exc}") from exc

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path
