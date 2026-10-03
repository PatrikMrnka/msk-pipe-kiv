# SPDX-License-Identifier: Apache-2.0
"""Plugin interfaces: segmenters, skeleton backends and attachment methods.

The registry imports plugin modules to list and validate them, so a plugin module must stay
light: import heavy dependencies (torch, VTK, SimpleITK, pystaple, ...) inside methods only.
"""

from __future__ import annotations

import importlib.util
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from mskpipe.config import PipelineConfig
    from mskpipe.core.step import StepContext

PluginKind = Literal["segmenter", "skeleton", "attachments"]
KINDS: tuple[PluginKind, ...] = ("segmenter", "skeleton", "attachments")


class PluginParams(BaseModel):
    """Base of plugin parameter models; unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Plugin(ABC):
    """Common metadata of all plugins. Subclass one of the kind-specific bases below."""

    kind: ClassVar[PluginKind]
    name: ClassVar[str]
    version: ClassVar[str] = "1"  # bump when a code change alters outputs (invalidates cache)
    description: ClassVar[str] = ""
    Params: ClassVar[type[PluginParams]] = PluginParams
    requires_modules: ClassVar[tuple[str, ...]] = ()
    requires_executables: ClassVar[tuple[str, ...]] = ()
    supports_gpu: ClassVar[bool] = False

    @classmethod
    def cache_identity(cls, config: PipelineConfig) -> Mapping[str, Any]:
        """Settings outside the step's config sections that change the plugin's outputs.

        Mixed into the step fingerprint (e.g. a pinned model version). Default: none.
        """
        return {}

    @classmethod
    def missing(cls) -> list[str]:
        """Dependencies not found in the current environment (empty list = available)."""
        missing = [m for m in cls.requires_modules if not _has_module(m)]
        missing += [e for e in cls.requires_executables if shutil.which(e) is None]
        return missing


# ------------------------------------------------------------------------------- segmenters


@dataclass(frozen=True)
class TaskRequest:
    """Labels needed from one tool task.

    ``tasks`` are the tool tasks allowed by the unified label scheme for these labels
    (e.g. ``("total", "total_mr")``); the plugin runs the one suited to the modality. Empty
    = the tool's default task. ``labels`` are tool label names (``hip_left``, ``ilium_r``).
    """

    tasks: tuple[str, ...]
    labels: frozenset[str]


@dataclass(frozen=True)
class SegmentationRequest:
    """What the ``segment`` step needs from one segmenter."""

    modality: str  # "ct" | "mri"
    jobs: tuple[TaskRequest, ...]

    @property
    def labels(self) -> frozenset[str]:
        return frozenset().union(*(job.labels for job in self.jobs))


@dataclass(frozen=True)
class SegmentationOutput:
    """Multi-label image in the tool's own label scheme and its name -> ID map.

    The map must be read from the tool (never hard-coded). Names must match the tool's
    entries in ``config/unified_labels.yaml``; the ``labelmap`` step maps them to the
    unified scheme. ``task`` names the tool task (``total``, ``total_mr``,
    ``appendicular_bones``, ...).
    """

    image: Path
    labels: Mapping[str, int]
    task: str = ""


class SegmenterPlugin(Plugin):
    kind: ClassVar[PluginKind] = "segmenter"
    modalities: ClassVar[frozenset[str]] = frozenset({"ct", "mri"})

    @abstractmethod
    def segment(
        self,
        ctx: StepContext,
        image: Path,
        out_dir: Path,
        params: PluginParams,
        request: SegmentationRequest,
    ) -> list[SegmentationOutput]:
        """Segment ``image`` into ``out_dir``; one multi-label output per tool task run.

        Outputs must lie on the voxel grid of ``image``. Run external tools through
        ``ctx.run_command`` (logging, device, cancellation).
        """


# --------------------------------------------------------------------------------- skeleton


class SkeletonPlugin(Plugin):
    kind: ClassVar[PluginKind] = "skeleton"

    @abstractmethod
    def build(
        self, ctx: StepContext, bones: Mapping[str, Path], out_dir: Path, params: PluginParams
    ) -> Path:
        """Build the skeletal model from bone meshes (name -> mesh); return the ``.osim``."""


# ------------------------------------------------------------------------------ attachments


class AttachmentsPlugin(Plugin):
    kind: ClassVar[PluginKind] = "attachments"

    @abstractmethod
    def compute(
        self,
        ctx: StepContext,
        bones: Mapping[str, Path],
        muscles: Mapping[str, Path],
        out_dir: Path,
        params: PluginParams,
    ) -> Mapping[str, Path]:
        """Compute attachment areas; return output files by muscle name."""


KIND_BASES: dict[PluginKind, type[Plugin]] = {
    "segmenter": SegmenterPlugin,
    "skeleton": SkeletonPlugin,
    "attachments": AttachmentsPlugin,
}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # missing parent package of a dotted name
        return False
