# SPDX-License-Identifier: Apache-2.0
"""What a run needs besides the code, in one report (``mskpipe setup status``)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from mskpipe.resources.atlas import ATLAS_INDEX, Source, resolve_atlas_dir
from mskpipe.resources.fast_rnrr import FastRnrrStatus, fast_rnrr_status, locate
from mskpipe.resources.gpu import GpuStatus, gpu_status
from mskpipe.resources.licence import format_problem, licence_number
from mskpipe.resources.paths import home, totalseg_home
from mskpipe.resources.weights import WeightsItem, required_ts_tasks, weights_status

if TYPE_CHECKING:
    from mskpipe.config import PipelineConfig


class LicenceStatus(BaseModel):
    needed: bool  # a configured task is licensed
    present: bool
    problem: str | None = None  # malformed stored key
    file: str


class AtlasStatus(BaseModel):
    needed: bool
    path: str | None = None
    source: Source
    ok: bool


class FastRnrrNeed(BaseModel):
    needed: bool  # attachments.params.nonrigid == fast_rnrr
    path: str | None = None  # executable a run would use
    source: str = "none"  # config, env, build, path or none
    build: FastRnrrStatus | None = None  # the build of `mskpipe setup fast-rnrr`

    @property
    def ok(self) -> bool:
        return self.path is not None


class SetupStatus(BaseModel):
    home: str
    totalseg_home: str
    licence: LicenceStatus
    weights: list[WeightsItem]
    atlas: AtlasStatus
    gpu: GpuStatus | None = None  # None: the GPU wheel list could not be read
    fast_rnrr: FastRnrrNeed | None = None

    @property
    def ready(self) -> bool:
        """Everything present (weights may still be downloaded on the first run)."""
        licence_ok = not self.licence.needed or (self.licence.present and not self.licence.problem)
        atlas_ok = not self.atlas.needed or self.atlas.ok
        rnrr_ok = self.fast_rnrr is None or not self.fast_rnrr.needed or self.fast_rnrr.ok
        weights_ok = all(w.status == "ok" for w in self.weights)
        return licence_ok and atlas_ok and rnrr_ok and weights_ok


def collect(config: PipelineConfig, modalities: Iterable[str] = ("ct", "mri")) -> SetupStatus:
    from mskpipe.plugins.segmenters.totalsegmentator import LICENSED_TASKS

    modalities = tuple(modalities)
    key = licence_number()
    licence = LicenceStatus(
        needed=bool(set(required_ts_tasks(config, modalities)) & LICENSED_TASKS),
        present=key is not None,
        problem=format_problem(key) if key else None,
        file=str(totalseg_home() / "config.json"),
    )
    needs_atlas = config.attachments.method == "atlas_registration"
    folder, source = resolve_atlas_dir(config.attachments.params.get("atlas_dir"))
    atlas = AtlasStatus(
        needed=needs_atlas,
        path=str(folder) if folder else None,
        source=source,
        ok=folder is not None and (folder / ATLAS_INDEX).is_file(),
    )
    return SetupStatus(
        home=str(home()),
        totalseg_home=str(totalseg_home()),
        licence=licence,
        weights=weights_status(config, modalities),
        atlas=atlas,
        gpu=_gpu(),
        fast_rnrr=_fast_rnrr(config),
    )


def _gpu() -> GpuStatus | None:
    try:
        return gpu_status()
    except (OSError, ValueError):
        return None


def _fast_rnrr(config: PipelineConfig) -> FastRnrrNeed:
    params = config.attachments.params
    needed = config.attachments.method == "atlas_registration" and (
        params.get("nonrigid") == "fast_rnrr"
    )
    path, source = locate(params.get("fast_rnrr_exe"))
    try:
        build = fast_rnrr_status()
    except (OSError, ValueError):
        build = None
    return FastRnrrNeed(needed=needed, path=str(path) if path else None, source=source, build=build)
