# SPDX-License-Identifier: Apache-2.0
"""Model weights of the segmenters: status and download in advance (offline computers).

* **TotalSegmentator** keeps its weights in its own folder (``nnunet/results``); they are
  downloaded per task with its ``totalseg_download_weights``. Licensed tasks
  (``appendicular_bones[_mr]``) need the licence key first.
* **MuscleMap** caches its model inside the installed package (``scripts/models``), i.e.
  in one environment. A copy is kept in ``<MSKPIPE_HOME>/weights/musclemap`` and linked
  (hard link, else copy) into every environment that lacks it, so a new release or the
  GPU environment does not download it again.

Weights are never redistributed with msk-pipe; they come from the tools' official sources.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from mskpipe.resources.paths import totalseg_weights_dir, weights_dir
from mskpipe.resources.tool import Log, run_tool

if TYPE_CHECKING:
    from mskpipe.config import PipelineConfig

Status = Literal["ok", "missing", "unknown"]

# totalseg_download_weights.TASK_TO_ID (TotalSegmentator 2.18) for the tasks msk-pipe uses
TS_TASK_IDS: dict[str, tuple[int, ...]] = {
    "total": (291, 292, 293, 294, 295, 298),
    "total_mr": (850, 851),
    "appendicular_bones": (304,),
    "appendicular_bones_mr": (855,),
}
TS_DOWNLOAD_MODULE = "totalsegmentator.bin.totalseg_download_weights"
MM_DOWNLOAD = "from scripts.mm_util import ensure_model_downloaded as e; e({region!r}, {version!r})"


class WeightsError(RuntimeError):
    """Weights could not be downloaded or linked."""


class WeightsItem(BaseModel):
    tool: Literal["totalsegmentator", "musclemap"]
    name: str  # task (TotalSegmentator) or region/version (MuscleMap)
    status: Status
    path: str | None = None
    detail: str = ""
    licensed: bool = False


# ---------------------------------------------------------------- TotalSegmentator


def required_ts_tasks(config: PipelineConfig, modalities: Iterable[str]) -> list[str]:
    """TotalSegmentator tasks the configuration runs for the given modalities."""
    from mskpipe.labelmap.scheme import load_scheme
    from mskpipe.plugins.segmenters.totalsegmentator import plan_runs

    scheme = load_scheme()
    tasks: list[str] = []
    for modality in modalities:
        request = scheme.requests(config, modality).get("totalsegmentator")
        if request is not None:
            tasks += [t for t in plan_runs(request) if t not in tasks]
    return tasks


def ts_folders(task: str) -> list[Path] | None:
    """Weight folders of ``task`` (from TotalSegmentator's task table, a plain dict
    module), or None when TotalSegmentator is not installed or the task is unknown."""
    ids = TS_TASK_IDS.get(task)
    if ids is None:
        return None
    try:
        from totalsegmentator.map_tasks_config import TASK_ID_WEIGHTS_CONFIGS
    except ImportError:
        return None
    root = totalseg_weights_dir()
    folders = []
    for task_id in ids:
        info = TASK_ID_WEIGHTS_CONFIGS.get(task_id)
        if not info or "foldername" not in info:
            return None
        folders.append(root / info.get("rel_path", "") / info["foldername"])
    return folders


def ts_status(task: str) -> WeightsItem:
    from mskpipe.plugins.segmenters.totalsegmentator import LICENSED_TASKS

    licensed = task in LICENSED_TASKS
    folders = ts_folders(task)
    if folders is None:
        return WeightsItem(
            tool="totalsegmentator",
            name=task,
            status="unknown",
            detail="TotalSegmentator is not installed in this environment",
            licensed=licensed,
        )
    missing = [f.name for f in folders if not f.is_dir()]
    return WeightsItem(
        tool="totalsegmentator",
        name=task,
        status="missing" if missing else "ok",
        path=str(totalseg_weights_dir()),
        detail=f"missing {', '.join(missing)}" if missing else f"{len(folders)} model(s)",
        licensed=licensed,
    )


def ts_download_command(task: str) -> list[str]:
    return [sys.executable, "-m", TS_DOWNLOAD_MODULE, "-t", task]


def download_ts(task: str, log: Log) -> WeightsItem:
    """Download the weights of one TotalSegmentator task (skips present ones)."""
    from mskpipe.resources.licence import licence_number

    item = ts_status(task)
    if item.status == "ok":
        log(f"TotalSegmentator {task}: already present")
        return item
    if item.licensed and licence_number() is None:
        raise WeightsError(
            f"TotalSegmentator {task} needs the licence key first: mskpipe setup licence <key>"
        )
    log(f"TotalSegmentator {task}: downloading ...")
    code = run_tool(ts_download_command(task), log)
    item = ts_status(task)
    if code != 0 or item.status != "ok":
        raise WeightsError(f"TotalSegmentator {task}: download failed (exit code {code})")
    return item


# ---------------------------------------------------------------- MuscleMap


def mm_files(version: str) -> tuple[str, ...]:
    from mskpipe.plugins.segmenters.musclemap import MODEL_JSON, MODEL_PTH

    return (MODEL_PTH, MODEL_JSON)


def mm_store_dir(version: str) -> Path:
    from mskpipe.plugins.segmenters.musclemap import REGION

    return weights_dir() / "musclemap" / REGION / f"v{version}"


def mm_env_dir(version: str) -> Path | None:
    """MuscleMap's own cache folder of ``version``, or None if MuscleMap is not installed."""
    from mskpipe.plugins.segmenters.musclemap import MuscleMapError, model_json_path

    try:
        return model_json_path(version).parent
    except MuscleMapError:
        return None


def _complete(folder: Path | None, names: Iterable[str]) -> bool:
    if folder is None:
        return False
    for name in names:
        try:
            if (folder / name).stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def _link_or_copy(src: Path, dst: Path) -> None:
    """``dst`` = ``src`` as a hard link (same volume) or a copy, replaced atomically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    tmp.replace(dst)


def sync_musclemap(version: str) -> Literal["stored", "restored", "none"]:
    """Make the environment's MuscleMap cache and the per-user store hold the same model:
    environment -> store after a download, store -> environment for a new environment."""
    names = mm_files(version)
    env, store = mm_env_dir(version), mm_store_dir(version)
    if env is None:
        return "none"
    if _complete(env, names):
        if _complete(store, names):
            return "none"
        for name in names:
            _link_or_copy(env / name, store / name)
        return "stored"
    if _complete(store, names):
        for name in names:
            _link_or_copy(store / name, env / name)
        return "restored"
    return "none"


def mm_status(version: str) -> WeightsItem:
    from mskpipe.plugins.segmenters.musclemap import REGION

    name = f"{REGION} v{version}"
    names = mm_files(version)
    env, store = mm_env_dir(version), mm_store_dir(version)
    if _complete(env, names):
        return WeightsItem(tool="musclemap", name=name, status="ok", path=str(env))
    if _complete(store, names):
        detail = "in the per-user store; linked into the environment on the first run"
        return WeightsItem(tool="musclemap", name=name, status="ok", path=str(store), detail=detail)
    if env is None:
        return WeightsItem(
            tool="musclemap",
            name=name,
            status="unknown",
            detail="MuscleMap is not installed in this environment",
        )
    return WeightsItem(tool="musclemap", name=name, status="missing", path=str(env))


def mm_download_command(version: str) -> list[str]:
    from mskpipe.plugins.segmenters.musclemap import REGION

    return [sys.executable, "-c", MM_DOWNLOAD.format(region=REGION, version=version)]


def download_musclemap(version: str, log: Log) -> WeightsItem:
    """Download the pinned MuscleMap model (MuscleMap's own Zenodo download) and keep a
    copy in the per-user store."""
    if mm_env_dir(version) is None:
        raise WeightsError("MuscleMap is not installed in this environment")
    try:
        if sync_musclemap(version) == "restored":
            log(f"MuscleMap {version}: linked from {mm_store_dir(version)}")
    except OSError as exc:
        log(f"MuscleMap {version}: cannot use the per-user store: {exc}")
    if not _complete(mm_env_dir(version), mm_files(version)):
        log(f"MuscleMap {version}: downloading ...")
        code = run_tool(mm_download_command(version), log)
        if code != 0 or not _complete(mm_env_dir(version), mm_files(version)):
            raise WeightsError(f"MuscleMap {version}: download failed (exit code {code})")
    else:
        log(f"MuscleMap {version}: already present")
    try:
        sync_musclemap(version)
    except OSError as exc:
        log(f"MuscleMap {version}: not copied to the per-user store: {exc}")
    return mm_status(version)


# ---------------------------------------------------------------- both


def weights_status(
    config: PipelineConfig, modalities: Iterable[str] = ("ct", "mri")
) -> list[WeightsItem]:
    """Status of every weight file the configuration needs for the given modalities."""
    from mskpipe.labelmap.scheme import load_scheme
    from mskpipe.plugins.segmenters.musclemap import model_version

    modalities = tuple(modalities)
    items = [ts_status(t) for t in required_ts_tasks(config, modalities)]
    scheme = load_scheme()
    if any("musclemap" in scheme.requests(config, m) for m in modalities):
        items.append(mm_status(model_version(config)))
    return items


def tool_weights_missing(config: PipelineConfig, modality: str, tool: str) -> bool:
    """Whether ``tool`` will download weights when it runs (a run then holds the setup
    lock, see :mod:`mskpipe.resources.lock`)."""
    return any(
        i.tool == tool and i.status == "missing" for i in weights_status(config, (modality,))
    )
