# SPDX-License-Identifier: Apache-2.0
"""Tables for the paper from finished runs (``mskpipe stats``).

Sources: ``manifest.json`` (environment, device, per-step resources and metrics),
``config.resolved.yaml`` (settings that differ from the defaults), ``06_mw2_input/export.json``
(muscles and attachment areas) and ``batch.json`` of batches (experiment, job, variant,
repeat). Paths may be run folders, folders of runs (``runtime.runs_dir``, its
``batches/*/batch.json`` are read too) or batch folders.

CSV files (UTF-8, one row per ...):

* ``runs.csv`` run; ``steps.csv`` run x step; ``tools.csv`` run x segmentation tool;
* ``muscles.csv`` run x muscle; ``areas.csv`` run x muscle x origin/insertion;
* ``registration.csv`` run x atlas bone;
* ``summary.csv`` mean, SD, min, max of the wall time per (experiment, job, variant,
  subject, modality, device, step), from steps executed in the run (cached steps and
  failed runs are left out); step ``total`` = runs that executed every step.

Times are wall-clock seconds; memory is the sampled peak RSS of the process tree (GB =
2^30 bytes). GPU memory is not reported (device-wide NVML value, see core.metrics).
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mskpipe.core.manifest import Manifest
from mskpipe.core.workspace import CONFIG_FILE, INPUT_FILE, MANIFEST_FILE, STEP_DIRS, STEPS

__all__ = ["COLUMNS", "StatsTables", "collect", "find_runs", "write_tables"]

GB = 2**30
PIPELINE_STEPS = STEPS[1:]
EXPORT_FILE = "export.json"
BATCH_FILE = "batch.json"

_KEY = ("run_id", "experiment", "job", "variant", "repeat", "subject", "modality", "device")
COLUMNS: dict[str, tuple[str, ...]] = {
    "runs": (
        *_KEY,
        "status",
        "created_at",
        "finished_at",
        "input_sha256",
        "device_requested",
        "device_reason",
        "gpu_name",
        "driver_version",
        "cpu",
        "cpu_logical",
        "cpu_physical",
        "ram_total_gb",
        "platform",
        "python",
        "mskpipe",
        "git_commit",
        "git_dirty",
        "totalsegmentator",
        "nnunetv2",
        "musclemap_commit",
        "pystaple_commit",
        "torch",
        "vtk",
        "cache",
        "threads",
        "config_diff",
        "n_steps_completed",
        "n_steps_cached",
        "executed_wall_s",
        "pipeline_wall_s",
        "peak_rss_gb",
        "n_muscles_selected",
        "n_muscles_exported",
        "n_muscles_excluded",
        "n_areas_inflated",
        "n_body_mismatch",
        "error",
        "run_dir",
    ),
    "steps": (
        *_KEY,
        "step",
        "status",
        "cached_from",
        "wall_s",
        "cpu_s",
        "peak_rss_gb",
        "step_time_s",
    ),
    "tools": (
        *_KEY,
        "tool",
        "tasks",
        "tool_device",
        "time_s",
        "n_requested",
        "not_provided",
        "empty",
    ),
    "muscles": (
        *_KEY,
        "muscle",
        "status",
        "excluded_by_config",
        "problems",
        "n_warnings",
        "repaired",
        "mesh_ok",
        "genus",
        "components",
        "ori_ins_distance_mm",
        "mean_edge_mm",
    ),
    "areas": (
        *_KEY,
        "muscle",
        "kind",
        "muscle_status",
        "body",
        "nearest_body",
        "body_mismatch",
        "n_points",
        "distinct",
        "span_mm",
        "shrink",
        "crossing",
        "patch_fraction",
        "gap_mean_mm",
        "gap_max_mm",
        "inflated",
        "inflated_radius_mm",
        "inflated_distinct",
        "inflated_span_mm",
    ),
    "registration": (
        *_KEY,
        "bone",
        "method",
        "status",
        "scale",
        "rigid_mean_mm",
        "rigid_p95_mm",
        "rigid_trimmed_mean_mm",
        "nonrigid_mean_mm",
        "nonrigid_p95_mm",
        "nonrigid_trimmed_mean_mm",
    ),
    "summary": (
        "experiment",
        "job",
        "variant",
        "subject",
        "modality",
        "device",
        "step",
        "n",
        "wall_mean_s",
        "wall_sd_s",
        "wall_min_s",
        "wall_max_s",
        "cpu_mean_s",
        "peak_rss_max_gb",
    ),
}


@dataclass
class StatsTables:
    tables: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {name: [] for name in COLUMNS}
    )
    skipped: list[str] = field(default_factory=list)  # unreadable run folders

    def __getitem__(self, name: str) -> list[dict[str, Any]]:
        return self.tables[name]


# ---------------------------------------------------------------------------- discovery


def find_runs(paths: Iterable[str | Path]) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    """Run folders below ``paths`` and batch labels per run id (from ``batch.json``)."""
    runs: dict[Path, None] = {}
    labels: dict[str, dict[str, Any]] = {}

    def read_batch(path: Path) -> None:
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for e in index.get("entries", []):
            if e.get("run_id"):
                labels[e["run_id"]] = {
                    "experiment": index.get("experiment"),
                    "job": e.get("job"),
                    "variant": e.get("variant"),
                    "repeat": e.get("repeat"),
                }
            if e.get("run_dir") and _is_run(Path(e["run_dir"])):
                runs[Path(e["run_dir"]).resolve()] = None

    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file() and path.name == BATCH_FILE:
            read_batch(path)
        elif _is_run(path):
            runs[path.resolve()] = None
        elif (path / BATCH_FILE).is_file():
            read_batch(path / BATCH_FILE)
        elif path.is_dir():
            for manifest in sorted(path.glob(f"*/{MANIFEST_FILE}")):
                if _is_run(manifest.parent):
                    runs[manifest.parent.resolve()] = None
            for index in sorted(path.glob(f"batches/*/{BATCH_FILE}")):
                read_batch(index)
    return sorted(runs, key=lambda p: p.name), labels


def _is_run(path: Path) -> bool:
    return (path / MANIFEST_FILE).is_file() and (path / INPUT_FILE).is_file()


# ---------------------------------------------------------------------------- collection


def collect(
    paths: Iterable[str | Path],
    *,
    experiment: str | None = None,
    default_config: Mapping[str, Any] | None = None,
) -> StatsTables:
    """Read every run below ``paths`` into the tables of :data:`COLUMNS`.

    ``experiment`` keeps only runs of that batch experiment. ``default_config`` (flat or
    nested) is the reference for ``config_diff``; default: the resolved default config.
    """
    run_dirs, labels = find_runs(paths)
    reference = _flatten(default_config if default_config is not None else _default_config())
    out = StatsTables()
    for run_dir in run_dirs:
        try:
            manifest = Manifest.load(run_dir / MANIFEST_FILE)
        except (OSError, ValueError) as exc:
            out.skipped.append(f"{run_dir}: {exc}")
            continue
        label = labels.get(manifest.run_id, {})
        if experiment is not None and label.get("experiment") != experiment:
            continue
        _add_run(out, run_dir, manifest, label, reference)
    out.tables["summary"] = summarize_steps(out["steps"], out["runs"])
    return out


def _add_run(
    out: StatsTables,
    run_dir: Path,
    m: Manifest,
    label: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    device = m.device or {}
    key = {
        "run_id": m.run_id,
        "experiment": label.get("experiment"),
        "job": label.get("job"),
        "variant": label.get("variant"),
        "repeat": label.get("repeat"),
        "subject": m.input.get("subject_id"),
        "modality": m.input.get("modality"),
        "device": device.get("selected"),
    }
    config = _read_config(run_dir)
    env = m.environment
    pkgs = env.packages

    executed, cached, rss = 0, 0, []
    for name, rec in m.steps.items():
        res = rec.resources
        if rec.status == "completed":
            executed += 1
        elif rec.status == "cached":
            cached += 1
        if res is not None and res.peak_rss_bytes:
            rss.append(res.peak_rss_bytes)
        out["steps"].append(
            {
                **key,
                "step": name,
                "status": rec.status,
                "cached_from": rec.metrics.get("cached_from"),
                "wall_s": _r(res.wall_s) if res and rec.status != "cached" else None,
                "cpu_s": _r(res.cpu_s) if res and rec.status != "cached" else None,
                "peak_rss_gb": _r(res.peak_rss_bytes / GB) if res else None,
                "step_time_s": _step_time(name, rec.metrics),
            }
        )
    walls = [
        r["wall_s"]
        for r in out["steps"]
        if r["run_id"] == m.run_id and r["status"] == "completed" and r["wall_s"] is not None
    ]
    all_executed = m.status == "completed" and all(
        m.steps.get(s) is not None and m.steps[s].status == "completed" for s in PIPELINE_STEPS
    )

    seg = m.steps.get("segment")
    for tool, info in ((seg.metrics.get("tools") or {}) if seg else {}).items():
        out["tools"].append(
            {
                **key,
                "tool": tool,
                "tasks": _join(info.get("tasks")),
                "tool_device": info.get("device"),
                "time_s": info.get("time_s"),
                "n_requested": info.get("n_requested"),
                "not_provided": _join(info.get("not_provided")),
                "empty": _join(info.get("empty")),
            }
        )

    att = m.steps.get("attachments")
    method = (config.get("attachments") or {}).get("params", {}).get("nonrigid")
    for bone, reg in ((att.metrics.get("registration") or {}) if att else {}).items():
        rigid, nonrigid = reg.get("rigid") or {}, reg.get("nonrigid") or {}
        out["registration"].append(
            {
                **key,
                "bone": bone,
                "method": method,
                "status": reg.get("status"),
                "scale": reg.get("scale"),
                "rigid_mean_mm": rigid.get("mean_mm"),
                "rigid_p95_mm": rigid.get("p95_mm"),
                "rigid_trimmed_mean_mm": rigid.get("trimmed_mean_mm"),
                "nonrigid_mean_mm": nonrigid.get("mean_mm"),
                "nonrigid_p95_mm": nonrigid.get("p95_mm"),
                "nonrigid_trimmed_mean_mm": nonrigid.get("trimmed_mean_mm"),
            }
        )

    muscles = _export_muscles(run_dir, m)
    n_inflated = n_mismatch = 0
    for rec in muscles:
        check = (rec.get("mesh") or {}).get("check") or {}
        out["muscles"].append(
            {
                **key,
                "muscle": rec.get("name"),
                "status": rec.get("status"),
                "excluded_by_config": rec.get("excluded_by_config"),
                "problems": _join(rec.get("problems")),
                "n_warnings": len(rec.get("warnings") or []),
                "repaired": (rec.get("mesh") or {}).get("repaired"),
                "mesh_ok": check.get("ok"),
                "genus": check.get("genus"),
                "components": check.get("components"),
                "ori_ins_distance_mm": rec.get("ori_ins_distance_mm"),
                "mean_edge_mm": rec.get("mean_edge_mm"),
            }
        )
        for kind, area in (rec.get("areas") or {}).items():
            inflated = area.get("inflated") or {}
            mismatch = area.get("nearest_body") not in (None, area.get("body"))
            if rec.get("status") != "excluded":
                n_inflated += bool(inflated)
                n_mismatch += mismatch
            out["areas"].append(
                {
                    **key,
                    "muscle": rec.get("name"),
                    "kind": kind,
                    "muscle_status": rec.get("status"),
                    "body": area.get("body"),
                    "nearest_body": area.get("nearest_body"),
                    "body_mismatch": mismatch,
                    **{
                        k: area.get(k)
                        for k in (
                            "n_points",
                            "distinct",
                            "span_mm",
                            "shrink",
                            "crossing",
                            "patch_fraction",
                            "gap_mean_mm",
                            "gap_max_mm",
                        )
                    },
                    "inflated": bool(inflated),
                    "inflated_radius_mm": inflated.get("radius_mm"),
                    "inflated_distinct": (inflated.get("after") or {}).get("distinct"),
                    "inflated_span_mm": (inflated.get("after") or {}).get("span_mm"),
                }
            )

    runtime = config.get("runtime") or {}
    gpus = {g.get("index"): g for g in device.get("gpus") or []}
    gpu = gpus.get(device.get("gpu_index")) if device.get("selected") == "cuda" else None
    failed = next((r for r in m.steps.values() if r.status in ("failed", "interrupted")), None)
    out["runs"].append(
        {
            **key,
            "status": m.status,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "finished_at": m.finished_at.isoformat() if m.finished_at else None,
            "input_sha256": (m.input.get("sha256") or "")[:12] or None,
            "device_requested": device.get("requested"),
            "device_reason": device.get("reason"),
            "gpu_name": gpu.get("name") if gpu else None,
            "driver_version": device.get("driver_version"),
            "cpu": env.cpu,
            "cpu_logical": env.cpu_logical,
            "cpu_physical": env.cpu_physical,
            "ram_total_gb": _r(env.ram_total_bytes / GB),
            "platform": env.platform,
            "python": env.python,
            "mskpipe": _pkg(pkgs, "mskpipe"),
            "git_commit": (env.git.commit or "")[:12] or None,
            "git_dirty": env.git.dirty,
            "totalsegmentator": _pkg(pkgs, "totalsegmentator"),
            "nnunetv2": _pkg(pkgs, "nnunetv2"),
            "musclemap_commit": _pkg(pkgs, "musclemap", commit=True),
            "pystaple_commit": _pkg(pkgs, "pystaple", commit=True),
            "torch": _pkg(pkgs, "torch"),
            "vtk": _pkg(pkgs, "vtk"),
            "cache": runtime.get("cache"),
            "threads": runtime.get("threads"),
            "config_diff": config_diff(config, reference),
            "n_steps_completed": executed,
            "n_steps_cached": cached,
            "executed_wall_s": _r(sum(walls)),
            "pipeline_wall_s": _r(sum(walls)) if all_executed else None,
            "peak_rss_gb": _r(max(rss) / GB) if rss else None,
            "n_muscles_selected": len(muscles) if muscles else None,
            "n_muscles_exported": sum(r.get("status") != "excluded" for r in muscles)
            if muscles
            else None,
            "n_muscles_excluded": sum(r.get("status") == "excluded" for r in muscles)
            if muscles
            else None,
            "n_areas_inflated": n_inflated if muscles else None,
            "n_body_mismatch": n_mismatch if muscles else None,
            "error": failed.error if failed else None,
            "run_dir": str(run_dir),
        }
    )


def _export_muscles(run_dir: Path, m: Manifest) -> list[dict[str, Any]]:
    rec = m.steps.get("export_mw2")
    if rec is None or rec.status not in ("completed", "cached"):
        return []
    path = run_dir / STEP_DIRS["export_mw2"] / EXPORT_FILE
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(index.get("muscles") or [])


def _step_time(step: str, metrics: Mapping[str, Any]) -> float | None:
    """The step's own timer (``segment_time_s``, ``export_time_s``, ...), if it has one."""
    for name in (f"{step}_time_s", f"{step.split('_')[0]}_time_s"):
        if isinstance(metrics.get(name), int | float):
            return metrics[name]
    return None


def _read_config(run_dir: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _default_config() -> dict[str, Any]:
    from mskpipe.config import PipelineConfig
    from mskpipe.core.registry import PluginError, resolve_plugins

    config = PipelineConfig()
    with suppress(PluginError):
        config = resolve_plugins(config)
    return config.model_dump(mode="json")


# runtime settings that do not change results, or that have their own column
_DIFF_IGNORED = ("runtime.",)


def config_diff(config: Mapping[str, Any], reference: Mapping[str, Any]) -> str:
    """``key=value`` pairs (``;``-separated) of settings differing from ``reference``."""
    flat = _flatten(config)
    keys = sorted(set(flat) | set(reference))
    parts = []
    for k in keys:
        if k.startswith(_DIFF_IGNORED) or k == "config_version":
            continue
        if k in flat and flat.get(k) != reference.get(k):
            parts.append(f"{k}={json.dumps(flat[k], ensure_ascii=False)}")
    return ";".join(parts)


def _flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in data.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping) and v:
            flat.update(_flatten(v, f"{key}."))
        else:
            flat[key] = v
    return flat


# ---------------------------------------------------------------------------- summary


def summarize_steps(
    steps: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Mean/SD/min/max wall time per group and step; only steps executed in the run."""
    group_keys = ("experiment", "job", "variant", "subject", "modality", "device")
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in steps:
        if row["status"] == "completed" and row["wall_s"] is not None:
            groups[(*(row[k] for k in group_keys), row["step"])].append(row)
    for row in runs:
        if row["pipeline_wall_s"] is not None:
            groups[(*(row[k] for k in group_keys), "total")].append(
                {
                    "wall_s": row["pipeline_wall_s"],
                    "cpu_s": None,
                    "peak_rss_gb": row["peak_rss_gb"],
                }
            )
    order = {s: i for i, s in enumerate((*PIPELINE_STEPS, "total"))}
    out = []
    for gkey in sorted(groups, key=lambda k: (*(str(v) for v in k[:-1]), order.get(k[-1], 99))):
        rows = groups[gkey]
        walls = [r["wall_s"] for r in rows]
        cpus = [r["cpu_s"] for r in rows if r.get("cpu_s") is not None]
        rss = [r["peak_rss_gb"] for r in rows if r.get("peak_rss_gb") is not None]
        out.append(
            {
                **dict(zip(group_keys, gkey[:-1], strict=True)),
                "step": gkey[-1],
                "n": len(walls),
                "wall_mean_s": _r(statistics.fmean(walls)),
                "wall_sd_s": _r(statistics.stdev(walls)) if len(walls) > 1 else None,
                "wall_min_s": _r(min(walls)),
                "wall_max_s": _r(max(walls)),
                "cpu_mean_s": _r(statistics.fmean(cpus)) if cpus else None,
                "peak_rss_max_gb": _r(max(rss)) if rss else None,
            }
        )
    return out


# ---------------------------------------------------------------------------- output


def write_tables(tables: StatsTables, out_dir: str | Path) -> list[Path]:
    """Write every table as ``<name>.csv`` (header also for empty tables)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, columns in COLUMNS.items():
        path = out_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in tables[name]:
                writer.writerow({k: _cell(v) for k, v in row.items()})
        written.append(path)
    return written


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _r(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _join(values: Any) -> str | None:
    if not values:
        return None
    return ";".join(str(v) for v in values)


def _pkg(packages: Mapping[str, Any], name: str, *, commit: bool = False) -> str | None:
    info = packages.get(name)
    if info is None:
        return None
    value = info.commit if commit else info.version
    return value[:12] if commit and value else value
