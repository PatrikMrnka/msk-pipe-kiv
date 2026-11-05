#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Fibre lengths over the motion: pipeline vs a reference Muscle Wrapping run (paper_results E16).

The fibre lengths of each muscle (``<muscle>_Lengths.sto`` of Muscle Wrapping 2.x) are
compared as a function of hip flexion with the same muscle of a reference run (on LHDL:
the original Muscle Wrapping model with the manual muscle surfaces and dissection
attachments). Every fibre is resampled on a common flexion grid (default 0-90 deg, 2 deg);
time is not used, because the two motions differ.

Fibres of two runs do not correspond one to one (each run decomposes the muscle anew),
so individual fibres are compared only as a distribution.

Metrics per muscle (mm), from the mean fibre length unless stated:
  L_ref, L_pipe   mean over the grid
  offset          mean(pipe - ref)
  rmse, rmse_c    RMSE; RMSE after removing the offset (shape of the curve)
  r               Pearson correlation
  excursion_*     L(last) - L(first) of the grid; excursion_err = pipe - ref
  iqr_*           interquartile range of the fibre lengths, mean over the grid
  iqr_ratio       iqr_pipe / iqr_ref (spread of the fibres)
  w1c             Wasserstein distance of the fibre-length distributions after removing
                  their means, mean over the grid (shape of the distribution)
With ``--ref2`` (a second run of the reference), the same metrics of ref2 vs ref are
written as ``noise_*`` (run-to-run variability of Muscle Wrapping).

Outputs in ``--out``: ``fibre_lengths.csv``, ``fibre_lengths.json`` (inputs with SHA-256,
grid, curves, percentiles), figures (PNG + PDF, when matplotlib is available):
  fibre_lengths            all muscles: mean and 5-95 % band of the fibres vs flexion
  fibre_lengths_violins    all muscles at one pose (``--pose``), lengths / mean of the
                           reference, split violins reference | pipeline
  fibre_lengths_<muscle>   detail (``--detail``): curve with band + violins at ``--poses``

Muscles are matched by name without side and separators (``gluteus_medius_r`` =
``GluteusMedius``); ``--map pipe=ref`` overrides a pair.

Usage (repository root)::

    pixi run -e dev-cpu python tools/fibre_lengths.py runs/<run> `
        --ref ../reference/mw2_baseline/Experiments_luca_all --out ../paper/stats/v0.3.0/e16

``RUN`` is a run folder (or its ``06_mw2_input``) after Muscle Wrapping was run there;
``--ref-motion`` defaults to ``hip_flexion_kinematics.mot`` next to ``--ref`` or above it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COORDINATE = "hip_flexion_r"
REF_MOTION = "hip_flexion_kinematics.mot"
DETAIL = ("gluteus_medius_r", "adductor_longus_r", "semitendinosus_r")
# name variants seen in Muscle Wrapping setups -> pipeline key
ALIASES = {
    "tensorfascialatae": "tensorfasciaelatae",
    "bicepsfemorislonghead": "bicepsfemoris",
    "bicepsfemorislh": "bicepsfemoris",
    "bicepsfemoriscaputlongum": "bicepsfemoris",
    "semitendinous": "semitendinosus",
    "semimembranous": "semimembranosus",
    "pyriformis": "piriformis",
}
REF_COLOR, PIPE_COLOR, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#3b3b38", "#6f6e69", "#e4e3dd"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- reading


@dataclass
class Table:
    path: Path
    header: dict[str, str]
    columns: list[str]
    data: np.ndarray  # rows x columns


def read_storage(path: Path) -> Table:
    """OpenSim .sto / .mot (header up to ``endheader``, then a tab-separated table)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: dict[str, str] = {}
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            start = i + 1
            break
        if "=" in line:
            k, value = line.split("=", 1)
            header[k.strip()] = value.strip()
    if start is None:
        raise SystemExit(f"{path}: no 'endheader'")
    columns = lines[start].split()
    rows = [ln.split() for ln in lines[start + 1 :] if ln.strip()]
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise SystemExit(f"{path}: {data.shape} values for {len(columns)} columns")
    return Table(path, header, columns, data)


def motion_angle(motion: Table, coordinate: str) -> tuple[np.ndarray, np.ndarray]:
    """Time and coordinate (deg) of a motion file."""
    if coordinate not in motion.columns:
        raise SystemExit(f"{motion.path}: no column {coordinate} ({', '.join(motion.columns)})")
    t = motion.data[:, motion.columns.index("time")]
    q = motion.data[:, motion.columns.index(coordinate)]
    if motion.header.get("inDegrees", "yes").lower() == "no":
        q = np.degrees(q)
    return t, q


def fibre_lengths_mm(table: Table) -> tuple[np.ndarray, np.ndarray]:
    """Time and fibre lengths (frames x fibres, mm) of a ``*_Lengths.sto``."""
    i = table.columns.index("time")
    fibres = np.delete(table.data, i, axis=1)
    if fibres.shape[1] == 0:
        raise SystemExit(f"{table.path}: no fibre columns")
    scale = 1000.0 if np.nanmedian(fibres) < 5.0 else 1.0  # Muscle Wrapping writes metres
    return table.data[:, i], fibres * scale


def angle_at(t: np.ndarray, motion: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mt, mq = motion
    if len(t) == len(mt) and np.allclose(t, mt, atol=1e-6):
        return mq.copy()
    return np.interp(t, mt, mq)


def on_grid(angle: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Resample frames x fibres on the angle grid (grid x fibres)."""
    order = np.argsort(angle, kind="stable")
    a, v = angle[order], values[order]
    keep = np.concatenate([[True], np.diff(a) > 0])  # repeated angles (rest pose): first
    a, v = a[keep], v[keep]
    return np.column_stack([np.interp(grid, a, v[:, j]) for j in range(v.shape[1])])


# --------------------------------------------------------------------------- matching


def key(name: str) -> str:
    stem = re.sub(r"_Lengths$", "", name, flags=re.I)
    stem = re.sub(r"[_\- ]?(r|l|right|left)$", "", stem, flags=re.I)
    k = re.sub(r"[^a-z]", "", stem.lower())
    return ALIASES.get(k, k)


def length_files(folder: Path) -> dict[str, Path]:
    return {p.name[: -len("_Lengths.sto")]: p for p in sorted(folder.glob("*_Lengths.sto"))}


def find_ref_motion(ref: Path) -> Path | None:
    for folder in (ref, *ref.parents[:2]):
        hit = folder / REF_MOTION
        if hit.is_file():
            return hit
    return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(muscle: str) -> str:
    return re.sub(r"_[rl]$", "", muscle).replace("_", " ")


# --------------------------------------------------------------------------- metrics


def wasserstein_1d(a: np.ndarray, b: np.ndarray, n: int = 1000) -> float:
    """W1 of two samples = mean |Q_a(p) - Q_b(p)| over p (quantile functions)."""
    p = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, p) - np.quantile(b, p))))


def iqr(x: np.ndarray, axis: int = -1) -> np.ndarray:
    q1, q3 = np.percentile(x, [25, 75], axis=axis)
    return q3 - q1


def compare(ref: np.ndarray, pipe: np.ndarray) -> dict[str, float]:
    """Metrics of two fibre sets on the same grid (grid x fibres each)."""
    mr, mp = ref.mean(axis=1), pipe.mean(axis=1)
    d = mp - mr
    r = float(np.corrcoef(mr, mp)[0, 1]) if np.std(mr) > 0 and np.std(mp) > 0 else float("nan")
    iqr_r, iqr_p = float(iqr(ref, 1).mean()), float(iqr(pipe, 1).mean())
    w1c = np.mean([wasserstein_1d(ref[g] - mr[g], pipe[g] - mp[g]) for g in range(len(mr))])
    return {
        "L_ref_mm": float(mr.mean()),
        "L_pipe_mm": float(mp.mean()),
        "offset_mm": float(d.mean()),
        "rmse_mm": float(np.sqrt(np.mean(d**2))),
        "rmse_c_mm": float(np.sqrt(np.mean((d - d.mean()) ** 2))),
        "r": r,
        "excursion_ref_mm": float(mr[-1] - mr[0]),
        "excursion_pipe_mm": float(mp[-1] - mp[0]),
        "excursion_err_mm": float((mp[-1] - mp[0]) - (mr[-1] - mr[0])),
        "iqr_ref_mm": iqr_r,
        "iqr_pipe_mm": iqr_p,
        "iqr_ratio": iqr_p / iqr_r if iqr_r > 0 else float("nan"),
        "w1c_mm": float(w1c),
    }


# --------------------------------------------------------------------------- figures


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available: no figures")
        return None
    return plt


def _style(ax) -> None:
    ax.tick_params(labelsize=7, colors=INK, length=2)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _curve(ax, grid: np.ndarray, fibres: np.ndarray, color: str, ls: str, name: str) -> None:
    lo, hi = np.percentile(fibres, [5, 95], axis=1)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.16, lw=0)
    ax.plot(grid, fibres.mean(axis=1), color=color, lw=1.6, ls=ls, label=name)


def _half_violin(ax, data: np.ndarray, pos: float, color: str, side: int, vert: bool) -> None:
    """Half of a violin at ``pos``: side -1 = left/below, +1 = right/above."""
    parts = ax.violinplot([data], positions=[pos], vert=vert, widths=0.85, showextrema=False)
    for body in parts["bodies"]:
        verts = body.get_paths()[0].vertices
        c = 0 if vert else 1  # the axis across the violin
        if side < 0:
            verts[:, c] = np.minimum(verts[:, c], pos)
        else:
            verts[:, c] = np.maximum(verts[:, c], pos)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
        body.set_linewidth(0.8)
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    span = (pos, pos + side * 0.36)
    for q, lw in ((q1, 0.6), (q2, 1.2), (q3, 0.6)):
        if vert:
            ax.plot(span, (q, q), color=INK, lw=lw, solid_capstyle="butt")
        else:
            ax.plot((q, q), span, color=INK, lw=lw, solid_capstyle="butt")


def _legend(fig, plt) -> None:
    handles = [
        plt.Line2D([], [], color=REF_COLOR, lw=1.6, ls="--", label="reference"),
        plt.Line2D([], [], color=PIPE_COLOR, lw=1.6, label="pipeline"),
        plt.matplotlib.patches.Patch(
            color=REF_COLOR, alpha=0.16, lw=0, label="reference, 5-95 % of fibres"
        ),
        plt.matplotlib.patches.Patch(
            color=PIPE_COLOR, alpha=0.16, lw=0, label="pipeline, 5-95 % of fibres"
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8)


def _save(fig, plt, out: Path, stem: str) -> list[Path]:
    paths = [out / f"{stem}.png", out / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def plot_all(rows, fibres, grid, out: Path) -> list[Path]:
    plt = _pyplot()
    if plt is None or not rows:
        return []
    n = len(rows)
    cols = min(6, n)
    nrows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(2.3 * cols, 1.9 * nrows + 0.3), squeeze=False)
    for ax, row in zip(axes.flat, rows, strict=False):
        ref, pipe = fibres[row["muscle"]]
        _curve(ax, grid, ref, REF_COLOR, "--", "reference")
        _curve(ax, grid, pipe, PIPE_COLOR, "-", "pipeline")
        title = f"{label(row['muscle'])} ({row['offset_mm']:+.0f} mm)"
        ax.set_title(title, fontsize=8, color=INK, pad=3)
        _style(ax)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("hip flexion (deg)", fontsize=7, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("fibre length (mm)", fontsize=7, color=INK)
    _legend(fig, plt)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, plt, out, "fibre_lengths")


def plot_violins(rows, fibres, grid, pose: float, out: Path) -> list[Path]:
    plt = _pyplot()
    if plt is None or not rows:
        return []
    g = int(np.argmin(np.abs(grid - pose)))
    fig, ax = plt.subplots(figsize=(6.0, 0.36 * len(rows) + 1.0))
    labels = []
    for i, row in enumerate(rows):
        ref, pipe = fibres[row["muscle"]]
        norm = ref[g].mean()
        y = len(rows) - 1 - i
        _half_violin(ax, ref[g] / norm, y, REF_COLOR, +1, vert=False)
        _half_violin(ax, pipe[g] / norm, y, PIPE_COLOR, -1, vert=False)
        labels.append(label(row["muscle"]))
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1], fontsize=7, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(f"fibre length / mean length of the reference (hip flexion {grid[g]:g} deg)",
                  fontsize=7, color=INK)  # fmt: skip
    _style(ax)
    ax.grid(axis="y", visible=False)
    handles = [
        plt.matplotlib.patches.Patch(color=REF_COLOR, alpha=0.55, label="reference (upper half)"),
        plt.matplotlib.patches.Patch(color=PIPE_COLOR, alpha=0.55, label="pipeline (lower half)"),
        plt.Line2D([], [], color=INK, lw=1.0, label="median (thick), quartiles"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, plt, out, "fibre_lengths_violins")


def plot_detail(row, ref, pipe, grid, poses, out: Path) -> list[Path]:
    plt = _pyplot()
    if plt is None:
        return []
    fig = plt.figure(figsize=(7.0, 5.2))
    top = fig.add_gridspec(2, len(poses), height_ratios=(1.25, 1), hspace=0.45, wspace=0.35)
    ax = fig.add_subplot(top[0, :])
    _curve(ax, grid, ref, REF_COLOR, "--", "reference")
    _curve(ax, grid, pipe, PIPE_COLOR, "-", "pipeline")
    for p in poses:
        ax.axvline(p, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("hip flexion (deg)", fontsize=7, color=INK)
    ax.set_ylabel("fibre length (mm)", fontsize=7, color=INK)
    title = (
        f"{label(row['muscle'])}: offset {row['offset_mm']:+.1f} mm, "
        f"RMSEc {row['rmse_c_mm']:.1f} mm, r {row['r']:.3f}, "
        f"excursion {row['excursion_pipe_mm']:.1f} vs {row['excursion_ref_mm']:.1f} mm"
    )
    ax.set_title(title, fontsize=8, color=INK, pad=4)
    _style(ax)
    lo = min(ref.min(), pipe.min())
    hi = max(ref.max(), pipe.max())
    pad = 0.05 * (hi - lo)
    for i, p in enumerate(poses):
        g = int(np.argmin(np.abs(grid - p)))
        axv = fig.add_subplot(top[1, i])
        _half_violin(axv, ref[g], 0.0, REF_COLOR, +1, vert=True)
        _half_violin(axv, pipe[g], 0.0, PIPE_COLOR, -1, vert=True)
        axv.set_xlim(-0.6, 0.6)
        axv.set_ylim(lo - pad, hi + pad)
        axv.set_xticks([])
        axv.set_title(f"{grid[g]:g} deg", fontsize=8, color=INK, pad=3)
        _style(axv)
        axv.grid(axis="x", visible=False)
        if i == 0:
            axv.set_ylabel("fibre length (mm)", fontsize=7, color=INK)
    handles = [
        plt.Line2D([], [], color=REF_COLOR, lw=1.6, ls="--", label="reference (violin right)"),
        plt.Line2D([], [], color=PIPE_COLOR, lw=1.6, label="pipeline (violin left)"),
        plt.Line2D([], [], color=INK, lw=1.0, label="median (thick), quartiles"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.86, bottom=0.06, left=0.09, right=0.98)
    return _save(fig, plt, out, f"fibre_lengths_{row['muscle']}")


# --------------------------------------------------------------------------- main


def load_set(folder: Path, motion: tuple[np.ndarray, np.ndarray], grid: np.ndarray, name: str):
    t, f = fibre_lengths_mm(read_storage(folder))
    a = angle_at(t, motion)
    if grid[0] < a.min() - 1e-6 or grid[-1] > a.max() + 1e-6:
        raise SystemExit(
            f"{name}: grid {grid[0]:g}-{grid[-1]:g} deg outside the motion "
            f"({a.min():.1f}-{a.max():.1f} deg): change --grid"
        )
    return on_grid(a, f, grid)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, help="run folder or its 06_mw2_input (after Muscle Wrapping)")
    ap.add_argument(
        "--ref", type=Path, required=True, help="folder with the reference *_Lengths.sto"
    )
    ap.add_argument("--ref-motion", type=Path, help=f"motion of the reference run ({REF_MOTION})")
    ap.add_argument("--ref2", type=Path, help="second run of the reference (Muscle Wrapping noise)")
    ap.add_argument("--ref2-motion", type=Path, help="motion of --ref2 (default: --ref-motion)")
    ap.add_argument("--motion", type=Path, help="motion of the pipeline run (from its setup)")
    ap.add_argument("--coordinate", default=COORDINATE)
    ap.add_argument(
        "--grid", nargs=3, type=float, default=(0.0, 90.0, 2.0), metavar=("FROM", "TO", "STEP")
    )
    ap.add_argument("--pose", type=float, default=0.0, help="pose of the violin figure (deg)")
    ap.add_argument("--poses", nargs="+", type=float, default=(0.0, 45.0, 90.0))
    ap.add_argument(
        "--detail", nargs="*", default=list(DETAIL), help="muscles with a detail figure"
    )
    ap.add_argument("--map", action="append", default=[], metavar="PIPE=REF", help="pair names")
    ap.add_argument("--out", type=Path, default=Path("fibre_lengths"))
    args = ap.parse_args(argv)

    mw2 = args.run / "06_mw2_input" if (args.run / "06_mw2_input").is_dir() else args.run
    setup = mw2 / "setup_MuscleGeneratorTool.xml"
    prefix, motion_name = "analysis/", "hip_flexion_r.mot"
    if setup.is_file():
        text = setup.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"<output_muscle_analysis_file_prefix>([^<]*)<", text)
        prefix = m.group(1).strip() if m else prefix
        m = re.search(r"<motion_file>([^<]*)<", text)
        motion_name = m.group(1).strip() if m else motion_name
    # Muscle Wrapping writes <prefix><muscle>_Lengths.sto; prefix "analysis/" = a folder
    analysis = mw2 / prefix if prefix.endswith(("/", "\\")) else (mw2 / prefix).parent
    motion_path = args.motion or mw2 / motion_name
    ref_motion_path = args.ref_motion or find_ref_motion(args.ref)
    if ref_motion_path is None:
        raise SystemExit(f"No {REF_MOTION} in {args.ref} or above it: give --ref-motion")

    pipe_files = length_files(analysis)
    ref_files = length_files(args.ref)
    ref2_files = length_files(args.ref2) if args.ref2 else {}
    if not pipe_files:
        raise SystemExit(f"No *_Lengths.sto in {analysis} (was Muscle Wrapping run there?)")
    if not ref_files:
        raise SystemExit(f"No *_Lengths.sto in {args.ref}")
    if args.ref2 and not ref2_files:
        raise SystemExit(f"No *_Lengths.sto in {args.ref2}")
    manual = dict(item.split("=", 1) for item in args.map)
    ref_by_key = {key(name): name for name in ref_files}

    exported: list[str] = []
    export_json = mw2 / "export.json"
    if export_json.is_file():
        records = json.loads(export_json.read_text(encoding="utf-8")).get("muscles", [])
        exported = [r["name"] for r in records if r.get("status") != "excluded"]
    missing_runs = sorted(set(exported) - set(pipe_files))
    if missing_runs:
        log(f"WARNING: exported but no lengths (Muscle Wrapping incomplete?): {missing_runs}")

    pipe_motion = motion_angle(read_storage(motion_path), args.coordinate)
    ref_motion = motion_angle(read_storage(ref_motion_path), args.coordinate)
    ref2_motion_path = args.ref2_motion or ref_motion_path
    ref2_motion = (
        motion_angle(read_storage(ref2_motion_path), args.coordinate) if args.ref2 else None
    )
    lo, hi, step = args.grid
    grid = np.arange(lo, hi + step / 2, step)

    rows: list[dict] = []
    fibres: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    unmatched: list[str] = []
    for name, path in pipe_files.items():
        ref_name = manual.get(name) or ref_by_key.get(key(name))
        if ref_name is None or ref_name not in ref_files:
            unmatched.append(name)
            continue
        pipe = load_set(path, pipe_motion, grid, name)
        ref = load_set(ref_files[ref_name], ref_motion, grid, ref_name)
        row = {"muscle": name, "reference": ref_name, "fibres_pipe": pipe.shape[1],
               "fibres_ref": ref.shape[1]}  # fmt: skip
        row |= {k: round(v, 4) for k, v in compare(ref, pipe).items()}
        if args.ref2:
            if ref_name in ref2_files:
                ref2 = load_set(ref2_files[ref_name], ref2_motion, grid, f"ref2 {ref_name}")
                noise = compare(ref, ref2)
                for k in ("offset_mm", "rmse_c_mm", "excursion_err_mm", "iqr_ratio", "w1c_mm"):
                    row[f"noise_{k}"] = round(noise[k], 4)
            else:
                log(f"{ref_name}: not in --ref2")
        rows.append(row)
        fibres[name] = (ref, pipe)
    rows.sort(key=lambda r: abs(r["offset_mm"]))
    not_in_pipe = sorted(set(ref_files) - {r["reference"] for r in rows})

    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) if rows else ["muscle"]
    with (args.out / "fibre_lengths.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    manifest = mw2.parent / "manifest.json"
    version = None
    if manifest.is_file():
        env = json.loads(manifest.read_text(encoding="utf-8")).get("environment", {})
        version = (env.get("packages", {}).get("mskpipe") or {}).get("version")
    inputs = {
        "pipeline_motion": str(motion_path),
        "reference_motion": str(ref_motion_path),
        "pipeline_lengths": {k: str(v) for k, v in pipe_files.items()},
        "reference_lengths": {k: str(v) for k, v in ref_files.items()},
    }
    hashed = [motion_path, ref_motion_path, *pipe_files.values(), *ref_files.values()]
    if args.ref2:
        inputs["reference2_motion"] = str(ref2_motion_path)
        inputs["reference2_lengths"] = {k: str(v) for k, v in ref2_files.items()}
        hashed += [ref2_motion_path, *ref2_files.values()]
    percentiles = (5, 25, 50, 75, 95)
    curves = {
        name: {
            src: {
                "mean_mm": arr.mean(axis=1).round(3).tolist(),
                **{
                    f"p{p}_mm": q.round(3).tolist()
                    for p, q in zip(
                        percentiles, np.percentile(arr, percentiles, axis=1), strict=True
                    )
                },
            }
            for src, arr in (("reference", ref), ("pipeline", pipe))
        }
        for name, (ref, pipe) in fibres.items()
    }
    summary = {
        "n_muscles": len(rows),
        "r_min": min((r["r"] for r in rows), default=None),
        "rmse_c_le_4mm": sum(r["rmse_c_mm"] <= 4.0 for r in rows),
        "unmatched_pipeline": unmatched,
        "reference_not_in_pipeline": not_in_pipe,
        "exported_without_lengths": missing_runs,
    }
    report = {
        "format": "mskpipe.fibre_lengths",
        "version": 2,
        "run": str(mw2.parent if mw2.name == "06_mw2_input" else mw2),
        "mskpipe": version,
        "reference": str(args.ref),
        "reference2": str(args.ref2) if args.ref2 else None,
        "coordinate": args.coordinate,
        "grid_deg": grid.tolist(),
        "summary": summary,
        "muscles": rows,
        "curves": curves,
        "inputs": inputs,
        "sha256": {str(p): sha256(Path(p)) for p in hashed},
    }
    (args.out / "fibre_lengths.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    figures = plot_all(rows, fibres, grid, args.out)
    figures += plot_violins(rows, fibres, grid, args.pose, args.out)
    by_name = {r["muscle"]: r for r in rows}
    for name in args.detail:
        if name in by_name:
            ref, pipe = fibres[name]
            figures += plot_detail(by_name[name], ref, pipe, grid, args.poses, args.out)
        else:
            log(f"--detail {name}: not compared")

    head = f"{'muscle':<26}{'offset':>8}{'RMSEc':>7}{'r':>7}{'exc ref':>9}{'exc pipe':>9}"
    head += f"{'IQR ref':>9}{'IQR pipe':>9}{'W1c':>6}"
    print(head + (f"{'noise W1c':>10}" if args.ref2 else ""))
    for r in rows:
        line = (
            f"{r['muscle']:<26}{r['offset_mm']:>+8.1f}{r['rmse_c_mm']:>7.1f}{r['r']:>7.3f}"
            f"{r['excursion_ref_mm']:>9.1f}{r['excursion_pipe_mm']:>9.1f}"
            f"{r['iqr_ref_mm']:>9.1f}{r['iqr_pipe_mm']:>9.1f}{r['w1c_mm']:>6.1f}"
        )
        if "noise_w1c_mm" in r:
            line += f"{r['noise_w1c_mm']:>10.1f}"
        print(line)
    if rows:
        print(
            f"\n{len(rows)} muscles; r >= {summary['r_min']:.3f}; "
            f"RMSEc <= 4 mm: {summary['rmse_c_le_4mm']}"
        )
    else:
        print("no muscles matched")
    if unmatched:
        print(f"not matched (use --map PIPE=REF): {unmatched}")
    if not_in_pipe:
        print(f"reference only: {not_in_pipe}")
    for p in [args.out / "fibre_lengths.csv", args.out / "fibre_lengths.json", *figures]:
        print(f"written {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
