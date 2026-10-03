# SPDX-License-Identifier: Apache-2.0
"""Run the implemented pipeline steps on one volume (developer tool until `mskpipe run`).

    # LHDL CT from scratch to the skeletal model, CPU, no cache (timings for the paper)
    pixi run -e cpu python tools/run_steps.py D:/data/lhdl/ct.nii.gz --modality ct `
        --until skeleton --set runtime.cache=false

    # same on the GPU
    pixi run -e gpu python tools/run_steps.py D:/data/lhdl/ct.nii.gz --modality ct `
        --until skeleton --set runtime.device=gpu --set runtime.cache=false

    # continue an existing run from a step (e.g. after changing code of labelmap)
    pixi run -e cpu python tools/run_steps.py --resume runs/<run> --from labelmap

Prints the run folder and a per-step summary (status, wall time, peak RAM/GPU memory).
Exit code 0 = all requested steps completed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

from mskpipe.config import ConfigError, InputSpec, load_config
from mskpipe.core.device import DeviceError
from mskpipe.core.registry import resolve_plugins
from mskpipe.core.runner import PipelineCancelled, PipelineError, RunResult, run_pipeline
from mskpipe.steps.labelmap import LabelmapStep
from mskpipe.steps.mesh import MeshStep
from mskpipe.steps.segment import SegmentStep
from mskpipe.steps.skeleton import SkeletonStep

STEPS = (SegmentStep(), LabelmapStep(), MeshStep(), SkeletonStep())
NAMES = [s.name for s in STEPS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("image", nargs="?", type=Path, help="input volume (.nii/.nii.gz)")
    ap.add_argument("--modality", choices=["ct", "mri"], help="modality of the input")
    ap.add_argument("--subject", help="subject id (default: file name)")
    ap.add_argument("--config", type=Path, help="YAML config file")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", type=Path, help="existing run folder to continue")
    ap.add_argument("--from", dest="from_step", choices=NAMES, help="re-run from this step")
    ap.add_argument("--until", choices=NAMES, default=NAMES[-1], help="last step to run")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print tool output")
    args = ap.parse_args(argv)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    console.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("mskpipe").addHandler(console)

    cancel = threading.Event()
    try:
        if args.resume:
            result = run_pipeline(
                STEPS,
                resume=args.resume,
                from_step=args.from_step,
                until_step=args.until,
                cancel=cancel,
            )
        else:
            if args.image is None or args.modality is None:
                ap.error("image and --modality are required unless --resume is given")
            spec = InputSpec.model_validate(
                {"image": args.image, "modality": args.modality, "subject_id": args.subject}
            )
            config = resolve_plugins(load_config(args.config, args.overrides))
            result = run_pipeline(
                STEPS, spec, config, from_step=args.from_step, until_step=args.until, cancel=cancel
            )
    except (ConfigError, DeviceError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except PipelineCancelled as exc:
        print(f"Cancelled: {exc.ws.root}", file=sys.stderr)
        return 130
    except PipelineError as exc:
        print(f"Failed in step '{exc.step}': {exc}\nRun folder: {exc.ws.root}", file=sys.stderr)
        return 1
    print_summary(result)
    return 0


def print_summary(result: RunResult) -> None:
    m = result.manifest
    print(f"\nRun: {result.ws.root}")
    print(f"Device: {(m.device or {}).get('selected', '?')} - {(m.device or {}).get('reason', '')}")
    print(f"{'step':<12}{'status':<11}{'wall s':>9}{'CPU s':>9}{'peak RAM GB':>13}{'GPU GB':>9}")
    for name, rec in m.steps.items():
        res = rec.resources
        wall = f"{res.wall_s:.1f}" if res else "-"
        cpu = f"{res.cpu_s:.1f}" if res else "-"
        ram = f"{res.peak_rss_bytes / 2**30:.2f}" if res and res.peak_rss_bytes else "-"
        gpu = res.gpu_mem_peak_bytes if res else None
        gpu_s = f"{gpu / 2**30:.2f}" if gpu else "-"
        print(f"{name:<12}{rec.status:<11}{wall:>9}{cpu:>9}{ram:>13}{gpu_s:>9}")


if __name__ == "__main__":
    sys.exit(main())
