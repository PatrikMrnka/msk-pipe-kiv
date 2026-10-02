# SPDX-License-Identifier: Apache-2.0
"""Step ``labelmap``: raw tool segmentations -> one cleaned unified label map.

Input (``01_segment/``): ``segmentation.json`` and the tool outputs it lists
(see :mod:`mskpipe.io.segmentation`).

Output (``02_labelmap/``)::

    labelmap.nii.gz     uint8 label map on the input image grid
    labels.json         structure name -> value, kind (mskpipe.io.labels; read by ``mesh``)
    labelmap.json       index: sources, parameters, per-structure status and metrics

Sources are chosen by ``segmentation.*`` and mapped by label names through
``config/unified_labels.yaml``; only the ``skeleton.side`` leg and the pelvis are built.
Cleaning follows ``labelmap.*`` (see :mod:`mskpipe.labelmap.build`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, ClassVar

from mskpipe.config import PipelineConfig
from mskpipe.core.step import Step, StepContext, StepError

INDEX_FILE = "labelmap.json"
FORMAT = "mskpipe.labelmap"
VERSION = 1


class LabelmapStep(Step):
    name: ClassVar[str] = "labelmap"
    version: ClassVar[str] = "1"
    config_sections: ClassVar[tuple[str, ...]] = ("segmentation", "labelmap")

    def fingerprint_extra(self, config: PipelineConfig) -> dict[str, Any]:
        from mskpipe.labelmap.scheme import scheme_sha256

        return {"scheme": scheme_sha256(), "side": config.skeleton.side}

    def run(self, ctx: StepContext) -> None:
        import nibabel as nib
        import numpy as np

        from mskpipe.io.labels import LABELMAP_FILE, LABELS_FILE
        from mskpipe.io.nifti import NiftiError, load_labelmap
        from mskpipe.io.segmentation import (
            SEGMENTATION_FILE,
            SegmentationIndex,
            SegmentationIndexError,
        )
        from mskpipe.labelmap.build import LabelmapError, build_labelmap
        from mskpipe.labelmap.scheme import SchemeError, load_scheme, scheme_sha256

        src = ctx.step_dir("segment")
        try:
            index = SegmentationIndex.load(src / SEGMENTATION_FILE)
            scheme = load_scheme()
            chosen = scheme.selection(ctx.config)
            plan = scheme.plan(ctx.config)
        except (SegmentationIndexError, SchemeError) as exc:
            raise StepError(str(exc)) from exc

        sources = [scheme.sources[s] for s in sorted(set(chosen.values()))]
        used = [o for o in index if any(s.accepts(o.tool, o.task) for s in sources)]
        if not used:
            raise StepError(
                f"No segmentation output for the configured sources {sorted(set(chosen.values()))}"
            )

        start = time.perf_counter()
        try:
            outputs = [(o, load_labelmap(src / o.file)) for o in used]
            result = build_labelmap(outputs, scheme, plan, ctx.config.labelmap, log=ctx.logger)
        except (NiftiError, LabelmapError) as exc:
            raise StepError(str(exc)) from exc
        elapsed = time.perf_counter() - start

        ok = [e for e in result.structures if e["status"] == "ok"]
        if not ok:
            raise StepError("No structure left in the label map (all missing or too small)")

        img = nib.Nifti1Image(result.data, result.affine)
        img.set_qform(result.affine, code=1)
        img.set_sform(result.affine, code=1)
        out = ctx.out_dir
        nib.save(img, out / LABELMAP_FILE)
        result.table.save(out / LABELS_FILE)
        index_doc = {
            "format": FORMAT,
            "version": VERSION,
            "scheme_sha256": scheme_sha256(),
            "sources": dict(chosen),
            "side": ctx.config.skeleton.side,
            "inputs": [o.file for o in used],
            "params": ctx.config.labelmap.model_dump(mode="json"),
            **result.info,
            "time_s": round(elapsed, 3),
            "structures": result.structures,
        }
        (out / INDEX_FILE).write_text(
            json.dumps(index_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name in (LABELMAP_FILE, LABELS_FILE, INDEX_FILE):
            ctx.record.add_output(out / name)

        for e in result.structures:
            if e["status"] == "ok":
                ctx.logger.info(
                    "[labelmap] %s: %d voxels (%+.2f %% vs raw), %d component(s) removed",
                    e["name"],
                    e["voxels"],
                    100 * e["volume_change"],
                    e["components_removed"],
                )
        missing = [e["name"] for e in result.structures if e["status"] == "missing"]
        if missing:
            ctx.logger.warning("[labelmap] missing: %s", ", ".join(missing))

        def total(key: str) -> int:
            return int(sum(e[key] for e in ok))

        ctx.record.metrics.update(
            {
                "sources": dict(chosen),
                "n_structures": len(result.structures),
                "n_ok": len(ok),
                "missing": missing,
                "too_small": [e["name"] for e in result.structures if e["status"] == "too_small"],
                "midline": result.info["midline"],
                "raw_overlap_voxels": total("raw_overlap"),
                "contested_voxels": total("contested"),
                "lost_to_bone_voxels": total("lost_to_bone"),
                "components_removed": total("components_removed"),
                "labelmap_time_s": round(elapsed, 3),
                "clean_time_s": round(float(np.sum([e["time_s"] for e in ok])), 3),
                "structures": {e["name"]: _summary(e) for e in ok},
            }
        )


def read_labelmap_index(labelmap_dir: Path) -> dict[str, Any]:
    """Load ``labelmap.json`` of a finished ``labelmap`` step."""
    path = Path(labelmap_dir) / INDEX_FILE
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StepError(f"Label map index not found: {path}") from None
    if index.get("format") != FORMAT or index.get("version") != VERSION:
        raise StepError(f"Unsupported label map index: {path}")
    return index


def _summary(entry: dict[str, Any]) -> dict[str, Any]:
    keys = ("kind", "voxels", "volume_ml", "volume_change", "components_removed", "time_s")
    return {k: entry[k] for k in keys}
