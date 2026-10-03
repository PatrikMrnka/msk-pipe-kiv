# SPDX-License-Identifier: Apache-2.0
"""MuscleMap whole-body model (CLI ``mm_segment``), contrast-agnostic: CT and MRI.

Label names are read from the model's label JSON (downloaded by MuscleMap with the
weights) and normalised to ``<anatomy>_<l|r>`` (``gluteus maximus`` + ``right`` ->
``gluteus_maximus_r``); label IDs differ between model versions, so they are never
hard-coded. The model version is pinned (``MODEL_VERSION``) for reproducibility.

Two MuscleMap quirks are handled here:

* ``mm_segment`` logs an exception and still exits with code 0, so the output file is
  checked explicitly;
* it saves the label image with the input's header, so an 8-bit input would scale the
  label IDs (up to ~8200) lossily. Such inputs are passed as a float32 copy instead.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mskpipe.plugins.base import (
    PluginParams,
    SegmentationOutput,
    SegmentationRequest,
    SegmenterPlugin,
)

if TYPE_CHECKING:
    import numpy as np

    from mskpipe.config import PipelineConfig
    from mskpipe.core.step import StepContext

MODEL_VERSION = "1.4"  # whole-body model on Zenodo; label IDs below were checked for it
REGION = "wholebody"
PACKAGE = "scripts"  # MuscleMap is installed as the top-level package "scripts"
MODEL_JSON = "contrast_agnostic_wholebody_model.json"
OUTPUT_FILE = "musclemap_wholebody.nii.gz"
WORK_DIR = "_musclemap"
_SLAB = 64  # slices converted at once (bounds memory for large CT volumes)


class MuscleMapError(RuntimeError):
    """MuscleMap output or model files cannot be used."""


def model_version(config: PipelineConfig) -> str:
    return config.segmentation.musclemap.model_version or MODEL_VERSION


def label_name(anatomy: str, side: str) -> str:
    """``gluteus maximus`` + ``right`` -> ``gluteus_maximus_r``; ``no side`` -> no suffix."""
    base = re.sub(r"[^a-z0-9]+", "_", anatomy.lower()).strip("_")
    side = side.strip().lower()
    if side in ("left", "l"):
        return f"{base}_l"
    if side in ("right", "r"):
        return f"{base}_r"
    return base


def parse_model_labels(config: Mapping[str, Any]) -> dict[str, int]:
    """Normalised label name -> value from a MuscleMap model JSON."""
    labels: dict[str, int] = {}
    for entry in config.get("labels", []):
        name = label_name(str(entry.get("anatomy", "")), str(entry.get("side", "")))
        value = int(entry["value"])
        if not name or value <= 0:
            continue
        if name in labels and labels[name] != value:
            raise MuscleMapError(f"Label '{name}' has two values ({labels[name]}, {value})")
        labels[name] = value
    if not labels:
        raise MuscleMapError("The model JSON lists no labels")
    return labels


def package_dir() -> Path:
    """Folder of the installed MuscleMap package (without importing it: it imports torch)."""
    spec = importlib.util.find_spec(PACKAGE)
    locations = list(spec.submodule_search_locations or []) if spec else []
    for loc in locations:
        if (Path(loc) / "mm_segment.py").is_file():
            return Path(loc)
    raise MuscleMapError(f"MuscleMap (package '{PACKAGE}') is not installed")


def model_json_path(version: str, root: Path | None = None) -> Path:
    """Label JSON cached by MuscleMap next to the weights of ``version``."""
    root = root or package_dir()
    return root / "models" / REGION / f"v{version}" / MODEL_JSON


def read_model_labels(version: str, root: Path | None = None) -> dict[str, int]:
    path = model_json_path(version, root)
    try:
        return parse_model_labels(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise MuscleMapError(
            f"MuscleMap model {version} label file not found: {path} "
            "(the model is downloaded on the first run; check the network connection)"
        ) from None
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise MuscleMapError(f"Invalid MuscleMap model file {path}: {exc}") from exc


def needs_float_copy(dtype: np.dtype) -> bool:
    """Whether MuscleMap would store label IDs lossily in an image of this dtype."""
    import numpy as np

    dtype = np.dtype(dtype)
    return np.issubdtype(dtype, np.integer) and np.iinfo(dtype).max < 32767


def prepare_input(image: Path, work: Path) -> Path:
    """The image to give MuscleMap: the input itself, or a float32 copy of 8-bit inputs."""
    import nibabel as nib
    import numpy as np

    img = nib.load(image)
    if not needs_float_copy(img.get_data_dtype()):
        return image
    data = img.get_fdata(dtype=np.float32)
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    copy = nib.Nifti1Image(data, img.affine, header)
    out = work / "input_float32.nii.gz"
    nib.save(copy, out)
    return out


def normalize_output(src: Path, dst: Path, valid: set[int]) -> None:
    """Save MuscleMap's label image as uint16; fail on values that are not model labels."""
    import nibabel as nib
    import numpy as np

    img = nib.load(src)
    shape = img.shape[:3]
    if len(img.shape) not in (3, 4) or (len(img.shape) == 4 and img.shape[3] != 1):
        raise MuscleMapError(f"{src.name}: expected a 3D label image, got shape {img.shape}")
    out = np.empty(shape, dtype=np.uint16)
    proxy = img.dataobj
    for z0 in range(0, shape[2], _SLAB):
        z1 = min(z0 + _SLAB, shape[2])
        block = np.asanyarray(proxy[:, :, z0:z1] if len(img.shape) == 3 else proxy[:, :, z0:z1, 0])
        if not np.issubdtype(block.dtype, np.integer):
            rounded = np.rint(block)
            if not np.array_equal(rounded, block):
                raise MuscleMapError(f"{src.name}: non-integer label values (header scaling?)")
            block = rounded
        if block.size and (block.min() < 0 or block.max() > np.iinfo(np.uint16).max):
            raise MuscleMapError(f"{src.name}: label values outside 0..65535")
        out[:, :, z0:z1] = block
    found = {int(v) for v in np.flatnonzero(np.bincount(out.ravel()))} - {0}
    unknown = sorted(found - valid)
    if unknown:
        raise MuscleMapError(
            f"{src.name}: values {unknown[:10]} are not labels of the model "
            "(label IDs were probably scaled by the image header)"
        )
    result = nib.Nifti1Image(out, img.affine)
    result.set_qform(img.affine, code=1)
    result.set_sform(img.affine, code=1)
    nib.save(result, dst)


def build_command(
    image: Path,
    work: Path,
    *,
    version: str,
    device: str,
    overlap: float,
    chunk_size: int | str,
    extra_args: list[str],
    executable: str = "mm_segment",
) -> list[str]:
    argv = [executable, "-i", str(image), "-o", str(work), "-r", REGION]
    argv += ["--model_version", version, "-g", "Y" if device == "cuda" else "N"]
    argv += ["-s", f"{overlap:g}", "-c", str(chunk_size)]
    return argv + list(extra_args)


def output_path(image: Path, work: Path) -> Path:
    """Where ``mm_segment`` writes the segmentation of ``image``."""
    base = re.sub(r"\.nii(\.gz)?$", "", image.name)
    return work / f"{base}_dseg.nii.gz"


class MuscleMap(SegmenterPlugin):
    name = "musclemap"
    description = "MuscleMap: hip and thigh muscles, pelvis, femur, tibia, fibula (CT and MRI)."
    requires_executables = ("mm_segment",)
    supports_gpu = True

    @classmethod
    def cache_identity(cls, config: PipelineConfig) -> Mapping[str, Any]:
        return {"model_version": model_version(config), "region": REGION}

    def segment(
        self,
        ctx: StepContext,
        image: Path,
        out_dir: Path,
        params: PluginParams,
        request: SegmentationRequest,
    ) -> list[SegmentationOutput]:
        from mskpipe.core.step import StepError

        cfg = ctx.config.segmentation.musclemap
        version = model_version(ctx.config)
        device = ctx.device.device_for(self.supports_gpu)
        work = out_dir / WORK_DIR
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        try:
            src = prepare_input(image, work)
            if src != image:
                ctx.logger.info("[segment] MuscleMap gets a float32 copy of the 8-bit input")
            argv = build_command(
                src,
                work,
                version=version,
                device=device,
                overlap=cfg.overlap,
                chunk_size=cfg.chunk_size,
                extra_args=cfg.extra_args,
                executable=shutil.which("mm_segment") or "mm_segment",
            )
            ctx.logger.info("[segment] MuscleMap %s %s on %s", REGION, version, device)
            ctx.run_command(argv)
            produced = output_path(src, work)
            if not produced.is_file():
                raise StepError(
                    "mm_segment produced no segmentation (it exits with code 0 even on "
                    "errors); see logs/segment.log"
                )
            labels = read_model_labels(version)
            out_file = out_dir / OUTPUT_FILE
            normalize_output(produced, out_file, set(labels.values()))
        except (MuscleMapError, OSError) as exc:
            raise StepError(f"MuscleMap: {exc}") from exc
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return [SegmentationOutput(out_file, labels, REGION)]
