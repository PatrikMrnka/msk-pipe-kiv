"""Pipeline configuration schema.

The pydantic models below are the single source of truth for every
user-tunable parameter and its default. A commented YAML template is
generated from them (:func:`mskpipe.config.loader.render_template`), and
the GUI can build its forms from :func:`json_schema`.

``InputSpec`` (what to process) is deliberately separate from
``PipelineConfig`` (how to process it), so one config can drive a batch.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

CONFIG_VERSION = 1

_IDENT = r"^[a-z][a-z0-9_]*$"
MuscleName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*_[lr]$")]


class StrictModel(BaseModel):
    """Base model: unknown keys are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


# --------------------------------------------------------------------------- enums


class Modality(StrEnum):
    CT = "ct"
    MRI = "mri"


class Device(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class Segmenter(StrEnum):
    TOTALSEGMENTATOR = "totalsegmentator"
    MUSCLEMAP = "musclemap"


# --------------------------------------------------------------------------- input


def _subject_from_path(path: Path) -> str:
    stem = re.sub(r"\.nii(\.gz)?$", "", path.name, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem).lstrip("._-")[:64]
    return cleaned or "subject"


class InputSpec(StrictModel):
    """One input volume. Not part of the config file (supplied by CLI/GUI/batch)."""

    image: Path = Field(description="Input volume (.nii or .nii.gz).")
    modality: Modality = Field(description="Imaging modality of the input volume.")
    subject_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        description="Identifier used in run names and outputs; defaults to the file name.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_subject_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("subject_id") and data.get("image"):
            data = {**data, "subject_id": _subject_from_path(Path(data["image"]))}
        return data

    @model_validator(mode="after")
    def _check_suffix(self) -> InputSpec:
        if not self.image.name.lower().endswith((".nii", ".nii.gz")):
            raise ValueError(
                f"image must be a NIfTI file (.nii or .nii.gz), got '{self.image.name}'"
            )
        return self


# --------------------------------------------------------------------------- runtime


class RuntimeConfig(StrictModel):
    device: Device = Field(
        Device.CPU,
        description="Compute device; 'auto' uses an NVIDIA GPU if available, otherwise CPU.",
    )
    threads: int | None = Field(None, ge=1, description="Max CPU threads; null = all cores.")
    seed: int = Field(0, ge=0, description="Random seed for stochastic steps.")
    runs_dir: Path = Field(Path("runs"), description="Directory in which run folders are created.")
    cache: bool = Field(
        True, description="Reuse step outputs whose inputs and settings are unchanged."
    )
    keep_intermediate: bool = Field(True, description="Keep intermediate outputs of all steps.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Log verbosity."
    )


# --------------------------------------------------------------------------- segmentation


class TotalSegmentatorConfig(StrictModel):
    fast: bool = Field(
        False, description="Low-resolution 3 mm model; faster, not recommended for meshing."
    )
    higher_order_resampling: bool = Field(
        True, description="Higher-order resampling to input space (avoids staircase masks)."
    )
    roi_subset: bool = Field(True, description="Predict only the structures the pipeline needs.")
    extra_args: list[str] = Field(
        default_factory=list, description="Extra CLI arguments passed verbatim."
    )


class MuscleMapConfig(StrictModel):
    model_version: str | None = Field(
        None, description="Model weights version; null = version pinned by msk-pipe."
    )
    extra_args: list[str] = Field(
        default_factory=list, description="Extra CLI arguments passed verbatim."
    )


class SegmentationConfig(StrictModel):
    bones: Segmenter = Field(
        Segmenter.TOTALSEGMENTATOR, description="Source of pelvis and femur masks."
    )
    tibia_fibula: Literal["musclemap", "ts_appendicular"] = Field(
        "musclemap",
        description="Tibia/fibula source; 'ts_appendicular' needs a TotalSegmentator licence.",
    )
    muscles: Segmenter = Field(
        Segmenter.MUSCLEMAP,
        description="Source of muscle masks; TotalSegmentator provides only the gluteal muscles.",
    )
    totalsegmentator: TotalSegmentatorConfig = Field(default_factory=TotalSegmentatorConfig)
    musclemap: MuscleMapConfig = Field(default_factory=MuscleMapConfig)


# --------------------------------------------------------------------------- labelmap
# Defaults taken from the BP pipeline (config.json: bone_preprocessing / muscle_preprocessing).
# Kernels are ball radii in voxels (as in BP: sitk.BinaryMorphologicalClosing/Opening,
# radius [k, k, k]); closing is applied first, then opening.


class PreprocessParams(StrictModel):
    min_voxels: int = Field(0, ge=0, description="Labels with fewer voxels are treated as missing.")
    opening_kernel: int = Field(
        1,
        ge=0,
        description="Morphological opening radius in voxels (applied after closing); 0 disables.",
    )
    closing_kernel: int = Field(
        1, ge=0, description="Morphological closing radius in voxels (applied first); 0 disables."
    )
    fill_holes: bool = Field(True, description="Fill internal holes of each label (3D).")
    keep_largest: bool = Field(
        True, description="Keep only the largest connected component of each label."
    )


class BonePreprocessParams(PreprocessParams):
    min_voxels: int = Field(
        2000, ge=0, description="Labels with fewer voxels are treated as missing."
    )


class MusclePreprocessParams(PreprocessParams):
    min_voxels: int = Field(
        1000, ge=0, description="Labels with fewer voxels are treated as missing."
    )
    bone_subtraction_dilation_radius: int = Field(
        0,
        ge=0,
        description="Bones dilated by this radius (voxels) are subtracted from muscles; 0 = none.",
    )


class LabelmapConfig(StrictModel):
    bones: BonePreprocessParams = Field(default_factory=BonePreprocessParams)
    muscles: MusclePreprocessParams = Field(default_factory=MusclePreprocessParams)


# --------------------------------------------------------------------------- mesh
# Defaults taken from the BP pipeline (config.json: bone_to_mesh / muscle_to_mesh).


class MeshParams(StrictModel):
    smooth_iterations: int = Field(
        30, ge=0, le=500, description="Windowed sinc smoothing iterations; 0 disables smoothing."
    )
    passband: float = Field(
        0.01, gt=0.0, lt=2.0, description="Smoothing pass band; lower = stronger smoothing."
    )
    target_reduction: float = Field(
        0.8,
        ge=0.0,
        lt=1.0,
        description="Fraction of triangles removed by quadric decimation; 0 disables.",
    )


class BoneMeshParams(MeshParams):
    smooth_iterations: int = Field(
        40, ge=0, le=500, description="Windowed sinc smoothing iterations; 0 disables smoothing."
    )


class MeshConfig(StrictModel):
    bones: BoneMeshParams = Field(default_factory=BoneMeshParams)
    muscles: MeshParams = Field(default_factory=MeshParams)


# --------------------------------------------------------------------------- skeleton
# Identifiers follow msk-STAPLE; verified against pystaple in tests/steps/test_skeleton.py.


class SkeletonConfig(StrictModel):
    backend: Literal["pystaple"] = Field("pystaple", description="Skeletal model generator.")
    side: Literal["r", "l"] = Field("r", description="Leg to model.")
    pelvis_algorithm: Literal["STAPLE", "Kai2014"] = Field(
        "STAPLE", description="Pelvis ACS algorithm."
    )
    femur_algorithm: Literal[
        "GIBOC-cylinder", "GIBOC-spheres", "GIBOC-ellipsoids", "Miranda", "Kai2014"
    ] = Field("GIBOC-cylinder", description="Femur ACS algorithm.")
    tibia_algorithm: Literal["Kai2014", "GIBOC-plateau", "GIBOC-ellipse", "GIBOC-centroids"] = (
        Field("Kai2014", description="Tibia ACS algorithm.")
    )
    joint_definitions: Literal["auto2020", "Modenese2018"] = Field(
        "auto2020", description="Joint definition scheme."
    )
    model_name: str = Field(
        "bone_model", pattern=r"^[A-Za-z][A-Za-z0-9_]*$", description="OpenSim model name."
    )


# --------------------------------------------------------------------------- attachments


class AttachmentsConfig(StrictModel):
    method: str = Field(
        "atlas_based",
        pattern=_IDENT,
        description="Attachment method plugin (see `mskpipe plugins`).",
    )
    atlas: str = Field(
        "lhdl", pattern=_IDENT, description="Atlas providing reference attachment areas."
    )
    params: dict[str, Any] = Field(
        default_factory=lambda: {"threshold_mm": 5.0, "clip_fraction": 0.22},
        description="Method-specific parameters, validated by the plugin.",
    )


# --------------------------------------------------------------------------- export


class ExportConfig(StrictModel):
    muscles: list[MuscleName] | Literal["all"] = Field(
        "all", description="Muscles exported for Muscle Wrapping 2.x, or 'all'."
    )
    units: Literal["m", "mm"] = Field("m", description="Length unit of exported .obj/.xml files.")


# --------------------------------------------------------------------------- root


class PipelineConfig(StrictModel):
    config_version: Literal[1] = Field(CONFIG_VERSION, description="Schema version of this file.")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    labelmap: LabelmapConfig = Field(default_factory=LabelmapConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    skeleton: SkeletonConfig = Field(default_factory=SkeletonConfig)
    attachments: AttachmentsConfig = Field(default_factory=AttachmentsConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @model_validator(mode="after")
    def _export_side_matches_skeleton(self) -> PipelineConfig:
        if self.export.muscles != "all":
            wrong = [m for m in self.export.muscles if not m.endswith(f"_{self.skeleton.side}")]
            if wrong:
                raise ValueError(
                    f"export.muscles {wrong} do not match skeleton.side='{self.skeleton.side}'"
                )
        return self

    def fingerprint(self, *sections: str) -> str:
        """Stable SHA-256 of the given top-level sections (used for step caching)."""
        known = set(type(self).model_fields) - {"config_version"}
        unknown = set(sections) - known
        if not sections or unknown:
            raise ValueError(
                f"fingerprint needs sections from {sorted(known)}, got {sorted(sections)}"
            )
        payload = {
            "config_version": self.config_version,
            **self.model_dump(mode="json", include=set(sections)),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def json_schema() -> dict[str, Any]:
    """JSON Schema of the config file (editor completion, GUI forms)."""
    return PipelineConfig.model_json_schema()
