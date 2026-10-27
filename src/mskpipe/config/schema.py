"""Pipeline configuration schema.

The pydantic models below are the single source of truth for every
user-tunable parameter and its default. A commented YAML template is
generated from them (:func:`mskpipe.config.loader.render_template`), and
the GUI can build its forms from :func:`json_schema`.

``InputSpec`` (what to process) is deliberately separate from
``PipelineConfig`` (how to process it), so one config can drive a batch.

Two levels of checks: the ``ge``/``le``/``gt``/``lt`` bounds and validators reject values
that are impossible or make a step fail (errors); :func:`recommended` attaches the range
of sensible values to a numeric field, and values outside it are reported as warnings by
:mod:`mskpipe.config.sanity` (``mskpipe config check``, pre-run checks, GUI).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import typing
from enum import StrEnum
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

CONFIG_VERSION = 1

_IDENT = r"^[a-z][a-z0-9_]*$"
MuscleName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*_[lr]$")]
SEED_MAX = 2**32 - 1  # numpy.random.RandomState accepts 0 .. 2**32 - 1

# Names Windows reserves for devices; a run folder or file with such a stem cannot be created
WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"{port}{i}" for port in ("COM", "LPT") for i in range(1, 10)}
)


# Anatomical range of the swept coordinates [deg], OpenSim sign conventions (knee flexion
# negative); key = coordinate name without the side suffix (warnings: mskpipe.config.sanity)
ANATOMICAL_RANGES_DEG: dict[str, tuple[float, float]] = {
    "hip_flexion": (-30.0, 120.0),
    "hip_adduction": (-50.0, 30.0),
    "hip_rotation": (-45.0, 45.0),
    "knee_angle": (-120.0, 10.0),
}


def recommended(low: float | None, high: float | None, why: str = "") -> dict[str, Any]:
    """``json_schema_extra`` of a numeric field: values outside ``[low, high]`` are valid
    but reported as warnings (``None`` = no bound on that side). ``why`` is shown with
    the warning and in the GUI tooltip."""
    extra: dict[str, Any] = {"recommended": [low, high]}
    if why:
        extra["recommended_why"] = why
    return extra


def windows_reserved(name: str) -> bool:
    """Whether ``name`` is a device name on Windows (also with an extension, ``CON.txt``)."""
    return name.split(".", 1)[0].upper() in WINDOWS_RESERVED


@cache
def known_muscles() -> tuple[str, ...]:
    """Muscles of the unified label scheme with side suffix (``gluteus_maximus_r``), in
    scheme order. Read directly from ``unified_labels.yaml`` (no import of the labelmap
    package, which depends on this module)."""
    text = files("mskpipe.config").joinpath("unified_labels.yaml").read_text(encoding="utf-8")
    structures = (yaml.safe_load(text) or {}).get("structures", {})
    return tuple(n for n, s in structures.items() if (s or {}).get("kind") == "muscle")


def _unknown_names(given: list[str], known: list[str], what: str) -> None:
    unknown = [n for n in given if n not in known]
    if unknown:
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name, known, n=1)
            hints.append(f"'{name}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        raise ValueError(f"unknown {what}: {', '.join(hints)}")


def blocked_args(args: list[str], blocked: frozenset[str], tool: str) -> None:
    """Reject command-line options of ``tool`` that msk-pipe sets itself. Long options may
    be abbreviated (argparse) or written as ``--opt=value``."""
    bad = []
    for arg in args:
        opt = arg.split("=", 1)[0]
        if opt in blocked or (
            opt.startswith("--") and len(opt) > 2 and any(b.startswith(opt) for b in blocked)
        ):
            bad.append(arg)
    if bad:
        raise ValueError(
            f"{', '.join(bad)}: set by msk-pipe itself (input, output, task, device, model, "
            f"resolution or saving of {tool}); use the matching config key instead"
        )


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


# Label-scheme source of tibia and fibula for each segmentation.bones (unified_labels.yaml)
TIBIA_FIBULA_SOURCE: dict[str, str] = {
    "totalsegmentator": "ts_appendicular",
    "musclemap": "musclemap",
}


# --------------------------------------------------------------------------- input


def _subject_from_path(path: Path) -> str:
    stem = re.sub(r"\.nii(\.gz)?$", "", path.name, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem).lstrip("._-")[:64].rstrip(".")
    if windows_reserved(cleaned):
        cleaned = f"{cleaned[:62]}_s"
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

    @field_validator("subject_id")
    @classmethod
    def _windows_name(cls, value: str) -> str:
        if windows_reserved(value) or value.endswith("."):
            raise ValueError(
                f"'{value}' cannot be a folder name on Windows (device name or trailing dot)"
            )
        return value


# --------------------------------------------------------------------------- runtime


class RuntimeConfig(StrictModel):
    device: Device = Field(
        Device.CPU,
        description="Compute device; 'auto' uses an NVIDIA GPU if available, otherwise CPU.",
    )
    threads: int | None = Field(
        None,
        ge=1,
        le=1024,
        description="Max CPU threads; null = all cores (more than the cores = warning).",
        json_schema_extra={"empty": "all cores"},
    )
    seed: int = Field(0, ge=0, le=SEED_MAX, description="Random seed for stochastic steps.")
    runs_dir: Path = Field(Path("runs"), description="Directory in which run folders are created.")
    cache: bool = Field(
        True, description="Reuse step outputs whose inputs and settings are unchanged."
    )
    keep_intermediate: bool = Field(True, description="Keep intermediate outputs of all steps.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Log verbosity."
    )


# --------------------------------------------------------------------------- segmentation


# Options of TotalSegmentator 2.18 and MuscleMap that msk-pipe sets itself, or that change
# where and in which form the output is written; extra_args must not contain them.
TOTALSEGMENTATOR_CONTROLLED = frozenset(
    {
        *("-i", "-o", "-ml", "--ml", "-ta", "--task", "-d", "--device"),
        *("-rs", "--roi_subset", "-rsr", "--roi_subset_robust"),
        *("-f", "--fast", "-ff", "--fastest", "-sl", "--save_lowres"),
        *("-ho", "--higher_order_resampling", "--high_order_resampling"),
        *("--higher_order_resampling_LEGACY", "--high_order_resampling_LEGACY"),
        *("-ns", "--nr_thr_saving", "-ot", "--output_type", "-ss", "--skip_saving"),
        *("-l", "--license_number", "-lt", "--list-tasks", "-lc", "--list-classes"),
        *("--version", "--test"),
    }
)
MUSCLEMAP_CONTROLLED = frozenset(
    {
        *("-i", "--input_image", "-o", "--output_dir", "-r", "--region", "-m", "--model"),
        *("-v", "--model_version", "-g", "--use_GPU", "-s", "--overlap"),
        *("-c", "--chunk_size", "-x", "--suffix"),
    }
)


class TotalSegmentatorConfig(StrictModel):
    fast: bool = Field(
        False,
        description="Low-resolution 3 mm model (tasks total/total_mr); faster, not recommended "
        "for meshing.",
    )
    higher_order_resampling: bool = Field(
        True,
        description="Higher-order upsampling of the segmentation to the input grid "
        "(--higher_order_resampling): smooth instead of staircase masks.",
    )
    roi_subset: bool = Field(
        True,
        description="Predict only the structures the pipeline needs (tasks total/total_mr).",
    )
    nr_thr_saving: int | Literal["auto"] = Field(
        "auto",
        description="Worker processes exporting the segmentation (--nr_thr_saving); each loads "
        "PyTorch. 'auto' = from RAM + page file (1-6), fewer on small machines.",
        json_schema_extra={
            "allowed": "1 to 16 or auto",
            **recommended(1, 6, "each worker loads PyTorch (1-2 GB of memory)"),
        },
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Extra CLI arguments passed verbatim (not those msk-pipe sets itself).",
    )

    @model_validator(mode="after")
    def _check_saving(self) -> TotalSegmentatorConfig:
        if isinstance(self.nr_thr_saving, int) and not 1 <= self.nr_thr_saving <= 16:
            raise ValueError("nr_thr_saving must be 1-16 or 'auto'")
        return self

    @field_validator("extra_args")
    @classmethod
    def _not_controlled(cls, value: list[str]) -> list[str]:
        blocked_args(value, TOTALSEGMENTATOR_CONTROLLED, "TotalSegmentator")
        return value


class MuscleMapConfig(StrictModel):
    model_version: str | None = Field(
        None,
        pattern=r"^\d+(\.\d+)*$",
        description="Whole-body model version (e.g. '1.4'); null = version pinned by msk-pipe.",
        json_schema_extra={"empty": "the version pinned by msk-pipe"},
    )
    overlap: float = Field(
        90.0,
        ge=0,
        le=99,
        description="Sliding-window overlap in percent; higher = slower, possibly more accurate.",
        json_schema_extra=recommended(
            50, 95, "below 50 window edges show, above 95 inference is very slow"
        ),
    )
    chunk_size: int | Literal["auto"] = Field(
        "auto",
        description="Axial slices per inference chunk, or 'auto' (from free CPU/GPU memory); "
        "MuscleMap segments 2D slices, any positive value works (memory vs speed).",
        json_schema_extra={"allowed": "1 or more, or auto"},
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Extra CLI arguments passed verbatim (not those msk-pipe sets itself).",
    )

    @field_validator("model_version", mode="before")
    @classmethod
    def _version_text(cls, value: Any) -> Any:
        # `--set ...model_version=1.4` and unquoted YAML give a number
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(value)
        return value

    @model_validator(mode="after")
    def _check_chunk(self) -> MuscleMapConfig:
        if isinstance(self.chunk_size, int) and self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer or 'auto'")
        return self

    @field_validator("extra_args")
    @classmethod
    def _not_controlled(cls, value: list[str]) -> list[str]:
        blocked_args(value, MUSCLEMAP_CONTROLLED, "MuscleMap")
        return value


class SegmentationConfig(StrictModel):
    bones: Segmenter = Field(
        Segmenter.TOTALSEGMENTATOR,
        description="Source of all bones (pelvis, sacrum, femur, tibia, fibula): "
        "'totalsegmentator' (tibia/fibula from its appendicular_bones task; needs a "
        "free academic licence key) or 'musclemap' (no key).",
    )
    muscles: Segmenter = Field(
        Segmenter.MUSCLEMAP,
        description="Source of muscle masks; TotalSegmentator provides only the gluteal muscles.",
    )
    totalsegmentator: TotalSegmentatorConfig = Field(default_factory=TotalSegmentatorConfig)
    musclemap: MuscleMapConfig = Field(default_factory=MuscleMapConfig)

    @model_validator(mode="before")
    @classmethod
    def _removed_tibia_fibula(cls, data: Any) -> Any:
        """``tibia_fibula`` (0.1.0) was removed: all bones come from ``bones``. A value that
        agrees with ``bones`` is dropped (configs of 0.1.0 runs stay valid), a mixed one is an
        error."""
        if not isinstance(data, dict) or "tibia_fibula" not in data:
            return data
        data = dict(data)
        old = data.pop("tibia_fibula")
        bones = str(data.get("bones", Segmenter.TOTALSEGMENTATOR))
        if old not in (None, "auto", TIBIA_FIBULA_SOURCE.get(bones)):
            raise ValueError(
                f"tibia_fibula={old!r} was removed: tibia and fibula come from the same tool as "
                f"the other bones (segmentation.bones, now {bones!r}); for bones without a "
                "TotalSegmentator licence key use segmentation.bones=musclemap"
            )
        return data


# --------------------------------------------------------------------------- labelmap
# Defaults reproduce the BP pipeline (config.json: bone_preprocessing / muscle_preprocessing):
# BP used SimpleITK balls of radius 1 voxel on 1 mm CT = all offsets within 1.5 voxels,
# hence 1.5 mm. Order: closing -> 3D hole filling -> opening -> small components removed.
# See mskpipe.labelmap.clean.


class PreprocessParams(StrictModel):
    min_volume_ml: float = Field(
        0.0,
        ge=0,
        le=1000,
        description="Labels smaller than this volume [ml] are treated as missing.",
    )
    closing_radius_mm: float = Field(
        1.5,
        ge=0,
        le=10,
        description="Morphological closing radius in mm (applied first, only into background); "
        "0 disables.",
        json_schema_extra=recommended(0, 3, "larger closings merge neighbouring structures"),
    )
    opening_radius_mm: float = Field(
        1.5,
        ge=0,
        le=10,
        description="Morphological opening radius in mm (applied after hole filling); 0 disables.",
        json_schema_extra=recommended(0, 3, "an opening removes parts thinner than 2 x radius"),
    )
    fill_holes: bool = Field(True, description="Fill internal cavities of each label (3D).")
    min_component_fraction: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Connected components smaller than this fraction of the largest are removed; "
            "1 keeps only the largest, 0 keeps all."
        ),
        json_schema_extra=recommended(0.05, 1, "below 0.05 segmentation noise is kept"),
    )

    @model_validator(mode="before")
    @classmethod
    def _min_voxels(cls, data: Any) -> Any:
        """``min_voxels`` (0.1.0) became ``min_volume_ml``: converted with 1 voxel = 1 mm^3,
        the grid of the BP (configs of 0.1.0 runs stay valid)."""
        if not isinstance(data, dict) or "min_voxels" not in data:
            return data
        data = dict(data)
        voxels = data.pop("min_voxels")
        if "min_volume_ml" in data:
            raise ValueError("min_voxels was replaced by min_volume_ml; give only min_volume_ml")
        if voxels is not None:
            data["min_volume_ml"] = voxels / 1000.0
        return data


class BoneTopologyParams(StrictModel):
    repair_holes: bool = Field(
        True,
        description="Fill through-holes of thin bone (iliac wing) beyond the anatomical ones "
        "(obturator foramen); off = only report them.",
    )
    max_hole_mm: float = Field(
        40.0,
        gt=0,
        le=200,
        description="Holes wider than this are reported, never filled by interpolation "
        "(voxels of another segmentation of the same bone fill holes of any size).",
        json_schema_extra=recommended(
            10, 60, "holes in the iliac wing of TLEM 2.0 MRI were 28-48 mm (E17)"
        ),
    )
    repair_gaps: bool = Field(
        True,
        description="Bridge gaps between fragments of a long bone (fibula); off = only report.",
    )
    max_gap_mm: float = Field(
        40.0,
        gt=0,
        le=200,
        description="Longer gaps are reported, never bridged.",
        json_schema_extra=recommended(
            10, 80, "the fibula gap of TLEM 2.0 MRI was 39 mm (E17); long bridges are guesses"
        ),
    )
    min_length_ratio: float = Field(
        0.85,
        ge=0,
        le=1,
        description="Fibula shorter than this fraction of the tibia is reported as a defect.",
        json_schema_extra=recommended(0.7, 0.95, "the fibula is about 0.9-1.0 of the tibia"),
    )
    on_unfixed: Literal["warn", "fail"] = Field(
        "warn",
        description="Defect that could not be repaired: 'warn' = log and report, 'fail' = the "
        "labelmap step fails.",
    )


class BonePreprocessParams(PreprocessParams):
    min_volume_ml: float = Field(
        2.0,
        ge=0,
        le=1000,
        description="Labels smaller than this volume [ml] are treated as missing.",
        json_schema_extra=recommended(
            0, 20, "the smallest bone (fibula) is about 40 ml; larger limits may drop it"
        ),
    )
    opening_radius_mm: float = Field(
        1.5,
        ge=0,
        le=10,
        description="Morphological opening radius in mm (applied after hole filling); 0 disables.",
        json_schema_extra=recommended(
            0, 2, "the iliac wing is 2-3 mm thin; a larger opening perforates it"
        ),
    )
    topology: BoneTopologyParams = Field(default_factory=BoneTopologyParams)


class MusclePreprocessParams(PreprocessParams):
    min_volume_ml: float = Field(
        1.0,
        ge=0,
        le=1000,
        description="Labels smaller than this volume [ml] are treated as missing.",
        json_schema_extra=recommended(
            0, 10, "the smallest muscles (obturators) are about 15-20 ml"
        ),
    )


class LabelmapConfig(StrictModel):
    bones: BonePreprocessParams = Field(default_factory=BonePreprocessParams)
    muscles: MusclePreprocessParams = Field(default_factory=MusclePreprocessParams)


# --------------------------------------------------------------------------- mesh
# Defaults taken from the BP pipeline (config.json: bone_to_mesh / muscle_to_mesh).
# Meshes are written in world coordinates (NIfTI RAS+, mm); see mskpipe.geometry.surface.


_SMOOTHING = recommended(10, 100, "more flattens details")


class MeshParams(StrictModel):
    smooth_iterations: int = Field(
        30,
        ge=0,
        le=500,
        description="Windowed sinc smoothing iterations; 0 disables smoothing.",
        json_schema_extra=_SMOOTHING,
    )
    passband: float = Field(
        0.01,
        gt=0.0,
        lt=2.0,
        description="Smoothing pass band; lower = stronger smoothing.",
        json_schema_extra=recommended(
            0.001, 0.1, "VTK recommends 0.01-0.1; above 0.5 there is hardly any smoothing"
        ),
    )
    target_reduction: float = Field(
        0.8,
        ge=0.0,
        le=0.99,
        description="Fraction of triangles removed by quadric decimation; 0 disables.",
        json_schema_extra=recommended(
            0, 0.9, "stronger decimation leaves too few triangles for Muscle Wrapping"
        ),
    )
    min_component_fraction: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Surface parts smaller than this fraction of the largest part are removed; "
            "1 keeps only the largest, 0 keeps all."
        ),
        json_schema_extra=recommended(0.01, 1, "below 0.01 loose fragments are kept"),
    )


class BoneMeshParams(MeshParams):
    smooth_iterations: int = Field(
        40,
        ge=0,
        le=500,
        description="Windowed sinc smoothing iterations; 0 disables smoothing.",
        json_schema_extra=_SMOOTHING,
    )


class MeshRepairConfig(StrictModel):
    enabled: bool = Field(
        True,
        description="Repair surfaces that fail the checks: mesh-level fixes, then remeshing "
        "of the mask with topology-preserving decimation (muscles also with openings and "
        "closings).",
    )
    max_opening_mm: float = Field(
        3.0,
        ge=0,
        le=10,
        description="Largest morphological opening tried when remeshing a muscle mask (1, 1.5, "
        "2, 3 mm); cuts thin bridges, e.g. a sheet of muscle left next to a bone (genus > 0).",
        json_schema_extra=recommended(0, 3, "larger openings erase thin muscles"),
    )
    max_closing_mm: float = Field(
        6.0,
        ge=0,
        le=20,
        description="Largest morphological closing tried when remeshing a muscle mask (steps "
        "1.5 mm); fills tunnels.",
        json_schema_extra=recommended(0, 8, "larger closings fill real concavities"),
    )
    max_volume_change: float = Field(
        0.05,
        gt=0,
        le=0.5,
        description="A repaired surface is accepted only if its volume changes by at most this "
        "fraction (an opening can erase thin muscles or cut off their ends).",
        json_schema_extra=recommended(0.01, 0.1, "above 10 % a repair changes the muscle"),
    )


class MeshConfig(StrictModel):
    bones: BoneMeshParams = Field(default_factory=BoneMeshParams)
    muscles: MeshParams = Field(default_factory=MeshParams)
    repair: MeshRepairConfig = Field(default_factory=MeshRepairConfig)
    on_invalid: Literal["fail", "warn"] = Field(
        "fail",
        description="A bone surface that is still open, non-manifold or inconsistently "
        "oriented after the repair: 'fail' stops the run after the mesh step, 'warn' "
        "continues (pystaple then fails). Invalid muscles: see export.on_invalid.",
    )


# --------------------------------------------------------------------------- skeleton
# Identifiers follow msk-STAPLE. Only the algorithms of hip_model.m are ported to pystaple;
# the others can be added to the Literal types once a backend implements them.


class SkeletonConfig(StrictModel):
    backend: Literal["pystaple"] = Field("pystaple", description="Skeletal model generator.")
    side: Literal["r", "l"] = Field("r", description="Leg to model.")
    pelvis_algorithm: Literal["STAPLE"] = Field("STAPLE", description="Pelvis ACS algorithm.")
    femur_algorithm: Literal["GIBOC-cylinder"] = Field(
        "GIBOC-cylinder", description="Femur ACS algorithm."
    )
    tibia_algorithm: Literal["Kai2014"] = Field("Kai2014", description="Tibia ACS algorithm.")
    joint_definitions: Literal["auto2020"] = Field(
        "auto2020", description="Joint definition scheme."
    )
    body_mass: float = Field(
        64.0,
        ge=10,
        le=300,
        description="Subject mass in kg; sets segment masses and inertias (gait2392 ratios).",
        json_schema_extra=recommended(35, 150, "adult subject"),
    )
    geometry_format: Literal["obj", "stl"] = Field(
        "obj", description="Format of the model's visualization geometries."
    )
    geometry_reduction: float = Field(
        0.3,
        gt=0,
        le=1,
        description="Fraction of triangles kept in the visualization geometries (1 = all).",
        json_schema_extra=recommended(0.1, 1, "below 0.1 the bones look coarse"),
    )
    model_name: str = Field(
        "bone_model",
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$",
        description="File name of the model (<model_name>.osim).",
    )

    @model_validator(mode="before")
    @classmethod
    def _removed_include_fibula(cls, data: Any) -> Any:
        """``include_fibula`` (0.1.0) was removed: the tibia body geometry is always tibia +
        fibula (STAPLE convention). ``true`` is dropped (configs of 0.1.0 runs stay valid)."""
        if not isinstance(data, dict) or "include_fibula" not in data:
            return data
        data = dict(data)
        if data.pop("include_fibula") not in (True, "true", "True"):
            raise ValueError(
                "include_fibula was removed: the tibia body of the model is always tibia + "
                "fibula (muscles attached to the fibula move with the shank)"
            )
        return data

    @field_validator("model_name")
    @classmethod
    def _windows_name(cls, value: str) -> str:
        if windows_reserved(value):
            raise ValueError(f"'{value}' cannot be a file name on Windows (device name)")
        return value


# --------------------------------------------------------------------------- attachments


RENAMED_ATTACHMENT_METHODS = {"bone_registration": "atlas_registration"}


class AttachmentsConfig(StrictModel):
    method: str = Field(
        "atlas_registration",
        pattern=_IDENT,
        description="Attachment method plugin (see `mskpipe plugins`).",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Method-specific parameters, validated by the plugin. "
            "Omitted ones take the plugin defaults (see `mskpipe plugins`)."
        ),
    )

    @field_validator("method", mode="before")
    @classmethod
    def _renamed_method(cls, value: Any) -> Any:
        # renamed before 0.1.0; configs and experiment files of earlier runs still load
        return RENAMED_ATTACHMENT_METHODS.get(value, value) if isinstance(value, str) else value

    @model_validator(mode="before")
    @classmethod
    def _removed_atlas(cls, data: Any) -> Any:
        """``atlas`` (0.1.0) was never read: the atlas is the folder ``params.atlas_dir``
        (or the installed one). Dropped, so configs of 0.1.0 runs stay valid."""
        if isinstance(data, dict) and "atlas" in data:
            data = {k: v for k, v in data.items() if k != "atlas"}
        return data


# --------------------------------------------------------------------------- export
# Muscle Wrapping 2.x input. Files are written in mm with scale_factors 0.001 (MW2 runs its
# decomposition in file units; its tolerances assume mm). See mskpipe.steps.export_mw2.


class MotionConfig(StrictModel):
    coordinate: str | None = Field(
        None,
        pattern=_IDENT,
        description="Coordinate swept by the motion and driving the muscle paths; "
        "null = hip_flexion_<side>.",
        json_schema_extra={"empty": "hip_flexion_<side>"},
    )
    end_deg: float = Field(
        90.0,
        ge=-180,
        le=180,
        description="Last value of the swept coordinate [deg] (the anatomical range of the "
        "coordinate is checked before the run, the range in the model by step export_mw2).",
        json_schema_extra={
            "recommended_text": ", ".join(
                f"{name.replace('_', ' ')} {lo:g} to {hi:g}"
                for name, (lo, hi) in ANATOMICAL_RANGES_DEG.items()
            )
        },
    )
    step_deg: float = Field(
        2.0,
        gt=0,
        le=45,
        description="Step of the swept coordinate [deg].",
        json_schema_extra=recommended(
            0.5, 10, "Muscle Wrapping time grows with the frames; coarse steps miss detail"
        ),
    )


class ExportConfig(StrictModel):
    muscles: list[MuscleName] | Literal["all"] = Field(
        "all", description="Muscles exported for Muscle Wrapping 2.x, or 'all'."
    )
    exclude: list[Annotated[str, StringConstraints(pattern=r"^[a-z][a-z_]*[a-z]$")]] = Field(
        default_factory=lambda: [
            "piriformis",
            "obturator_internus",
            "obturator_externus",
            "tensor_fasciae_latae",
        ],
        description="Muscles (without side suffix) not exported for Muscle Wrapping 2.x. "
        "Default: muscles whose insertion is reached through an unsegmented tendon or "
        "fascia running around a bony pulley or down to the tibia; their insertion cannot be "
        "placed on the segmented belly (reasons in export.json). [] exports all.",
    )
    num_of_lines: int = Field(
        100,
        ge=1,
        le=2000,
        description="Fibres per muscle.",
        json_schema_extra=recommended(
            10, 500, "Muscle Wrapping time grows linearly (E13: 100 fibres x 15 muscles = 25 s)"
        ),
    )
    line_res: int = Field(
        15,
        ge=1,
        le=200,
        description="Segments per fibre (points - 1).",
        json_schema_extra=recommended(
            5, 50, "fewer segments = coarser fibre paths; more = slower Muscle Wrapping"
        ),
    )
    decomposition_method: Literal["kukacka", "hoang"] = Field(
        "kukacka", description="Muscle decomposition method of Muscle Wrapping 2.x."
    )
    bone_weights: Literal[
        "InverseDistance", "PositionOnFibreQuadraticSpline", "PositionOnFibreSCurve"
    ] = Field(
        "InverseDistance",
        description="Luca2018viaPointsAlgorithm: weights of the two bones moving a fibre point.",
    )
    pelvis_with_sacrum: bool = Field(
        True,
        description="Pelvis body geometry = hip bones + sacrum (when segmented), so that "
        "sacral fibre ends are assigned to the pelvis.",
    )
    align_child_frames: bool = Field(
        True,
        description="Rotate the child frame of joints that cannot reach the pose of the "
        "image (STAPLE 1-DOF knee) so that all bones are in that pose in the first frame; "
        "false = keep the STAPLE frames (residual reported).",
    )
    motion: MotionConfig = Field(default_factory=MotionConfig)
    muscle_analysis: bool = Field(
        True, description="Let Muscle Wrapping write fibre lengths and moment arms (analysis/)."
    )
    fibre_export: bool = Field(
        False, description="Let Muscle Wrapping write fibres and muscle meshes per frame (fibres/)."
    )
    inflate_radius_mm: float = Field(
        5.0,
        gt=0,
        le=50,
        description="An attachment outline that collapses when projected onto the muscle "
        "(missing tendon; Muscle Wrapping would crash) is replaced by the outline of a patch "
        "of the muscle surface of this radius [mm] around it (grown until it is a disc).",
        json_schema_extra=recommended(2, 15, "larger patches no longer resemble the attachment"),
    )
    on_invalid: Literal["exclude", "keep"] = Field(
        "exclude",
        description="Muscles whose mesh fails the checks (and could not be repaired in step "
        "mesh): 'exclude' "
        "from the setup XML or 'keep' (written with a warning). Attachment outline checks "
        "are reported only and never exclude a muscle.",
    )

    @field_validator("muscles")
    @classmethod
    def _known_muscles(cls, value: list[str] | str) -> list[str] | str:
        if isinstance(value, list):
            _unknown_names(value, list(known_muscles()), "muscle in export.muscles")
        return value

    @field_validator("exclude")
    @classmethod
    def _known_bases(cls, value: list[str]) -> list[str]:
        bases = list(dict.fromkeys(re.sub(r"_[lr]$", "", m) for m in known_muscles()))
        _unknown_names(value, bases, "muscle in export.exclude (names without _l/_r)")
        return value


# --------------------------------------------------------------------------- root


# export.<old> -> mesh.repair.<new> (moved after 0.1.0)
MOVED_EXPORT_KEYS = {
    "repair": "enabled",
    "repair_max_opening_mm": "max_opening_mm",
    "repair_max_closing_mm": "max_closing_mm",
    "repair_max_volume_change": "max_volume_change",
}


class PipelineConfig(StrictModel):
    config_version: Literal[1] = Field(CONFIG_VERSION, description="Schema version of this file.")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    labelmap: LabelmapConfig = Field(default_factory=LabelmapConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    skeleton: SkeletonConfig = Field(default_factory=SkeletonConfig)
    attachments: AttachmentsConfig = Field(default_factory=AttachmentsConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @model_validator(mode="before")
    @classmethod
    def _moved_keys(cls, data: Any) -> Any:
        """Repair settings moved from ``export`` to ``mesh.repair`` (configs of 0.1.0 runs
        stay valid, e.g. for ``--resume``)."""
        if not isinstance(data, dict) or not isinstance(data.get("export"), dict):
            return data
        export = dict(data["export"])
        moved = {new: export.pop(old) for old, new in MOVED_EXPORT_KEYS.items() if old in export}
        if not moved:
            return data
        mesh = dict(data.get("mesh") or {})
        repair = dict(mesh.get("repair") or {})
        for key, value in moved.items():
            repair.setdefault(key, value)
        mesh["repair"] = repair
        return {**data, "export": export, "mesh": mesh}

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


def unknown_key_hint(model: type[BaseModel], loc: tuple[Any, ...]) -> str:
    """`` (did you mean 'x'?)`` for an unknown key at ``loc`` of ``model``; '' if no close
    match (or ``loc`` leaves the nested models)."""
    for part in loc[:-1]:
        field = model.model_fields.get(str(part))
        sub = (
            next(
                (
                    a
                    for a in (field.annotation, *typing.get_args(field.annotation))
                    if isinstance(a, type) and issubclass(a, BaseModel)
                ),
                None,
            )
            if field is not None
            else None
        )
        if sub is None:
            return ""
        model = sub
    close = difflib.get_close_matches(str(loc[-1]), list(model.model_fields), n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""
