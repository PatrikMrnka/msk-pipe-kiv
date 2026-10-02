# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage

from mskpipe.config import InputSpec, load_config
from mskpipe.core.runner import PipelineError, run_pipeline
from mskpipe.core.step import Step, StepContext
from mskpipe.io.labels import LABELMAP_FILE, LABELS_FILE, LabelTable
from mskpipe.io.segmentation import SEGMENTATION_FILE, SegmentationIndex, ToolOutput
from mskpipe.steps.labelmap import INDEX_FILE, LabelmapStep, read_labelmap_index
from mskpipe.steps.mesh import MeshStep, mesh_paths

SHAPE = (60, 30, 40)
# x is flipped: world x = 59 - i, so the patient's right (+x) is at low i.
AFFINE = np.array(
    [[-1.0, 0.0, 0.0, 59.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.5, -100.0], [0, 0, 0, 1]]
)
RIGHT, LEFT = slice(14, 20), slice(40, 46)


def _ts_total() -> tuple[np.ndarray, dict[str, int]]:
    d = np.zeros(SHAPE, np.uint8)
    d[10:25, 8:22, 28:36] = 78  # hip_right
    d[35:50, 8:22, 28:36] = 77  # hip_left
    d[RIGHT, 12:18, 10:27] = 76  # femur_right
    d[LEFT, 12:18, 10:27] = 75  # femur_left
    labels = {"femur_left": 75, "femur_right": 76, "hip_left": 77, "hip_right": 78}
    return d, labels


def _ts_appendicular() -> tuple[np.ndarray, dict[str, int]]:
    d = np.zeros(SHAPE, np.uint8)
    d[RIGHT, 12:18, 0:11] = 3  # tibia, both legs in one label; overlaps the femur at k = 10
    d[LEFT, 12:18, 0:11] = 3
    return d, {"tibia": 3, "patella": 2}


def _musclemap() -> tuple[np.ndarray, dict[str, int]]:
    d = np.zeros(SHAPE, np.uint16)
    d[8:26, 20:28, 20:34] = 6122  # gluteus_maximus_r, overlaps hip_right at j = 20..21
    d[30:32, 2:4, 2:4] = 6182  # piriformis_r: 8 voxels, below min_voxels
    labels = {"gluteus_maximus_r": 6122, "iliacus_r": 6142, "piriformis_r": 6182}
    return d, labels


class FakeSegment(Step):
    """Writes tool outputs and segmentation.json like the segment step."""

    name = "segment"
    outputs = ("ts_total", "ts_appendicular", "musclemap")
    affine = AFFINE

    def run(self, ctx: StepContext) -> None:
        makers = {
            "ts_total": ("totalsegmentator", "total", _ts_total),
            "ts_appendicular": ("totalsegmentator", "appendicular_bones", _ts_appendicular),
            "musclemap": ("musclemap", "default", _musclemap),
        }
        entries = []
        for key in FakeSegment.outputs:
            tool, task, make = makers[key]
            data, labels = make()
            affine = FakeSegment.affine if key == "musclemap" else AFFINE
            nib.save(nib.Nifti1Image(data, affine), ctx.out_dir / f"{key}.nii.gz")
            entries.append(ToolOutput(tool=tool, task=task, file=f"{key}.nii.gz", labels=labels))
        SegmentationIndex(outputs=tuple(entries)).save(ctx.out_dir / SEGMENTATION_FILE)


@pytest.fixture(autouse=True)
def _reset():
    FakeSegment.outputs = ("ts_total", "ts_appendicular", "musclemap")
    FakeSegment.affine = AFFINE


@pytest.fixture
def spec(tmp_path: Path) -> InputSpec:
    image = tmp_path / "subject.nii.gz"
    image.write_bytes(b"fake volume")
    return InputSpec(image=image, modality="ct")


def config(tmp_path: Path, *overrides: str):
    return load_config(
        overrides=[
            f"runtime.runs_dir={(tmp_path / 'runs').as_posix()}",
            "segmentation.tibia_fibula=ts_appendicular",
            "labelmap.bones.min_voxels=100",
            "labelmap.muscles.min_voxels=100",
            *overrides,
        ]
    )


def _run(tmp_path, spec, *overrides, steps=None):
    steps = steps or [FakeSegment(), LabelmapStep()]
    return run_pipeline(steps, spec, config(tmp_path, *overrides))


def _load(result):
    out = result.ws.step_dir("labelmap")
    img = nib.load(out / LABELMAP_FILE)
    return out, np.asanyarray(img.dataobj), img.affine, LabelTable.load(out / LABELS_FILE)


def test_unified_labelmap(tmp_path, spec):
    result = _run(tmp_path, spec)
    out, data, affine, table = _load(result)
    assert data.dtype == np.uint8 and data.shape == SHAPE
    assert np.allclose(affine, AFFINE)

    v = {label.name: label.value for label in table}
    assert v["pelvis_no_sacrum"] == 1 and v["femur_r"] == 2
    # both hip bones in one label, cleaned separately (never fused)
    assert ndimage.label(data == v["pelvis_no_sacrum"])[1] == 2
    # sides by world coordinates: right = +x = low i here
    assert (data[RIGHT] == v["femur_r"]).any() and not (data[LEFT] == v["femur_r"]).any()
    assert (data[15:19, 13:17, 1:8] == v["tibia_r"]).all()  # edges rounded by the opening
    assert not (data[LEFT] == v["tibia_r"]).any()
    # only the skeleton.side leg (r) and the pelvis are built
    assert "femur_l" not in v and "tibia_l" not in v
    assert not data[LEFT, :, :27].any()
    # femur and tibia both claim k = 10 -> background (joint space)
    assert (data[RIGHT, 12:18, 10] == 0).all()
    # muscle voxels inside the hip bone go to the bone
    assert (data[11:24, 20:22, 29:34] == v["pelvis_no_sacrum"]).all()
    assert (data[10:25, 22:27, 28:33] == v["gluteus_maximus_r"]).all()

    index = read_labelmap_index(out)
    st = {s["name"]: s for s in index["structures"]}
    assert st["piriformis_r"]["status"] == "too_small" and not (data == v["piriformis_r"]).any()
    assert st["iliacus_r"]["status"] == "missing"
    assert st["tibia_r"]["source_labels"] == ["tibia"]
    assert st["pelvis_no_sacrum"]["parts"] == ["hip_left", "hip_right"]
    assert st["gluteus_maximus_r"]["lost_to_bone"] > 0
    assert st["femur_r"]["contested"] == st["tibia_r"]["contested"] > 0
    assert index["midline"]["from"] == "pelvis_no_sacrum"
    assert index["sources"] == {
        "bones": "totalsegmentator",
        "tibia_fibula": "ts_appendicular",
        "muscles": "musclemap",
    }

    rec = result.manifest.steps["labelmap"]
    m = rec.metrics
    assert m["too_small"] == ["piriformis_r"] and "iliacus_r" in m["missing"]
    assert m["contested_voxels"] == 2 * st["femur_r"]["contested"]
    assert m["lost_to_bone_voxels"] > 0
    assert set(m["structures"]) == {"pelvis_no_sacrum", "femur_r", "tibia_r", "gluteus_maximus_r"}
    assert {f"02_labelmap/{n}" for n in (LABELMAP_FILE, LABELS_FILE, INDEX_FILE)} <= set(
        rec.outputs
    )
    assert json.loads((out / INDEX_FILE).read_text(encoding="utf-8"))["params"]["bones"]


def test_mesh_reads_labelmap(tmp_path, spec):
    result = _run(tmp_path, spec, steps=[FakeSegment(), LabelmapStep(), MeshStep()])
    bones = mesh_paths(result.ws.step_dir("mesh"), "bone")
    assert set(bones) == {"pelvis_no_sacrum", "femur_r", "tibia_r"}


def test_left_side(tmp_path, spec):
    result = _run(tmp_path, spec, "skeleton.side=l")
    _, data, _, table = _load(result)
    names = {label.name for label in table}
    assert {"femur_l", "tibia_l", "gluteus_maximus_l"} <= names
    assert not any(n.endswith("_r") for n in names)
    assert (data[41:45, 13:17, 1:8] == table.get("tibia_l").value).all()
    assert not data[RIGHT, :, :27].any()


def test_image_centre_midline_without_pelvis(tmp_path, spec):
    FakeSegment.outputs = ("ts_appendicular",)
    result = _run(tmp_path, spec, "segmentation.bones=musclemap", "segmentation.muscles=musclemap")
    _, data, _, table = _load(result)
    index = read_labelmap_index(result.ws.step_dir("labelmap"))
    assert index["midline"] == {"x_mm": 29.5, "from": "image_centre"}
    assert (data[15:19, 13:17, 1:8] == table.get("tibia_r").value).all()


def test_closing_params_change_cache_key(tmp_path, spec):
    first = _run(tmp_path, spec)
    again = _run(tmp_path, spec)
    assert again.manifest.steps["labelmap"].status == "cached"
    other = _run(tmp_path, spec, "labelmap.bones.closing_radius_mm=0")
    assert other.manifest.steps["labelmap"].status == "completed"
    assert first.ws.run_key != other.ws.run_key


def test_grid_mismatch_fails(tmp_path, spec):
    FakeSegment.affine = AFFINE + np.diag([0, 0, 0.5, 0])
    with pytest.raises(PipelineError, match="same voxel grid"):
        _run(tmp_path, spec)


def test_missing_index_fails(tmp_path, spec):
    with pytest.raises(PipelineError, match="Segmentation index not found"):
        _run(tmp_path, spec, steps=[LabelmapStep()])


def test_no_output_for_sources_fails(tmp_path, spec):
    FakeSegment.outputs = ("musclemap",)
    with pytest.raises(PipelineError, match="No segmentation output"):
        _run(
            tmp_path,
            spec,
            "segmentation.bones=totalsegmentator",
            "segmentation.muscles=totalsegmentator",
        )
