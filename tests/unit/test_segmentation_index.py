# SPDX-License-Identifier: Apache-2.0
import pytest

from mskpipe.io.segmentation import SegmentationIndex, SegmentationIndexError, ToolOutput


def test_roundtrip(tmp_path):
    index = SegmentationIndex(
        outputs=(
            ToolOutput(tool="totalsegmentator", task="total", file="ts.nii.gz", labels={"a": 1}),
            ToolOutput(tool="musclemap", file="mm.nii.gz", labels={"b": 6122}),
        )
    )
    path = index.save(tmp_path / "segmentation.json")
    loaded = SegmentationIndex.load(path)
    assert loaded == index and len(loaded) == 2
    assert [o.tool for o in loaded] == ["totalsegmentator", "musclemap"]


def test_invalid(tmp_path):
    with pytest.raises(SegmentationIndexError, match="not found"):
        SegmentationIndex.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "other", "version": 1}', encoding="utf-8")
    with pytest.raises(SegmentationIndexError, match="Invalid"):
        SegmentationIndex.load(bad)
    with pytest.raises(ValueError):
        ToolOutput(tool="x", file="a.nii.gz", labels={"a": 0})
    with pytest.raises(ValueError, match="duplicate"):
        SegmentationIndex(
            outputs=(
                ToolOutput(tool="x", file="a", labels={}),
                ToolOutput(tool="y", file="a", labels={}),
            )
        )
