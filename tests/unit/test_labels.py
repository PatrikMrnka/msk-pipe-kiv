# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from mskpipe.io.labels import Label, LabelKind, LabelTable, LabelTableError


def table() -> LabelTable:
    return LabelTable(
        labels=(
            Label(name="femur_r", value=1, kind="bone"),
            Label(name="gluteus_maximus_r", value=21, kind="muscle"),
        )
    )


def test_roundtrip_and_lookup(tmp_path):
    path = table().save(tmp_path / "labels.json")
    loaded = LabelTable.load(path)
    assert loaded == table()
    assert loaded.get("femur_r").value == 1
    assert [lb.name for lb in loaded.of_kind(LabelKind.MUSCLE)] == ["gluteus_maximus_r"]
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "mskpipe.labels"


@pytest.mark.parametrize(
    "labels, match",
    [
        ([{"name": "a", "value": 1, "kind": "bone"}] * 2, "duplicate label names"),
        (
            [{"name": "a", "value": 1, "kind": "bone"}, {"name": "b", "value": 1, "kind": "bone"}],
            "duplicate label values",
        ),
        ([{"name": "a", "value": 0, "kind": "bone"}], "greater than or equal to 1"),
        ([{"name": "a", "value": 1, "kind": "tendon"}], "kind"),
    ],
)
def test_invalid(tmp_path, labels, match):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps({"format": "mskpipe.labels", "version": 1, "labels": labels}), encoding="utf-8"
    )
    with pytest.raises(LabelTableError, match=match):
        LabelTable.load(path)


def test_missing_file(tmp_path):
    with pytest.raises(LabelTableError, match="not found"):
        LabelTable.load(tmp_path / "labels.json")
