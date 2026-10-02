# SPDX-License-Identifier: Apache-2.0
import pytest

from mskpipe.config import load_config
from mskpipe.labelmap.scheme import SchemeError, load_scheme, parse_scheme, scheme_sha256


def test_bundled_scheme_is_valid():
    scheme = load_scheme()
    assert scheme.structures["pelvis_no_sacrum"].kind == "bone"
    assert len([s for s in scheme.structures.values() if s.kind == "muscle"]) == 42
    assert all(1 <= s.value <= 255 for s in scheme.structures.values())
    assert len(scheme_sha256()) == 64


def test_default_plan_uses_ts_bones_and_musclemap():
    plan = load_scheme().plan(load_config())
    assert plan["pelvis_no_sacrum"] == ("totalsegmentator", ("hip_left", "hip_right"))
    assert plan["femur_r"] == ("totalsegmentator", ("femur_right",))
    assert plan["tibia_r"] == ("musclemap", ("tibia_r",))
    assert plan["gluteus_maximus_r"] == ("musclemap", ("gluteus_maximus_r",))
    assert len(plan) == 25  # skeleton.side (r) leg + pelvis
    assert not any(name.endswith("_l") for name in plan)


def test_plan_side():
    left = load_scheme().plan(load_config(overrides=["skeleton.side=l"]))
    assert "femur_l" in left and "femur_r" not in left and "pelvis_no_sacrum" in left
    both = load_scheme().plan(load_config(), sides=("r", "l"))
    assert len(both) == 49


def test_plan_follows_segmentation_config():
    config = load_config(
        overrides=[
            "segmentation.bones=musclemap",
            "segmentation.tibia_fibula=ts_appendicular",
            "segmentation.muscles=totalsegmentator",
        ]
    )
    plan = load_scheme().plan(config, sides=("r", "l"))
    assert plan["pelvis_no_sacrum"] == ("musclemap", ("ilium_l", "ilium_r"))
    assert plan["tibia_r"] == plan["tibia_l"] == ("ts_appendicular", ("tibia",))
    muscles = {n for n in plan if load_scheme().structures[n].kind == "muscle"}
    assert muscles == {f"gluteus_{m}_{s}" for m in ("maximus", "medius", "minimus") for s in "rl"}


def test_split_names():
    scheme = load_scheme()
    assert scheme.sources["ts_appendicular"].split_names() == {"tibia", "fibula"}
    assert scheme.sources["musclemap"].split_names() == set()


BASE = """
structures:
  femur_r: {value: 1, kind: bone, group: bones}
  femur_l: {value: 2, kind: bone, group: bones}
  tibia_r: {value: 3, kind: bone, group: tibia_fibula}
sources:
  totalsegmentator:
    tool: totalsegmentator
    labels:
"""


@pytest.mark.parametrize(
    "labels, match",
    [
        ("      femur_r: [femur]\n      tibia_r: [femur]\n", "only by one _l/_r pair"),
        ("      knee_r: [knee]\n", "unknown structures"),
        ("      femur_r: []\n", "no labels"),
    ],
)
def test_invalid_scheme(labels, match):
    with pytest.raises(SchemeError, match=match):
        parse_scheme(BASE + labels)


def test_duplicate_values_rejected():
    text = BASE.replace("value: 2", "value: 1") + "      femur_r: [femur_right]\n"
    with pytest.raises(SchemeError, match="unique"):
        parse_scheme(text)
