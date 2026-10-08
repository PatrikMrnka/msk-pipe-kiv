# SPDX-License-Identifier: Apache-2.0
"""Every setting has an explanation in config/help.yaml, and no explanation is stale."""

from mskpipe.config import json_schema
from mskpipe.config.help import help_for, load_help, plugin_help
from mskpipe.core.registry import default_registry

GUI_CONTROLS = {"image", "modality", "subject", "device", "runs_dir", "until", "verbose"}
NOT_SHOWN = {"config_version", "runtime.device", "runtime.runs_dir"}  # window controls
PLUGIN_GROUPS = {"attachments.params"}  # free objects in the schema, groups in the GUI


def _paths(node, defs, prefix=()):
    """(leaf paths, group paths) of the config schema."""
    while "$ref" in node:
        node = {
            **defs[node["$ref"].rsplit("/", 1)[-1]],
            **{k: v for k, v in node.items() if k != "$ref"},
        }
    if node.get("type") == "object" and "properties" in node:
        leaves, groups = [], [prefix] if prefix else []
        for name, child in node["properties"].items():
            sub_leaves, sub_groups = _paths(child, defs, (*prefix, name))
            leaves += sub_leaves
            groups += sub_groups
        return leaves, groups
    return [prefix], []


def _schema_paths():
    schema = json_schema()
    leaves, groups = _paths(schema, schema["$defs"])
    leaves = {".".join(p) for p in leaves} - NOT_SHOWN - PLUGIN_GROUPS
    return leaves, {".".join(p) for p in groups} | PLUGIN_GROUPS


def _plugin_keys():
    reg = default_registry()
    keys = set()
    for info in reg.describe():
        if info.source != "mskpipe":
            continue
        plugin = reg.get(info.kind, info.name)
        for param in plugin.Params.model_fields:
            keys.add(f"plugins.{info.kind}.{info.name}.{param}")
    return keys


def test_every_setting_is_explained():
    leaves, _ = _schema_paths()
    missing = sorted(k for k in leaves | _plugin_keys() if k not in load_help())
    assert missing == []


def test_no_stale_explanations():
    leaves, groups = _schema_paths()
    known = leaves | groups | _plugin_keys() | {f"gui.{c}" for c in GUI_CONTROLS}
    assert sorted(set(load_help()) - known) == []


def test_lookup_and_examples():
    entry = help_for(("mesh", "bones", "smooth_iterations"))
    assert entry is not None and entry.example and "  " not in entry.text
    assert (
        help_for("labelmap.muscles.min_voxels").text == help_for("labelmap.bones.min_voxels").text
    )
    assert plugin_help("attachments", "bone_registration", "nonrigid") is not None
    assert help_for("nope.nope") is None
    leaves, _ = _schema_paths()
    without_example = sorted(k for k in leaves if load_help()[k].example is None)
    assert without_example == []
