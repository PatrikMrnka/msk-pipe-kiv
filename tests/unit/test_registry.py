# SPDX-License-Identifier: Apache-2.0
import sys
from collections.abc import Mapping
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
from pydantic import Field

from mskpipe.config import ConfigError, PipelineConfig, load_config
from mskpipe.core import registry as registry_mod
from mskpipe.core.registry import (
    BUILTINS,
    PluginError,
    PluginSpec,
    Registry,
    format_plugins,
    resolve_plugins,
)
from mskpipe.core.runner import step_fingerprint
from mskpipe.core.step import Step, StepContext
from mskpipe.plugins.base import AttachmentsPlugin, PluginParams, SkeletonPlugin


class DemoParams(PluginParams):
    radius_mm: float = Field(2.0, gt=0)
    iterations: int = 10


class Demo(AttachmentsPlugin):
    name = "demo"
    version = "3"
    description = "Test method."
    Params = DemoParams

    def compute(self, ctx, bones, muscles, out_dir, params) -> Mapping[str, Path]:
        return {}


class NeedsTool(AttachmentsPlugin):
    name = "needs_tool"
    requires_modules = ("surely_not_installed_pkg",)
    requires_executables = ("surely-not-a-tool",)

    def compute(self, ctx, bones, muscles, out_dir, params) -> Mapping[str, Path]:
        return {}


class Skel(SkeletonPlugin):
    name = "pystaple"

    def build(self, ctx, bones, out_dir, params) -> Path:
        return out_dir / "m.osim"


class AtlasBased(Demo):
    name = "atlas_based"
    Params = PluginParams


@pytest.fixture
def reg() -> Registry:
    r = Registry()
    for plugin in (Demo, NeedsTool, Skel, AtlasBased):
        r.register(plugin)
    return r


# ---------------------------------------------------------------------- registration


def test_get_and_names(reg):
    assert reg.get("attachments", "demo") is Demo
    assert reg.names("attachments") == ["atlas_based", "demo", "needs_tool"]
    assert reg.names("segmenter") == []


def test_unknown_plugin_lists_available(reg):
    with pytest.raises(PluginError, match="available: atlas_based, demo, needs_tool"):
        reg.get("attachments", "cpd")
    assert issubclass(PluginError, ConfigError)


def test_duplicate_and_bad_names(reg):
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Demo)
    with pytest.raises(ValueError, match="Invalid plugin name"):
        reg.add(PluginSpec("attachments", "Bad-Name", "x:y"))
    with pytest.raises(ValueError, match="Unknown plugin kind"):
        reg.add(PluginSpec("mesher", "x", "x:y"))  # type: ignore[arg-type]


def test_wrong_kind_is_rejected():
    reg = Registry([PluginSpec("skeleton", "demo", f"{__name__}:Demo")])
    with pytest.raises(PluginError, match="not a subclass of SkeletonPlugin"):
        reg.get("skeleton", "demo")


def test_name_mismatch_is_rejected():
    reg = Registry([PluginSpec("attachments", "other", f"{__name__}:Demo")])
    with pytest.raises(PluginError, match="declares name 'demo'"):
        reg.get("attachments", "other")


def test_unloadable_target():
    reg = Registry([PluginSpec("attachments", "ghost", "no_such_module_xyz:Ghost")])
    with pytest.raises(PluginError, match="Cannot load"):
        reg.get("attachments", "ghost")
    (info,) = reg.describe()
    assert info.error and not info.available


# ---------------------------------------------------------------------- params


def test_validate_params_defaults_and_values(reg):
    assert reg.validate_params("attachments", "demo", {}, where="attachments") == DemoParams()
    p = reg.validate_params("attachments", "demo", {"iterations": "5"}, where="attachments")
    assert p.iterations == 5


@pytest.mark.parametrize(
    ("params", "path"),
    [
        ({"radius_mm": -1}, r"attachments\.params\.radius_mm"),
        ({"typo": 1}, r"attachments\.params\.typo"),
    ],
)
def test_validate_params_errors_use_config_paths(reg, params, path):
    with pytest.raises(PluginError, match=path):
        reg.validate_params("attachments", "demo", params, where="attachments")


def test_resolve_fills_defaults(reg):
    cfg = load_config(overrides=["attachments.method=demo", "attachments.params={iterations: 7}"])
    resolved = resolve_plugins(cfg, reg)
    assert resolved.attachments.params == {"radius_mm": 2.0, "iterations": 7}
    assert cfg.attachments.params == {"iterations": 7}  # input left untouched
    assert resolve_plugins(resolved, reg) == resolved  # idempotent
    assert isinstance(resolved, PipelineConfig)


def test_resolve_rejects_unknown_method(reg):
    cfg = load_config(overrides=["attachments.method=cpd"])
    with pytest.raises(PluginError, match="Unknown attachments plugin 'cpd'"):
        resolve_plugins(cfg, reg)


# ---------------------------------------------------------------------- availability, listing


def test_missing_dependencies(reg):
    assert NeedsTool.missing() == ["surely_not_installed_pkg", "surely-not-a-tool"]
    with pytest.raises(PluginError, match="needs surely_not_installed_pkg"):
        reg.create("attachments", "needs_tool")
    assert isinstance(reg.create("attachments", "demo"), Demo)


def test_describe_and_format(reg):
    infos = {i.name: i for i in reg.describe("attachments")}
    assert infos["demo"].params == {"radius_mm": 2.0, "iterations": 10}
    assert infos["demo"].params_schema["properties"]["radius_mm"]["exclusiveMinimum"] == 0
    assert infos["demo"].available and not infos["needs_tool"].available
    text = format_plugins(infos.values())
    assert "demo" in text and "v3" in text and "missing surely_not_installed_pkg" in text


def test_identity(reg):
    assert reg.identity("attachments", "demo") == {
        "plugin": "attachments/demo",
        "version": "3",
        "source": "local",
    }
    builtin = Registry.default(external=False).identity("attachments", "atlas_based")
    assert builtin == {"plugin": "attachments/atlas_based", "version": "1"}


# ---------------------------------------------------------------------- entry points


def test_entry_points(monkeypatch, tmp_path):
    (tmp_path / "ext_plugin_mod.py").write_text(
        "from mskpipe.plugins.base import AttachmentsPlugin\n"
        "class Ext(AttachmentsPlugin):\n"
        "    name = 'ext'\n"
        "    def compute(self, ctx, bones, muscles, out_dir, params):\n"
        "        return {}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    eps = {
        "mskpipe.attachments": [
            EntryPoint("ext", "ext_plugin_mod:Ext", "mskpipe.attachments"),
            EntryPoint("atlas_based", "ext_plugin_mod:Ext", "mskpipe.attachments"),  # clash
        ]
    }
    monkeypatch.setattr(registry_mod, "entry_points", lambda group: eps.get(group, []))
    reg = Registry.default()
    assert reg.get("attachments", "ext").name == "ext"
    assert reg.spec("attachments", "atlas_based").source == "mskpipe"  # built-in wins
    assert reg.identity("attachments", "ext")["source"] == "external"


# ---------------------------------------------------------------------- built-ins


def test_builtins_load_and_resolve_defaults():
    reg = Registry.default(external=False)
    assert {(k, n) for k, n, _ in BUILTINS} == {(i.kind, i.name) for i in reg.describe()}
    assert all(i.error is None for i in reg.describe())
    resolved = resolve_plugins(PipelineConfig(), reg)
    assert resolved.attachments.params == {"threshold_mm": 5.0, "clip_fraction": 0.22}


def test_listing_builtins_imports_no_heavy_modules(monkeypatch):
    for mod in [
        m for m in sys.modules if m.startswith("mskpipe.plugins.") and m != "mskpipe.plugins.base"
    ]:
        monkeypatch.delitem(sys.modules, mod)
    before = set(sys.modules)
    Registry.default(external=False).describe()
    new_roots = {m.split(".")[0] for m in set(sys.modules) - before}
    heavy = {"torch", "vtk", "SimpleITK", "totalsegmentator", "pystaple", "nibabel", "trimesh"}
    assert not heavy & new_roots


# ---------------------------------------------------------------------- step fingerprint


class _PluginStep(Step):
    name = "attachments"
    config_sections = ("attachments",)
    ident: dict

    def run(self, ctx: StepContext) -> None: ...

    def fingerprint_extra(self, config):
        return self.ident


def test_plugin_version_changes_step_fingerprint():
    cfg = PipelineConfig()
    a, b, plain = _PluginStep(), _PluginStep(), _PluginStep()
    a.ident = {"plugin": "attachments/demo", "version": "1"}
    b.ident = {"plugin": "attachments/demo", "version": "2"}
    plain.ident = {}
    assert step_fingerprint(a, cfg, "up") != step_fingerprint(b, cfg, "up")
    assert step_fingerprint(plain, cfg, "up") != step_fingerprint(a, cfg, "up")
