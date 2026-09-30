# SPDX-License-Identifier: Apache-2.0
"""Named plugins: built-ins plus third-party ones declared as entry points.

External packages register a plugin in their ``pyproject.toml``::

    [project.entry-points."mskpipe.attachments"]
    my_method = "my_pkg.module:MyMethod"

Plugins are imported lazily (on ``get``/``describe``); built-ins win on name clashes.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Any

from pydantic import ValidationError

from mskpipe.config import ConfigError, PipelineConfig
from mskpipe.plugins.base import KIND_BASES, KINDS, Plugin, PluginKind, PluginParams

logger = logging.getLogger("mskpipe")

ENTRY_POINT_GROUPS: dict[PluginKind, str] = {
    "segmenter": "mskpipe.segmenters",
    "skeleton": "mskpipe.skeleton",
    "attachments": "mskpipe.attachments",
}
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_P = "mskpipe.plugins"

BUILTINS: tuple[tuple[PluginKind, str, str], ...] = (
    ("segmenter", "totalsegmentator", f"{_P}.segmenters.totalsegmentator:TotalSegmentator"),
    ("segmenter", "musclemap", f"{_P}.segmenters.musclemap:MuscleMap"),
    ("skeleton", "pystaple", f"{_P}.skeleton.pystaple_backend:PyStaple"),
    ("attachments", "atlas_based", f"{_P}.attachments.atlas_based:AtlasBased"),
)


class PluginError(ConfigError):
    """Unknown, broken or misconfigured plugin; the message is meant for end users."""


@dataclass(frozen=True)
class PluginSpec:
    kind: PluginKind
    name: str
    target: str  # "module:Class"
    source: str = "mskpipe"  # distribution providing the plugin
    source_version: str | None = None


@dataclass(frozen=True)
class PluginInfo:
    """Listing entry (``mskpipe plugins``, GUI)."""

    kind: PluginKind
    name: str
    source: str
    version: str | None = None
    description: str = ""
    supports_gpu: bool = False
    missing: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)  # name -> default ("<required>")
    params_schema: dict[str, Any] = field(default_factory=dict)
    error: str | None = None  # the plugin could not be loaded

    @property
    def available(self) -> bool:
        return self.error is None and not self.missing


class Registry:
    def __init__(self, specs: Iterable[PluginSpec] = ()) -> None:
        self._specs: dict[tuple[str, str], PluginSpec] = {}
        self._classes: dict[tuple[str, str], type[Plugin]] = {}
        for spec in specs:
            self.add(spec)

    @classmethod
    def default(cls, *, external: bool = True) -> Registry:
        """Built-in plugins, plus entry-point plugins of installed packages."""
        reg = cls(PluginSpec(kind, name, target) for kind, name, target in BUILTINS)
        if external:
            reg._add_entry_points()
        return reg

    # ------------------------------------------------------------------ registration

    def add(self, spec: PluginSpec) -> None:
        if spec.kind not in KINDS:
            raise ValueError(f"Unknown plugin kind '{spec.kind}'; expected one of {KINDS}")
        if not _NAME.match(spec.name):
            raise ValueError(f"Invalid plugin name '{spec.name}' (use [a-z][a-z0-9_]*)")
        key = (spec.kind, spec.name)
        if key in self._specs:
            raise ValueError(f"Plugin {spec.kind}/{spec.name} is already registered")
        self._specs[key] = spec

    def register(self, plugin: type[Plugin], *, source: str = "local") -> type[Plugin]:
        """Register an already imported plugin class (tests, notebooks). Usable as decorator."""
        spec = PluginSpec(plugin.kind, plugin.name, _target(plugin), source)
        self.add(spec)
        self._classes[(spec.kind, spec.name)] = self._check(spec, plugin)
        return plugin

    def _add_entry_points(self) -> None:
        for kind, group in ENTRY_POINT_GROUPS.items():
            for ep in entry_points(group=group):
                dist = ep.dist
                spec = PluginSpec(
                    kind,
                    ep.name,
                    ep.value,
                    dist.name if dist is not None else "external",
                    dist.version if dist is not None else None,
                )
                try:
                    self.add(spec)
                except ValueError as exc:
                    logger.warning("Ignoring plugin entry point '%s' (%s): %s", ep.name, group, exc)

    # ------------------------------------------------------------------ lookup

    def names(self, kind: PluginKind) -> list[str]:
        return sorted(name for k, name in self._specs if k == kind)

    def spec(self, kind: PluginKind, name: str) -> PluginSpec:
        try:
            return self._specs[(kind, name)]
        except KeyError:
            known = ", ".join(self.names(kind)) or "none"
            raise PluginError(f"Unknown {kind} plugin '{name}'; available: {known}") from None

    def get(self, kind: PluginKind, name: str) -> type[Plugin]:
        """Plugin class; imports its module on first use."""
        key = (kind, name)
        if key not in self._classes:
            spec = self.spec(kind, name)
            self._classes[key] = self._check(spec, _load(spec))
        return self._classes[key]

    def create(self, kind: PluginKind, name: str) -> Plugin:
        """Plugin instance ready to run; fails early if its dependencies are missing."""
        plugin = self.get(kind, name)
        missing = plugin.missing()
        if missing:
            raise PluginError(
                f"{kind} plugin '{name}' needs {', '.join(missing)}, which is not installed"
            )
        return plugin()

    def validate_params(
        self, kind: PluginKind, name: str, params: Mapping[str, Any], *, where: str
    ) -> PluginParams:
        """Validate plugin parameters; errors use dotted config paths (``where.params.x``)."""
        plugin = self.get(kind, name)
        try:
            return plugin.Params.model_validate(dict(params))
        except ValidationError as exc:
            lines = [f"Invalid configuration (plugin {kind}/{name}):"]
            for err in exc.errors():
                loc = ".".join(str(p) for p in (where, "params", *err["loc"]))
                lines.append(f"  {loc}: {err['msg']}")
            raise PluginError("\n".join(lines)) from None

    def identity(self, kind: PluginKind, name: str) -> dict[str, str]:
        """What identifies a plugin's results (step fingerprints, manifest)."""
        spec = self.spec(kind, name)
        ident = {"plugin": f"{kind}/{name}", "version": self.get(kind, name).version}
        if spec.source != "mskpipe":
            ver = spec.source_version
            ident["source"] = f"{spec.source}=={ver}" if ver else spec.source
        return ident

    def describe(self, kind: PluginKind | None = None) -> list[PluginInfo]:
        infos = []
        for (k, name), spec in sorted(self._specs.items()):
            if kind is not None and k != kind:
                continue
            try:
                plugin = self.get(k, name)
            except PluginError as exc:
                infos.append(PluginInfo(k, name, spec.source, error=str(exc)))
                continue
            infos.append(
                PluginInfo(
                    kind=k,
                    name=name,
                    source=spec.source,
                    version=plugin.version,
                    description=plugin.description,
                    supports_gpu=plugin.supports_gpu,
                    missing=tuple(plugin.missing()),
                    params=_param_defaults(plugin.Params),
                    params_schema=plugin.Params.model_json_schema(),
                )
            )
        return infos

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _check(spec: PluginSpec, plugin: Any) -> type[Plugin]:
        base = KIND_BASES[spec.kind]
        where = f"{spec.kind} plugin '{spec.name}' ({spec.target})"
        if not (inspect.isclass(plugin) and issubclass(plugin, base)):
            raise PluginError(f"{where} is not a subclass of {base.__name__}")
        if inspect.isabstract(plugin):
            raise PluginError(f"{where} does not implement all abstract methods")
        if getattr(plugin, "name", None) != spec.name:
            raise PluginError(f"{where} declares name '{getattr(plugin, 'name', None)}'")
        if not (inspect.isclass(plugin.Params) and issubclass(plugin.Params, PluginParams)):
            raise PluginError(f"{where}: Params must subclass PluginParams")
        return plugin


@lru_cache(maxsize=1)
def default_registry() -> Registry:
    """Process-wide registry (built-ins + installed entry points)."""
    return Registry.default()


def resolve_plugins(config: PipelineConfig, registry: Registry | None = None) -> PipelineConfig:
    """Check plugin names in ``config`` and fill plugin parameters with their defaults.

    Call before a run so that ``config.resolved.yaml`` records every parameter used.
    """
    reg = registry or default_registry()
    reg.get("skeleton", config.skeleton.backend)
    att = config.attachments
    params = reg.validate_params("attachments", att.method, att.params, where="attachments")
    resolved = att.model_copy(update={"params": params.model_dump(mode="json")})
    return config.model_copy(update={"attachments": resolved})


def format_plugins(infos: Iterable[PluginInfo]) -> str:
    """Plain-text listing for the CLI."""
    lines: list[str] = []
    for info in infos:
        if info.error:
            status = "broken"
        elif info.missing:
            status = f"missing {', '.join(info.missing)}"
        else:
            status = "ok"
        gpu = " [gpu]" if info.supports_gpu else ""
        src = "" if info.source == "mskpipe" else f" <{info.source}>"
        lines.append(
            f"{info.kind:<12} {info.name:<18} v{info.version or '?'}{gpu}{src}  ({status})"
        )
        if info.description:
            lines.append(f"{'':<13}{info.description}")
        if info.params:
            params = ", ".join(f"{k}={v}" for k, v in info.params.items())
            lines.append(f"{'':<13}params: {params}")
        if info.error:
            lines.append(f"{'':<13}{info.error}")
    return "\n".join(lines)


def _load(spec: PluginSpec) -> Any:
    module_name, _, attr = spec.target.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
        for part in attr.split(".") if attr else ():
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        raise PluginError(
            f"Cannot load {spec.kind} plugin '{spec.name}' from '{spec.target}': {exc}"
        ) from exc
    return obj


def _target(plugin: type[Plugin]) -> str:
    return f"{plugin.__module__}:{plugin.__qualname__}"


def _param_defaults(model: type[PluginParams]) -> dict[str, Any]:
    return {
        name: "<required>" if f.is_required() else f.get_default(call_default_factory=True)
        for name, f in model.model_fields.items()
    }
