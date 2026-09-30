"""Loading, overriding and rendering of the pipeline configuration.

Precedence: schema defaults < YAML file < ``key.path=value`` overrides.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ValidationError

from mskpipe.config.schema import PipelineConfig

__all__ = [
    "ConfigError",
    "deep_merge",
    "dump_config",
    "load_config",
    "parse_override",
    "render_template",
    "validate_config",
]


class ConfigError(ValueError):
    """Invalid configuration; the message is meant for end users."""


# --------------------------------------------------------------------------- loading


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> PipelineConfig:
    """Load a config file (optional) and apply ``key.path=value`` overrides."""
    data: dict[str, Any] = _read_yaml(Path(path)) if path is not None else {}
    for item in overrides:
        data = deep_merge(data, parse_override(item))
    return validate_config(data, source=str(path) if path is not None else "defaults")


def validate_config(data: Mapping[str, Any], source: str = "config") -> PipelineConfig:
    try:
        return PipelineConfig.model_validate(dict(data))
    except ValidationError as exc:
        raise ConfigError(_format_errors(exc, source)) from None


def parse_override(item: str) -> dict[str, Any]:
    """``'mesh.bones.smooth_iterations=10'`` -> nested dict; value parsed as YAML."""
    key, sep, raw = item.partition("=")
    parts = key.strip().split(".")
    if not sep or any(not p.strip() for p in parts):
        raise ConfigError(f"Invalid override '{item}'; expected key.path=value")
    try:
        value: Any = yaml.safe_load(raw) if raw.strip() else None
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid value in override '{item}': {exc}") from None
    for part in reversed(parts):
        value = {part.strip(): value}
    return value


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into a copy of ``base`` (lists are replaced)."""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config file '{path}': {exc.strerror}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in '{path}': {exc}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file '{path}' must contain a mapping at the top level")
    return data


def _format_errors(exc: ValidationError, source: str) -> str:
    lines = [f"Invalid configuration ({source}):"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- output


def dump_config(config: PipelineConfig, path: str | Path | None = None) -> str:
    """Serialise a resolved config to YAML; write it to ``path`` if given."""
    text = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def render_template(config: PipelineConfig | None = None) -> str:
    """Commented YAML with all keys, descriptions and allowed values."""
    lines = [
        "# msk-pipe configuration",
        "# Every key is optional; omitted keys use the defaults shown here.",
    ]
    _render(config or PipelineConfig(), 0, lines)
    return "\n".join(lines) + "\n"


def _render(model: BaseModel, level: int, out: list[str]) -> None:
    pad = "  " * level
    values = model.model_dump(mode="json")
    for name, field in type(model).model_fields.items():
        child = getattr(model, name)
        if level == 0 and isinstance(child, BaseModel):
            out.append("")
        doc = field.description or ""
        choices = _choices(field.annotation)
        if choices:
            doc = f"{doc} Options: {' | '.join(choices)}".strip()
        if doc:
            out.append(f"{pad}# {doc}")
        if isinstance(child, BaseModel):
            out.append(f"{pad}{name}:")
            _render(child, level + 1, out)
        else:
            dumped = yaml.safe_dump(values[name], default_flow_style=True, allow_unicode=True)
            out.append(f"{pad}{name}: {dumped.strip().removesuffix('...').strip()}")


def _choices(annotation: Any) -> list[str]:
    """Allowed values if the annotation is an Enum/Literal (optionally | None)."""
    args = _flatten_union(annotation)
    choices: list[str] = []
    for arg in args:
        if typing.get_origin(arg) is Literal:
            choices += [str(a) for a in typing.get_args(arg)]
        elif isinstance(arg, type) and issubclass(arg, Enum):
            choices += [str(m.value) for m in arg]
        else:
            return []
    return choices


def _flatten_union(annotation: Any) -> list[Any]:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return [a for a in typing.get_args(annotation) if a is not type(None)]
    return [annotation]
