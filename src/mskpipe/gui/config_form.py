# SPDX-License-Identifier: Apache-2.0
"""Configuration form built from the JSON Schema of :class:`~mskpipe.config.PipelineConfig`.

One tab per config section; nested models become group boxes. Widgets by schema type:
boolean -> check box, enum -> combo box, integer/number -> spin box (nullable: text),
string -> line edit (folders/executables with a browse button), anything else (lists,
unions such as ``int | "auto"``) -> one-line YAML. ``attachments.params`` is built from
the ``Params`` schema of the selected attachment plugin.

The form is the source of truth for a run: :meth:`ConfigForm.overrides` returns the
values that differ from the defaults as ``key.path=value`` strings for ``mskpipe run``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mskpipe import api
from mskpipe.config.help import Help, help_for, plugin_help
from mskpipe.core.registry import PluginError, Registry, default_registry
from mskpipe.gui.help_badge import HelpBadge, help_html, label_with_badge
from mskpipe.gui.muscles import MusclePicker, base_name, muscle_names

__all__ = ["HIDDEN", "ConfigForm", "flatten", "humanize", "nest", "yaml_value"]

Path_ = tuple[str, ...]
# set by the main window's own controls
HIDDEN: frozenset[Path_] = frozenset({("runtime", "device"), ("runtime", "runs_dir")})
PLUGIN_PARAMS: dict[Path_, tuple[Path_, str]] = {
    ("attachments", "params"): (("attachments", "method"), "attachments"),
}
_PATH_SUFFIXES = ("_dir", "_exe", "_file", "_path")
MUSCLES_PATH: Path_ = ("export", "muscles")
EXCLUDE_PATH: Path_ = ("export", "exclude")
LABELS: dict[Path_, str] = {EXCLUDE_PATH: "Exported muscles"}

_UNITS = {"mm": "mm", "s": "s", "deg": "°"}
_ERROR_LINE = re.compile(r"^\s+(\S+): (.+)$")


# ---------------------------------------------------------------------------- helpers


def humanize(name: str) -> str:
    """``repair_max_opening_mm`` -> ``Repair max opening [mm]``."""
    parts = name.split("_")
    unit = _UNITS.get(parts[-1]) if len(parts) > 1 else None
    if unit:
        parts = parts[:-1]
    text = " ".join(parts)
    text = text[:1].upper() + text[1:]
    return f"{text} [{unit}]" if unit else text


def yaml_value(value: Any) -> str:
    """One-line YAML of a value (``--set`` and YAML fields)."""
    return (
        yaml.safe_dump(value, default_flow_style=True, allow_unicode=True)
        .strip()
        .removesuffix("...")
        .strip()
    )


def flatten(data: Mapping[str, Any], prefix: Path_ = ()) -> dict[Path_, Any]:
    out: dict[Path_, Any] = {}
    for key, value in data.items():
        path = (*prefix, key)
        if isinstance(value, Mapping) and value:
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def nest(flat: Mapping[Path_, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path, value in flat.items():
        node = out
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return out


def _resolve(node: Mapping[str, Any], defs: Mapping[str, Any]) -> dict[str, Any]:
    node = dict(node)
    while "$ref" in node:
        target = defs[node.pop("$ref").rsplit("/", 1)[-1]]
        node = {**target, **node}
    return node


def _kind(node: Mapping[str, Any], defs: Mapping[str, Any]) -> tuple[str, dict[str, Any], bool]:
    """(kind, resolved node, nullable); kind in group|dict|bool|choice|int|float|str|yaml|const."""
    node = _resolve(node, defs)
    if "anyOf" in node:
        options = [_resolve(o, defs) for o in node["anyOf"]]
        rest = [o for o in options if o.get("type") != "null"]
        nullable = len(rest) < len(options)
        if len(rest) == 1:
            kind, inner, _ = _kind({**rest[0], **_meta(node)}, defs)
            return kind, inner, nullable
        return "yaml", node, nullable
    if "const" in node:
        return "const", node, False
    if "enum" in node:
        return "choice", node, False
    kind = node.get("type")
    if kind == "object":
        return ("group" if "properties" in node else "dict"), node, False
    return (
        {"boolean": "bool", "integer": "int", "number": "float", "string": "str"}.get(
            kind or "", "yaml"
        ),
        node,
        False,
    )


def _meta(node: Mapping[str, Any]) -> dict[str, Any]:
    return {k: node[k] for k in ("description", "title", "default") if k in node}


# ---------------------------------------------------------------------------- widgets


class _NoWheelSpin(QSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:  # scrolling the form, not the value
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoWheelDouble(QDoubleSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoWheelCombo(QComboBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class Field:
    """One editable value: ``widget`` for the layout, ``get``/``set`` for its value."""

    def __init__(
        self,
        path: Path_,
        widget: QWidget,
        get: Callable[[], Any],
        set_: Callable[[Any], None],
        connect: Callable[[Callable[[], None]], None],
        editor: QWidget | None = None,
    ) -> None:
        self.path = path
        self.widget = widget
        self.get = get
        self.set = set_
        self.connect = connect
        self.editor = editor or widget  # widget to mark/focus
        self.label: QLabel | None = None


def _bool_field(path: Path_, node: dict[str, Any]) -> Field:
    box = QCheckBox()
    return Field(
        path,
        box,
        box.isChecked,
        lambda v: box.setChecked(bool(v)),
        lambda cb: box.toggled.connect(lambda _=None: cb()),
    )


def _choice_field(path: Path_, values: list[Any]) -> Field:
    combo = _NoWheelCombo()
    combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    for v in values:
        combo.addItem(str(v), v)

    def set_(value: Any) -> None:
        i = combo.findData(value)
        if i < 0:  # value from a file that is not offered (validation reports it)
            combo.addItem(str(value), value)
            i = combo.count() - 1
        combo.setCurrentIndex(i)

    return Field(
        path,
        combo,
        combo.currentData,
        set_,
        lambda cb: combo.currentIndexChanged.connect(lambda _=None: cb()),
    )


def _int_field(path: Path_, node: dict[str, Any]) -> Field:
    spin = _NoWheelSpin()
    spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    low = node.get("minimum", node.get("exclusiveMinimum", -(2**31) + 1))
    if "exclusiveMinimum" in node and "minimum" not in node:
        low += 1
    high = node.get("maximum", node.get("exclusiveMaximum", 2**31 - 1))
    if "exclusiveMaximum" in node and "maximum" not in node:
        high -= 1
    spin.setRange(int(low), int(high))
    return Field(
        path,
        spin,
        spin.value,
        lambda v: spin.setValue(int(v)),
        lambda cb: spin.valueChanged.connect(lambda _=None: cb()),
    )


def _float_field(path: Path_, node: dict[str, Any]) -> Field:
    spin = _NoWheelDouble()
    spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    decimals = 4
    eps = 10.0**-decimals
    spin.setDecimals(decimals)
    low = node.get("minimum", node.get("exclusiveMinimum", -1e9))
    if "exclusiveMinimum" in node and "minimum" not in node:
        low += eps
    high = node.get("maximum", node.get("exclusiveMaximum", 1e9))
    if "exclusiveMaximum" in node and "maximum" not in node:
        high -= eps
    spin.setRange(float(low), float(high))
    default = node.get("default")
    if isinstance(default, int | float) and default:
        spin.setSingleStep(min(1.0, 10.0 ** math.floor(math.log10(abs(default)))))
    else:
        spin.setSingleStep(0.1)
    return Field(
        path,
        spin,
        spin.value,
        lambda v: spin.setValue(float(v)),
        lambda cb: spin.valueChanged.connect(lambda _=None: cb()),
    )


def _text_field(
    path: Path_, *, nullable: bool, parse: Callable[[str], Any] | None = None, browse: str = ""
) -> Field:
    edit = QLineEdit()
    if nullable:
        edit.setPlaceholderText("default (empty)")

    def get() -> Any:
        text = edit.text().strip()
        if nullable and not text:
            return None
        if parse is None:
            return text
        try:
            return parse(text)
        except (ValueError, yaml.YAMLError):
            return text  # reported by validation

    def set_(value: Any) -> None:
        if value is None:
            edit.setText("")
        elif parse is None:
            edit.setText(str(value))
        else:
            edit.setText(yaml_value(value))

    widget: QWidget = edit
    if browse:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        button = QPushButton("…")
        button.setFixedWidth(32)
        button.setToolTip("Browse")

        def pick() -> None:
            start = edit.text() or str(Path.cwd())
            if browse == "dir":
                chosen = QFileDialog.getExistingDirectory(widget, "Select folder", start)
            else:
                chosen, _ = QFileDialog.getOpenFileName(widget, "Select file", start)
            if chosen:
                edit.setText(chosen)

        button.clicked.connect(pick)
        row.addWidget(button)
    return Field(
        path, widget, get, set_, lambda cb: edit.textChanged.connect(lambda _=None: cb()), edit
    )


def _parse_yaml(text: str) -> Any:
    return yaml.safe_load(text) if text else None


def _parse_number(kind: str) -> Callable[[str], Any]:
    def parse(text: str) -> Any:
        return int(text) if kind == "int" else float(text)

    return parse


# ---------------------------------------------------------------------------- form


class ConfigForm(QWidget):
    """Tabbed editor of a pipeline configuration."""

    changed = Signal()

    def __init__(self, registry: Registry | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._registry = registry or default_registry()
        schema = api.config_schema()
        self._defs = schema.get("$defs", {})
        self._defaults = api.resolved_config(registry=self._registry).model_dump(mode="json")
        self._fields: dict[Path_, Field] = {}
        self._plugin_boxes: dict[Path_, QGroupBox] = {}
        self._badges: dict[Path_, HelpBadge] = {}
        self._pickers: dict[Path_, MusclePicker] = {}
        self._building = False

        self.tabs = QTabWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        self._building = True
        for name, node in schema["properties"].items():
            kind, resolved, _ = _kind(node, self._defs)
            if kind != "group":
                continue
            page = QWidget()
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            self._add_group(form, (name,), resolved)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            index = self.tabs.addTab(scroll, humanize(name))
            self.tabs.setTabToolTip(index, self._help_tip((name,)))
        self._building = False
        self.set_values(self._defaults)

    # ------------------------------------------------------------------ building

    def _add_group(self, form: QFormLayout, path: Path_, node: Mapping[str, Any]) -> None:
        for name, child in node.get("properties", {}).items():
            child_path = (*path, name)
            if child_path in HIDDEN or child_path == MUSCLES_PATH:  # GUI: always 'all'
                continue
            kind, resolved, nullable = _kind(child, self._defs)
            if kind == "const":
                continue
            if kind == "group":
                box = QGroupBox(humanize(name))
                box.setToolTip(self._help_tip(child_path) or _tooltip(child_path, resolved))
                inner = QFormLayout(box)
                inner.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
                self._add_group(inner, child_path, resolved)
                form.addRow(box)
                continue
            if child_path in PLUGIN_PARAMS:
                box = QGroupBox(humanize(name))
                QFormLayout(box)
                self._plugin_boxes[child_path] = box
                form.addRow(box)
                continue
            field = self._make_field(child_path, kind, resolved, nullable)
            label = QLabel(LABELS.get(child_path, humanize(name)))
            tip = self._help_tip(child_path) or _tooltip(child_path, resolved)
            label.setToolTip(tip)
            field.editor.setToolTip(tip)
            field.label = label
            field.connect(lambda p=child_path: self._on_change(p))
            self._fields[child_path] = field
            form.addRow(label_with_badge(label, self._badge(child_path)), field.widget)

    def _make_field(self, path: Path_, kind: str, node: dict[str, Any], nullable: bool) -> Field:
        if path == EXCLUDE_PATH:
            return self._exported_muscles_field()
        plugin_kind = next((k for p, (m, k) in PLUGIN_PARAMS.items() if m == path), None)
        if plugin_kind is not None:
            return _choice_field(path, self._registry.names(plugin_kind))  # type: ignore[arg-type]
        if kind == "bool":
            return _bool_field(path, node)
        if kind == "choice":
            return _choice_field(path, list(node["enum"]))
        if kind in ("int", "float") and not nullable:
            return (_int_field if kind == "int" else _float_field)(path, node)
        if kind in ("int", "float"):
            return _text_field(path, nullable=True, parse=_parse_number(kind))
        if kind == "str":
            browse = ""
            if node.get("format") == "path" or path[-1].endswith(_PATH_SUFFIXES):
                browse = "dir" if path[-1].endswith(("_dir", "_path")) else "file"
            return _text_field(path, nullable=nullable, browse=browse)
        return _text_field(path, nullable=nullable, parse=_parse_yaml)

    def _exported_muscles_field(self) -> Field:
        """One check-box list for ``export.exclude``: checked = exported, unchecked = excluded
        (``export.muscles`` stays 'all' in the GUI)."""
        from mskpipe.steps.export_mw2 import EXCLUSION_REASONS

        names = muscle_names()
        excluded_by_default = list(self._defaults.get("export", {}).get("exclude", []))
        picker = MusclePicker(
            names,
            reasons=EXCLUSION_REASONS,
            defaults=[n for n in names if n not in excluded_by_default],
        )
        order: dict[str, int] = {}  # defaults first: the default value compares equal
        for name in [*excluded_by_default, *names]:
            order.setdefault(name, len(order))

        def get() -> Any:
            checked = set(picker.checked())
            excluded = [n for n in picker.names() if n not in checked]
            return sorted(excluded, key=lambda n: order.get(n, len(order)))

        def set_(value: Any) -> None:
            excluded = [str(v) for v in (value or [])]
            picker.ensure(excluded)
            picker.set_checked([n for n in picker.names() if n not in excluded])

        self._pickers[EXCLUDE_PATH] = picker
        return Field(EXCLUDE_PATH, picker, get, set_, lambda cb: picker.changed.connect(cb))

    def _help(self, path: Path_) -> Help | None:
        for params_path, (method_path, kind) in PLUGIN_PARAMS.items():
            n = len(params_path)
            if path[:n] == params_path and len(path) > n and method_path in self._fields:
                return plugin_help(kind, str(self._fields[method_path].get()), path[n])
        return help_for(path)

    def _help_tip(self, path: Path_) -> str:
        """Explanation of ``path`` as rich-text tooltip ('' if there is none)."""
        info = self._help(path)
        return help_html(info, ".".join(path)) if info is not None else ""

    def _badge(self, path: Path_) -> HelpBadge | None:
        tip = self._help_tip(path)
        if not tip:
            return None
        badge = HelpBadge(tip)
        self._badges[path] = badge
        return badge

    def help_text(self, *path: str) -> str:
        """Explanation shown by the '?' of a setting ('' if there is none)."""
        badge = self._badges.get(tuple(path))
        return badge.toolTip() if badge is not None else ""

    def picker(self, *path: str) -> MusclePicker:
        return self._pickers[tuple(path)]

    def _apply_muscle_list(self, given: Mapping[Path_, Any]) -> None:
        """A config with an explicit ``export.muscles`` list: muscles not in it are shown
        unchecked (added to ``export.exclude``), as the GUI keeps ``export.muscles = all``."""
        listed = given.get(MUSCLES_PATH)
        field = self._fields.get(EXCLUDE_PATH)
        if not isinstance(listed, list) or field is None:
            return
        exported = {base_name(str(m)) for m in listed}
        picker = self._pickers[EXCLUDE_PATH]
        excluded = set(field.get())  # before new names are added (unchecked) to the list
        picker.ensure(sorted(exported))
        excluded |= {n for n in picker.names() if n not in exported}
        field.set(sorted(excluded))

    def _build_plugin_params(self, params_path: Path_, values: Mapping[str, Any]) -> None:
        """(Re)build the parameter fields of the plugin selected for ``params_path``."""
        box = self._plugin_boxes[params_path]
        form = box.layout()
        assert isinstance(form, QFormLayout)
        for path in [p for p in self._fields if p[: len(params_path)] == params_path]:
            del self._fields[path]
        for path in [p for p in self._badges if p[: len(params_path)] == params_path]:
            del self._badges[path]
        while form.rowCount():
            form.removeRow(0)
        method_path, plugin_kind = PLUGIN_PARAMS[params_path]
        method = self._fields[method_path].get()
        try:
            plugin = self._registry.get(plugin_kind, method)  # type: ignore[arg-type]
        except PluginError as exc:
            form.addRow(QLabel(str(exc)))
            return
        schema = plugin.Params.model_json_schema()
        defs = {**self._defs, **schema.get("$defs", {})}
        box.setTitle(f"{humanize(params_path[-1])} ({method})")
        box.setToolTip(self._help_tip(params_path))
        saved, self._defs = self._defs, defs
        try:
            self._add_group(form, params_path, schema)
        finally:
            self._defs = saved
        defaults = self._plugin_defaults(params_path, method)
        for key, value in {**defaults, **values}.items():
            field = self._fields.get((*params_path, key))
            if field is not None:
                field.set(value)

    def _plugin_defaults(self, params_path: Path_, method: str) -> dict[str, Any]:
        _, plugin_kind = PLUGIN_PARAMS[params_path]
        try:
            plugin = self._registry.get(plugin_kind, method)  # type: ignore[arg-type]
            return plugin.Params().model_dump(mode="json")
        except (PluginError, ValueError):
            return {}

    # ------------------------------------------------------------------ values

    def values(self) -> dict[str, Any]:
        """Nested config values of all fields (hidden ones excluded)."""
        return nest({path: field.get() for path, field in self._fields.items()})

    def set_values(self, data: Mapping[str, Any]) -> None:
        """Set fields from a (possibly partial) nested config; missing keys = defaults."""
        given = flatten(data)
        merged = {**flatten(self._defaults), **given}
        self._building = True
        try:
            for path, field in list(self._fields.items()):
                if path in merged and path[:2] not in PLUGIN_PARAMS:
                    field.set(merged[path])
            for params_path in self._plugin_boxes:
                n = len(params_path)
                params = {  # only given values; the rest = defaults of the selected plugin
                    p[n]: v for p, v in given.items() if p[:n] == params_path and len(p) > n
                }
                self._build_plugin_params(params_path, params)
            self._apply_muscle_list(given)
        finally:
            self._building = False
        self._refresh_marks()
        self.changed.emit()

    def reset(self) -> None:
        self.set_values({})

    def reference(self) -> dict[Path_, Any]:
        """Flat defaults the current values are compared with (plugin params: defaults of
        the selected plugin)."""
        ref = {p: v for p, v in flatten(self._defaults).items() if p[:2] not in PLUGIN_PARAMS}
        for params_path, (method_path, _) in PLUGIN_PARAMS.items():
            method = self._fields[method_path].get()
            for key, value in self._plugin_defaults(params_path, method).items():
                ref[(*params_path, key)] = value
        return ref

    def changes(self) -> dict[Path_, Any]:
        """Fields whose value differs from the default, in form order."""
        ref = self.reference()
        return {
            path: field.get()
            for path, field in self._fields.items()
            if path not in ref or field.get() != ref[path]
        }

    def overrides(self) -> list[str]:
        """``key.path=value`` for ``mskpipe run --set``."""
        return [f"{'.'.join(p)}={yaml_value(v)}" for p, v in self.changes().items()]

    # ------------------------------------------------------------------ files

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a config file into the form; returns its full resolved values (the caller
        takes the hidden ``runtime.device``/``runtime.runs_dir`` from it).

        Raises :class:`mskpipe.api.SetupError`.
        """
        data = api.resolved_config(path, registry=self._registry).model_dump(mode="json")
        self.set_values(data)
        return data

    def save(self, path: str | Path, extra: Mapping[str, Any] | None = None) -> None:
        """Write the non-default values (plus ``extra``, nested) as a config file."""
        data = nest(self.changes())
        for key, value in flatten(extra or {}).items():
            node = data
            for part in key[:-1]:
                node = node.setdefault(part, {})
            node[key[-1]] = value
        header = (
            "# msk-pipe configuration written by mskpipe gui.\n"
            "# Only values that differ from the defaults; see `mskpipe config init`.\n"
        )
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True) if data else "{}\n"
        Path(path).write_text(header + body, encoding="utf-8")

    # ------------------------------------------------------------------ validation

    def validate(self, extra_overrides: list[str] | None = None) -> str | None:
        """Check the values with the config schema; marks bad fields, returns the error."""
        for field in self._fields.values():
            field.editor.setStyleSheet("")
        try:
            api.resolved_config(
                overrides=[*(extra_overrides or []), *self.overrides()], registry=self._registry
            )
        except api.SetupError as exc:
            message = str(exc)
            for line in message.splitlines():
                match = _ERROR_LINE.match(line)
                field = self._field_for(match.group(1)) if match else None
                if field is not None:
                    field.editor.setStyleSheet("border: 1px solid #c0392b;")
            self.error_label.setText(message)
            self.error_label.show()
            return message
        self.error_label.hide()
        return None

    def _field_for(self, location: str) -> Field | None:
        """Field of an error location; unions add parts (``a.b.list[str].0`` -> ``a.b``)."""
        parts = tuple(location.split("."))
        for n in range(len(parts), 0, -1):
            if parts[:n] in self._fields:
                return self._fields[parts[:n]]
        return None

    def field(self, *path: str) -> Field:
        return self._fields[tuple(path)]

    def tab_names(self) -> list[str]:
        return [self.tabs.tabText(i) for i in range(self.tabs.count())]

    # ------------------------------------------------------------------ internals

    def _on_change(self, path: Path_) -> None:
        if self._building:
            return
        for params_path, (method_path, _) in PLUGIN_PARAMS.items():
            if path == method_path:
                self._building = True
                try:
                    self._build_plugin_params(params_path, {})
                finally:
                    self._building = False
        self._refresh_marks()
        self.changed.emit()

    def _refresh_marks(self) -> None:
        ref = self.reference()
        for path, field in self._fields.items():
            if field.label is None:
                continue
            font = QFont(field.label.font())
            font.setBold(path not in ref or field.get() != ref[path])
            field.label.setFont(font)


def _tooltip(path: Path_, node: Mapping[str, Any]) -> str:
    text = node.get("description") or ""
    return f"{text}\n\n{'.'.join(path)}".strip()
