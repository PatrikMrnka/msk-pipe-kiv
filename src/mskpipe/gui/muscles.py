# SPDX-License-Identifier: Apache-2.0
"""Check-box list of the exported muscles (GUI view of ``export.exclude``).

The GUI keeps ``export.muscles = all`` and shows one list: checked muscles are exported,
unchecked ones are written to ``export.exclude``. Muscles are offered by base name
(without side), in the order of the unified label scheme.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["MusclePicker", "base_name", "muscle_names", "pretty"]

_SIDE = re.compile(r"_[lr]$")


def base_name(name: str) -> str:
    return _SIDE.sub("", name)


def pretty(base: str) -> str:
    text = base.replace("_", " ")
    return text[:1].upper() + text[1:]


def muscle_names() -> list[str]:
    """Muscles of the unified label scheme, without side, in scheme order."""
    from mskpipe.labelmap.scheme import load_scheme

    names: list[str] = []
    for name, structure in load_scheme().structures.items():
        if structure.kind == "muscle" and base_name(name) not in names:
            names.append(base_name(name))
    return names


class _GridList(QListWidget):
    """Wrapping list whose cells fit the longest name and whose height fits its rows."""

    def fit(self) -> None:
        metrics = self.fontMetrics()
        texts = [self.item(i).text() for i in range(self.count())] or [""]
        size = QSize(
            max(metrics.horizontalAdvance(text) for text in texts) + 40,  # + check box
            metrics.height() + 8,
        )
        for i in range(self.count()):
            self.item(i).setSizeHint(size)
        self.setGridSize(size)
        self._fit_height()

    def _fit_height(self) -> None:
        grid = self.gridSize()
        if grid.width() <= 0:
            return
        columns = max(1, self.viewport().width() // grid.width())
        rows = max(1, math.ceil(self.count() / columns))
        self.setFixedHeight(rows * grid.height() + 2 * self.frameWidth() + 4)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_height()


class MusclePicker(QWidget):
    """Grid of muscles with check boxes and All / None / Defaults buttons.

    ``reasons``: why a muscle is unchecked by default (tooltip); ``defaults``: muscles
    checked by the Defaults button.
    """

    changed = Signal()

    def __init__(
        self,
        names: Sequence[str],
        *,
        reasons: Mapping[str, str] | None = None,
        defaults: Sequence[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._reasons = dict(reasons or {})
        self._defaults = list(defaults) if defaults is not None else None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list = _GridList()
        self.list.setFlow(QListView.Flow.LeftToRight)
        self.list.setWrapping(True)
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setUniformItemSizes(True)
        self.list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.list.itemChanged.connect(lambda _item: self.changed.emit())
        layout.addWidget(self.list)
        for name in names:
            self._add(name)
        self.list.fit()

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        for text, action in (("All", self._check_all), ("None", self._check_none)):
            button = QPushButton(text)
            button.clicked.connect(action)
            buttons.addWidget(button)
        if self._defaults is not None:
            button = QPushButton("Defaults")
            button.clicked.connect(lambda: self.set_checked(self._defaults or []))
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ items

    def _add(self, name: str) -> QListWidgetItem:
        item = QListWidgetItem(pretty(name))
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setCheckState(Qt.CheckState.Unchecked)
        tip = name
        if name in self._reasons:
            tip += f"\n\nNot exported by default: {self._reasons[name]}"
        item.setToolTip(tip)
        self.list.addItem(item)
        return item

    def _items(self) -> list[QListWidgetItem]:
        return [self.list.item(i) for i in range(self.list.count())]

    def names(self) -> list[str]:
        return [item.data(Qt.ItemDataRole.UserRole) for item in self._items()]

    def ensure(self, names: Iterable[str]) -> None:
        """Add muscles the scheme does not offer (values from a config file are kept)."""
        known = set(self.names())
        added = [name for name in names if name not in known]
        self.list.blockSignals(True)
        try:
            for name in added:
                self._add(name)
        finally:
            self.list.blockSignals(False)
        if added:
            self.list.fit()

    # ------------------------------------------------------------------ values

    def checked(self) -> list[str]:
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._items()
            if item.checkState() == Qt.CheckState.Checked
        ]

    def set_checked(self, names: Iterable[str]) -> None:
        wanted = list(names)
        self.ensure(wanted)
        self.list.blockSignals(True)
        try:
            for item in self._items():
                state = item.data(Qt.ItemDataRole.UserRole) in wanted
                item.setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)
        finally:
            self.list.blockSignals(False)
        self.changed.emit()

    def _check_all(self) -> None:
        self.set_checked(self.names())

    def _check_none(self) -> None:
        self.set_checked([])
