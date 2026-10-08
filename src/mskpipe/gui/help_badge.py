# SPDX-License-Identifier: Apache-2.0
"""Round '?' next to a setting; hovering (or clicking) it shows its explanation."""

from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QEnterEvent, QMouseEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolTip, QWidget

from mskpipe.config.help import Help

__all__ = ["HelpBadge", "help_html", "label_with_badge"]

BADGE_STYLE = (
    "QLabel { border: 1px solid #8a8a8a; border-radius: 8px; color: #555555;"
    " font-weight: bold; font-size: 10px; }"
    " QLabel:hover { background: #2471a3; border-color: #2471a3; color: white; }"
)


def help_html(info: Help, key: str | None = None) -> str:
    """Rich-text explanation (wrapped by Qt) with the example and the config key."""
    parts = [f"<p>{html.escape(info.text)}</p>"]
    if info.example:
        parts.append(f"<p><i>Example:</i> {html.escape(info.example)}</p>")
    if key:
        parts.append(f'<p style="color:#7f8c8d">{html.escape(key)}</p>')
    return "".join(parts)


class HelpBadge(QLabel):
    """Shows its tooltip at once on hover (no tooltip delay) and on click."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__("?", parent)
        self.setFixedSize(16, 16)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(BADGE_STYLE)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setToolTip(text)

    def show_help(self) -> None:
        QToolTip.showText(QCursor.pos(), self.toolTip(), self)

    def enterEvent(self, event: QEnterEvent) -> None:
        super().enterEvent(event)
        self.show_help()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.show_help()
        event.accept()


def label_with_badge(label: QLabel, badge: HelpBadge | None) -> QWidget:
    """``label`` followed by its '?' (form label column)."""
    if badge is None:
        return label
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(label)
    row.addWidget(badge)
    row.addStretch(1)
    return box
