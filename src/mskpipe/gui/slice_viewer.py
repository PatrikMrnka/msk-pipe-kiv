# SPDX-License-Identifier: Apache-2.0
"""Axial, coronal and sagittal slices of the input volume (GUI, read only).

The volume is put into the closest canonical RAS+ voxel order by flips and axis swaps only
(numpy views, no resampling and no copy of the data), so every view shows exact voxel
slices; an oblique acquisition is shown along its voxel axes. The views follow the
radiological convention: patient right on the left of the screen, anterior on the left
of the sagittal view. Plane colours as in 3D Slicer: axial red, coronal green, sagittal
yellow; each crosshair line has the colour of the plane it marks.

Mouse: left button = move the crosshair, wheel = next/previous slice, right drag =
window (horizontal) and level (vertical), double click = this view alone / all views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

__all__ = [
    "VIEWS",
    "SlicePane",
    "SliceViewer",
    "Volume",
    "load_volume",
    "plane",
    "to_gray",
]

VIEWS = ("axial", "coronal", "sagittal")
# canonical (RAS+) voxel axes of each view: (slice axis, screen horizontal, screen vertical);
# both screen axes run against the voxel axis (patient right / anterior / superior first)
AXES = {"axial": (2, 0, 1), "coronal": (1, 0, 2), "sagittal": (0, 1, 2)}
LETTERS = {  # left, right, top, bottom edge of the view
    "axial": ("R", "L", "A", "P"),
    "coronal": ("R", "L", "S", "I"),
    "sagittal": ("A", "P", "S", "I"),
}
COLORS = {"axial": "#e74c3c", "coronal": "#2ecc71", "sagittal": "#f1c40f"}
VIEW_OF_AXIS = {AXES[v][0]: v for v in VIEWS}  # slice axis -> view

PRESETS = (  # label, (level, width) in the units of the volume, or a rule
    ("Auto (0.5-99.5 %)", "auto"),
    ("Full range", "full"),
    ("CT soft tissue (L 40, W 400)", (40.0, 400.0)),
    ("CT bone (L 400, W 1800)", (400.0, 1800.0)),
    ("Custom", "custom"),
)
_SAMPLE_VOXELS = 2_000_000  # voxels used for the automatic window


# --------------------------------------------------------------------------- volume


@dataclass(frozen=True, eq=False)
class Volume:
    """Volume in canonical RAS+ voxel order; ``data`` holds the stored (unscaled) values."""

    path: Path
    data: np.ndarray
    slope: float
    inter: float
    affine: np.ndarray  # canonical voxel -> world RAS [mm]
    file_affine: np.ndarray  # voxel of the file -> world RAS [mm]
    file_shape: tuple[int, ...]
    spacing: tuple[float, float, float]  # [mm] along the canonical axes
    auto_window: tuple[float, float]  # (low, high), 0.5 and 99.5 percentile
    value_range: tuple[float, float]  # of the sampled voxels
    oblique: bool
    notes: tuple[str, ...] = field(default=())

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.data.shape)  # type: ignore[return-value]

    def value(self, ijk) -> float:
        return float(self.data[tuple(int(c) for c in ijk)]) * self.slope + self.inter

    def world(self, ijk) -> np.ndarray:
        return (self.affine @ np.r_[np.asarray(ijk, float), 1.0])[:3]

    def file_index(self, ijk) -> tuple[int, int, int]:
        """Voxel index in the file (as other viewers show it) of a canonical index."""
        idx = np.linalg.solve(self.file_affine, np.r_[self.world(ijk), 1.0])[:3]
        return tuple(int(np.rint(c)) for c in idx)  # type: ignore[return-value]


def load_volume(path: str | Path) -> Volume:
    """Read a NIfTI volume for viewing (first volume of a 4D series)."""
    import nibabel as nib
    from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation

    path = Path(path)
    img = nib.load(str(path))
    proxy = img.dataobj
    notes: list[str] = []
    if nib.is_proxy(proxy):
        slope = _finite(getattr(proxy, "slope", 1.0), 1.0) or 1.0
        inter = _finite(getattr(proxy, "inter", 0.0), 0.0)
        raw = np.asanyarray(proxy.get_unscaled())  # stored dtype, e.g. int16
    else:
        slope, inter = 1.0, 0.0
        raw = np.asanyarray(proxy)
    if raw.ndim > 3:
        extra = int(np.prod(raw.shape[3:]))
        raw = raw[(slice(None),) * 3 + (0,) * (raw.ndim - 3)]
        if extra > 1:
            notes.append(f"{extra} volumes, the first is shown")
    while raw.ndim < 3:
        raw = raw[..., None]
    if raw.dtype == bool:
        raw = raw.view(np.uint8)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"unsupported voxel type {raw.dtype}")

    affine = np.asarray(img.affine, float)
    ornt = io_orientation(affine)
    data = apply_orientation(raw, ornt)  # flips and transposes: views
    canonical = affine @ inv_ornt_aff(ornt, raw.shape)
    norms = np.linalg.norm(canonical[:3, :3], axis=0)
    spacing = tuple(float(n) if n > 0 else 1.0 for n in norms)
    directions = canonical[:3, :3] / np.where(norms > 0, norms, 1.0)
    oblique = not np.allclose(np.abs(directions), np.eye(3), atol=1e-3)
    if oblique:
        notes.append("oblique volume, shown along its voxel axes")

    step = max(1, math.ceil((data.size / _SAMPLE_VOXELS) ** (1 / 3)))
    sample = data[::step, ::step, ::step].astype(np.float32).ravel() * slope + inter
    sample = sample[np.isfinite(sample)]
    if sample.size:
        low, high = (float(v) for v in np.percentile(sample, [0.5, 99.5]))
        vmin, vmax = float(sample.min()), float(sample.max())
    else:
        low, high, vmin, vmax = 0.0, 1.0, 0.0, 1.0
    if high <= low:
        low, high = vmin, max(vmax, vmin + 1.0)
    return Volume(
        path=path,
        data=data,
        slope=float(slope),
        inter=float(inter),
        affine=canonical,
        file_affine=affine,
        file_shape=tuple(int(n) for n in img.shape),
        spacing=spacing,  # type: ignore[arg-type]
        auto_window=(low, high),
        value_range=(vmin, vmax),
        oblique=oblique,
        notes=tuple(notes),
    )


def _finite(value, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


# --------------------------------------------------------------------------- slices


def plane(data: np.ndarray, view: str, index: int) -> np.ndarray:
    """Slice ``index`` of a canonical volume as shown on the screen (rows top to bottom)."""
    take: list[int | slice] = [slice(None)] * 3
    take[AXES[view][0]] = index
    return data[tuple(take)].T[::-1, ::-1]


def to_gray(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Window ``low``..``high`` mapped to 0..255 (uint8, C-contiguous)."""
    scale = 255.0 / max(high - low, 1e-12)
    out = (np.asarray(values, np.float32) - np.float32(low)) * np.float32(scale)
    np.nan_to_num(out, copy=False, nan=0.0)
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def _qimage(gray: np.ndarray) -> QImage:
    h, w = gray.shape
    return QImage(gray.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()


# --------------------------------------------------------------------------- one view


class SlicePane(QWidget):
    """One view: the slice in physical proportions, the crosshair and the orientation."""

    picked = Signal(str, float, float)  # view, column, row (image pixels)
    scrolled = Signal(str, int)  # view, slices (+ = towards right / anterior / superior)
    window_dragged = Signal(int, int)  # dx, dy [screen pixels]
    toggled = Signal(str)  # double click

    def __init__(self, view: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = view
        self.setMinimumSize(180, 180)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._image: QImage | None = None
        self._pixel = (1.0, 1.0)  # mm per image pixel (horizontal, vertical)
        self._cross: tuple[float, float] | None = None
        self._caption = ""
        self._message = ""
        self._drag: QPointF | None = None
        self._wheel = 0

    # ------------------------------------------------------------------ content

    def set_slice(
        self, gray: np.ndarray, pixel: tuple[float, float], cross: tuple[int, int], caption: str
    ) -> None:
        self._image, self._pixel = _qimage(gray), pixel
        self._cross, self._caption, self._message = cross, caption, ""
        self.update()

    def set_message(self, text: str) -> None:
        self._image, self._cross, self._message = None, None, text
        self.update()

    def image_rect(self) -> QRectF:
        """Where the slice is drawn: centred, physical proportions, as large as fits."""
        if self._image is None:
            return QRectF()
        w = self._image.width() * self._pixel[0]
        h = self._image.height() * self._pixel[1]
        scale = min(self.width() / w, self.height() / h)
        rect = QRectF(0, 0, w * scale, h * scale)
        rect.moveCenter(QRectF(self.rect()).center())
        return rect

    def to_image(self, pos: QPointF) -> tuple[float, float] | None:
        rect = self.image_rect()
        if rect.isEmpty():
            return None
        col = (pos.x() - rect.left()) / rect.width() * self._image.width()  # type: ignore[union-attr]
        row = (pos.y() - rect.top()) / rect.height() * self._image.height()  # type: ignore[union-attr]
        return col, row

    # ------------------------------------------------------------------ painting

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#000000"))
        color = QColor(COLORS[self.view])
        if self._image is None:
            p.setPen(QColor("#bdc3c7"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
        else:
            rect = self.image_rect()
            p.drawImage(rect, self._image)  # nearest neighbour: voxels stay visible
            if self._cross is not None:
                col, row = self._cross
                x = rect.left() + (col + 0.5) * rect.width() / self._image.width()
                y = rect.top() + (row + 0.5) * rect.height() / self._image.height()
                _, h_axis, v_axis = AXES[self.view]
                p.setPen(QPen(QColor(COLORS[VIEW_OF_AXIS[h_axis]]), 1))
                p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                p.setPen(QPen(QColor(COLORS[VIEW_OF_AXIS[v_axis]]), 1))
                p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            font = QFont(p.font())
            font.setBold(True)
            p.setFont(font)
            A = Qt.AlignmentFlag
            left, right, top, bottom = LETTERS[self.view]
            r = QRectF(self.rect().adjusted(6, 4, -6, -4))
            light = QColor("#ecf0f1")
            _text(p, r, A.AlignLeft | A.AlignVCenter, left, light)
            _text(p, r, A.AlignRight | A.AlignVCenter, right, light)
            _text(p, r, A.AlignHCenter | A.AlignTop, top, light)
            _text(p, r, A.AlignHCenter | A.AlignBottom, bottom, light)
        caption = self.view.capitalize() + (f"  {self._caption}" if self._caption else "")
        top_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        _text(p, QRectF(self.rect().adjusted(6, 4, -6, -4)), top_left, caption, color)
        p.setPen(QPen(color, 2))
        p.drawRect(self.rect().adjusted(1, 1, -1, -1))
        p.end()

    # ------------------------------------------------------------------ mouse

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pick(event.position())
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag = event.position()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._pick(event.position())
        elif event.buttons() & Qt.MouseButton.RightButton and self._drag is not None:
            delta = event.position() - self._drag
            self._drag = event.position()
            self.window_dragged.emit(round(delta.x()), round(delta.y()))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._drag = None
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggled.emit(self.view)
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._wheel += event.angleDelta().y()
        steps = int(self._wheel / 120)  # touchpads send fractions of a notch
        if steps:
            self._wheel -= steps * 120
            self.scrolled.emit(self.view, steps)
        event.accept()

    def _pick(self, pos: QPointF) -> None:
        hit = self.to_image(pos)
        if hit is not None:
            self.picked.emit(self.view, *hit)


# --------------------------------------------------------------------------- window


class _LoadSignals(QObject):
    done = Signal(int, object)  # (request number, Volume | error message)


class _LoadTask(QRunnable):
    def __init__(self, number: int, path: Path, signals: _LoadSignals) -> None:
        super().__init__()
        self.number, self.path, self.signals = number, path, signals

    def run(self) -> None:
        try:
            result: object = load_volume(self.path)
        except MemoryError:
            result = "not enough memory to load the volume"
        except Exception as exc:  # any unreadable file
            result = str(exc) or type(exc).__name__
        self.signals.done.emit(self.number, result)


class SliceViewer(QWidget):
    """Separate window with the three views of one volume.

    ``pool``: thread pool for reading the volume; the main window passes its own, so it
    waits for a read in progress when it closes.
    """

    loaded = Signal()  # a volume (or an error) is shown

    def __init__(self, parent: QWidget | None = None, *, pool: QThreadPool | None = None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Image viewer")
        self.resize(1280, 560)
        self.volume: Volume | None = None
        self.error: str | None = None
        self.cursor = [0, 0, 0]  # canonical voxel index
        self.maximized: str | None = None
        self._pool = pool or QThreadPool.globalInstance()
        self._signals = _LoadSignals(self)
        self._signals.done.connect(self._loaded)
        self._request = 0
        self._source: tuple[Path, tuple[int, int]] | None = None
        self._loading = False

        self.file_label = QLabel()
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preset_combo = QComboBox()
        for label, value in PRESETS:
            self.preset_combo.addItem(label, value)
        self.preset_combo.currentIndexChanged.connect(lambda _=None: self.apply_preset())
        self.level_spin = _spin("Level (window centre)")
        self.width_spin = _spin("Window width")
        self.level_spin.valueChanged.connect(lambda _=None: self._window_edited())
        self.width_spin.valueChanged.connect(lambda _=None: self._window_edited())
        self.status_label = QLabel()
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hint = QLabel(
            "Left button: crosshair · wheel: slice · right drag: window/level · "
            "double click: one view / all"
        )
        hint.setStyleSheet("color: #7f8c8d")

        top = QHBoxLayout()
        top.addWidget(self.file_label, 1)
        top.addWidget(QLabel("Window"))
        top.addWidget(self.preset_combo)
        top.addWidget(QLabel("L"))
        top.addWidget(self.level_spin)
        top.addWidget(QLabel("W"))
        top.addWidget(self.width_spin)

        self.panes: dict[str, SlicePane] = {}
        self.sliders: dict[str, QSlider] = {}
        self._columns: dict[str, QWidget] = {}
        views = QHBoxLayout()
        views.setSpacing(4)
        for view in VIEWS:
            pane = SlicePane(view)
            pane.picked.connect(self.pick)
            pane.scrolled.connect(self.scroll)
            pane.window_dragged.connect(self.drag_window)
            pane.toggled.connect(self.toggle)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.valueChanged.connect(lambda value, v=view: self.set_slice(v, value))
            column = QWidget()
            col = QVBoxLayout(column)
            col.setContentsMargins(0, 0, 0, 0)
            col.addWidget(pane, 1)
            col.addWidget(slider)
            views.addWidget(column, 1)
            self.panes[view], self.sliders[view], self._columns[view] = pane, slider, column

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(views, 1)
        layout.addWidget(self.status_label)
        layout.addWidget(hint)
        self._enable(False)

    # ------------------------------------------------------------------ loading

    def open(self, path: str | Path) -> None:
        """Show ``path``; reads it in the background unless it is already shown."""
        path = Path(path).resolve()
        stat = path.stat()
        source = (path, (stat.st_mtime_ns, stat.st_size))
        if source == self._source and (self._loading or self.volume is not None):
            return
        self._source = source
        self._request += 1
        self._loading = True
        self.volume, self.error = None, None
        self.setWindowTitle(f"{path.name} - image viewer")
        self.file_label.setText(path.name)
        self.status_label.setText("")
        for pane in self.panes.values():
            pane.set_message("Loading…")
        self._enable(False)
        self._pool.start(_LoadTask(self._request, path, self._signals))

    def is_loading(self) -> bool:
        return self._loading

    def _loaded(self, request: int, result: object) -> None:
        if request != self._request:  # another file was opened meanwhile, or closed
            return
        self._loading = False
        if not isinstance(result, Volume):
            self.error = str(result)
            for pane in self.panes.values():
                pane.set_message(f"Cannot read the volume:\n{self.error}")
            self.loaded.emit()
            return
        vol = self.volume = result
        self.cursor = [n // 2 for n in vol.shape]
        span = vol.value_range[1] - vol.value_range[0]
        decimals = 1 if span >= 100 else 4
        for spin in (self.level_spin, self.width_spin):
            spin.blockSignals(True)
            spin.setDecimals(decimals)
            spin.blockSignals(False)
        for view in VIEWS:
            slider = self.sliders[view]
            slider.blockSignals(True)
            slider.setRange(0, vol.shape[AXES[view][0]] - 1)
            slider.blockSignals(False)
        nx, ny, nz = vol.file_shape[:3]
        sx, sy, sz = (float(v) for v in np.linalg.norm(vol.file_affine[:3, :3], axis=0))
        self.file_label.setText(
            f"<b>{vol.path.name}</b> · {nx} x {ny} x {nz} voxels · "
            f"{sx:.3g} x {sy:.3g} x {sz:.3g} mm"
        )
        self._enable(True)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self.apply_preset()
        self.loaded.emit()

    def _enable(self, on: bool) -> None:
        for w in (self.preset_combo, self.level_spin, self.width_spin, *self.sliders.values()):
            w.setEnabled(on)

    # ------------------------------------------------------------------ window / level

    def window(self) -> tuple[float, float]:
        """(low, high) of the grey scale in the units of the volume."""
        level, width = self.level_spin.value(), self.width_spin.value()
        return level - width / 2, level + width / 2

    def apply_preset(self) -> None:
        vol, value = self.volume, self.preset_combo.currentData()
        if vol is None or value == "custom":
            return
        if value == "auto":
            low, high = vol.auto_window
        elif value == "full":
            low, high = vol.value_range
        else:
            level, width = value
            low, high = level - width / 2, level + width / 2
        self._set_window((low + high) / 2, max(high - low, self._min_width()))
        self.refresh()

    def drag_window(self, dx: int, dy: int) -> None:
        vol = self.volume
        if vol is None:
            return
        k = max(vol.value_range[1] - vol.value_range[0], 1e-6) / 500  # full range in 500 px
        width = max(self.width_spin.value() + dx * k, self._min_width())
        self._set_window(self.level_spin.value() - dy * k, width)
        self._custom()
        self.refresh()

    def _window_edited(self) -> None:
        self._custom()
        self.refresh()

    def _set_window(self, level: float, width: float) -> None:
        for spin, value in ((self.level_spin, level), (self.width_spin, width)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _custom(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(self.preset_combo.findData("custom"))
        self.preset_combo.blockSignals(False)

    def _min_width(self) -> float:
        return 10.0 ** -self.width_spin.decimals()

    # ------------------------------------------------------------------ navigation

    def pick(self, view: str, col: float, row: float) -> None:
        """Move the crosshair to an image pixel of ``view``."""
        if self.volume is None:
            return
        _, h_axis, v_axis = AXES[view]
        shape = self.volume.shape
        self.cursor[h_axis] = _clamp(shape[h_axis] - 1 - math.floor(col), shape[h_axis])
        self.cursor[v_axis] = _clamp(shape[v_axis] - 1 - math.floor(row), shape[v_axis])
        self.refresh()

    def scroll(self, view: str, steps: int) -> None:
        axis = AXES[view][0]
        self.set_slice(view, self.cursor[axis] + steps)

    def set_slice(self, view: str, index: int) -> None:
        if self.volume is None:
            return
        axis = AXES[view][0]
        self.cursor[axis] = _clamp(index, self.volume.shape[axis])
        self.refresh()

    def toggle(self, view: str) -> None:
        """Show ``view`` alone, or all views again."""
        self.maximized = None if self.maximized == view else view
        for name, column in self._columns.items():
            column.setVisible(self.maximized in (None, name))

    def refresh(self) -> None:
        vol = self.volume
        if vol is None:
            return
        low, high = self.window()
        for view in VIEWS:
            s_axis, h_axis, v_axis = AXES[view]
            index = self.cursor[s_axis]
            values = plane(vol.data, view, index).astype(np.float32) * vol.slope + vol.inter
            cross = (
                vol.shape[h_axis] - 1 - self.cursor[h_axis],
                vol.shape[v_axis] - 1 - self.cursor[v_axis],
            )
            self.panes[view].set_slice(
                to_gray(values, low, high),
                (vol.spacing[h_axis], vol.spacing[v_axis]),
                cross,
                f"{index + 1} / {vol.shape[s_axis]}",
            )
            slider = self.sliders[view]
            slider.blockSignals(True)
            slider.setValue(index)
            slider.blockSignals(False)
        x, y, z = vol.world(self.cursor)
        i, j, k = vol.file_index(self.cursor)
        parts = [
            f"Voxel [{i}, {j}, {k}]",
            f"RAS {x:.1f}, {y:.1f}, {z:.1f} mm",
            f"value {vol.value(self.cursor):.6g}",
            *vol.notes,
        ]
        self.status_label.setText("  ·  ".join(parts))

    # ------------------------------------------------------------------ closing

    def closeEvent(self, event) -> None:
        self._request += 1  # a read still in progress is dropped
        self._loading = False
        self.volume, self._source = None, None  # a CT can take gigabytes
        for pane in self.panes.values():
            pane.set_message("")
        super().closeEvent(event)


def _text(p: QPainter, rect: QRectF, flags, text: str, color: QColor) -> None:
    """Text with a dark shadow, readable over bright bone too."""
    p.setPen(QColor(0, 0, 0, 200))
    p.drawText(rect.translated(1, 1), flags, text)
    p.setPen(color)
    p.drawText(rect, flags, text)


def _spin(tooltip: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-1e9, 1e9)
    spin.setKeyboardTracking(False)
    spin.setToolTip(tooltip)
    spin.setMinimumWidth(90)
    return spin


def _clamp(index: int, n: int) -> int:
    return max(0, min(int(index), n - 1))
