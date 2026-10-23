# SPDX-License-Identifier: Apache-2.0
"""Axial, coronal and sagittal slices of the input volume with label maps (GUI, read only).

The volume is put into the closest canonical RAS+ voxel order by flips and axis swaps only
(numpy views, no resampling and no copy of the data), so every view shows exact voxel
slices; an oblique acquisition is shown along its voxel axes. The views follow the
radiological convention: patient right on the left of the screen, anterior on the left
of the sagittal view. Plane colours as in 3D Slicer: axial red, coronal green, sagittal
yellow; each crosshair line has the colour of the plane it marks.

A label map of the run (unified map, raw tool outputs, defect repairs; see
:mod:`mskpipe.gui.slice_labels`) or any integer NIfTI can be laid over the slices, filled
or as outlines, with every structure shown or hidden on its own.

Mouse: left button = move the crosshair, wheel = next/previous slice, right drag =
window (horizontal) and level (vertical), double click = this view alone / all views,
double click on a structure = crosshair on it. L = labels on/off.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mskpipe.gui import slice_labels as sl

__all__ = [
    "VIEWS",
    "SlicePane",
    "SliceViewer",
    "Volume",
    "load_volume",
    "main",
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


def _qimage(pixels: np.ndarray) -> QImage:
    """Grey (h, w) or RGB (h, w, 3) uint8 slice as a QImage (own copy of the data)."""
    h, w = pixels.shape[:2]
    if pixels.ndim == 3:
        return QImage(pixels.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QImage(pixels.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()


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
        self, pixels: np.ndarray, pixel: tuple[float, float], cross: tuple[int, int], caption: str
    ) -> None:
        self._image, self._pixel = _qimage(pixels), pixel
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
    done = Signal(int, object)  # (request number, result | error message)


class _LoadTask(QRunnable):
    """Runs ``fn(arg)`` in the pool; the result (or the error text) goes to ``signals``."""

    def __init__(self, number: int, fn: Callable[[Any], Any], arg: Any, signals: _LoadSignals):
        super().__init__()
        self.number, self.fn, self.arg, self.signals = number, fn, arg, signals

    def run(self) -> None:
        try:
            result: object = self.fn(self.arg)
        except MemoryError:
            result = "not enough memory to load the file"
        except Exception as exc:  # any unreadable file
            result = str(exc) or type(exc).__name__
        self.signals.done.emit(self.number, result)


VALUE_ROLE = Qt.ItemDataRole.UserRole


class SliceViewer(QWidget):
    """Separate window with the three views of one volume and an optional label map.

    ``pool``: thread pool for reading files; the main window passes its own, so it waits
    for a read in progress when it closes.
    """

    loaded = Signal()  # a volume (or an error) is shown
    layer_loaded = Signal()  # a label map (or an error) is shown

    def __init__(self, parent: QWidget | None = None, *, pool: QThreadPool | None = None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Image viewer")
        self.resize(1500, 640)
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
        # label maps
        self.sources: list[sl.LayerSource] = []
        self.layer: sl.LabelLayer | None = None
        self.layer_error: str | None = None
        self._layer_signals = _LoadSignals(self)
        self._layer_signals.done.connect(self._layer_done)
        self._layer_request = 0
        self._layer_loading = False
        self._hidden: dict[str, set[int]] = {}  # label file -> hidden values
        self._lut = np.zeros((1, 3), np.uint8)
        self._alpha = np.zeros(1, np.float32)
        self._plane_cache: dict[str, tuple[int, np.ndarray]] = {}

        # image row ----------------------------------------------------------
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
        top = QHBoxLayout()
        top.addWidget(self.file_label, 1)
        top.addWidget(QLabel("Window"))
        top.addWidget(self.preset_combo)
        top.addWidget(QLabel("L"))
        top.addWidget(self.level_spin)
        top.addWidget(QLabel("W"))
        top.addWidget(self.width_spin)

        # label row ----------------------------------------------------------
        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumWidth(260)
        self.layer_combo.addItem("None", None)
        self.layer_combo.currentIndexChanged.connect(self._layer_chosen)
        self.open_run_button = QPushButton("Open run…")
        self.open_run_button.setToolTip("Input image and label maps of a run folder")
        self.open_run_button.clicked.connect(self._browse_run)
        self.add_labels_button = QPushButton("Add label map…")
        self.add_labels_button.setToolTip("Any integer NIfTI over the image (nearest voxel)")
        self.add_labels_button.clicked.connect(self._browse_labels)
        self.show_check = QCheckBox("Show (L)")
        self.show_check.setChecked(True)
        self.show_check.toggled.connect(lambda _=None: self.refresh())
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setFixedWidth(120)
        self.opacity_slider.setToolTip("Opacity of the labels")
        self.opacity_slider.valueChanged.connect(lambda _=None: self.refresh())
        self.outline_check = QCheckBox("Outline")
        self.outline_check.setToolTip("Only the borders of the structures")
        self.outline_check.toggled.connect(lambda _=None: self.refresh())
        labels_row = QHBoxLayout()
        labels_row.addWidget(QLabel("Labels"))
        labels_row.addWidget(self.layer_combo, 1)
        labels_row.addWidget(self.open_run_button)
        labels_row.addWidget(self.add_labels_button)
        labels_row.addSpacing(12)
        labels_row.addWidget(self.show_check)
        labels_row.addWidget(QLabel("Opacity"))
        labels_row.addWidget(self.opacity_slider)
        labels_row.addWidget(self.outline_check)
        QShortcut(QKeySequence("L"), self, activated=self.show_check.toggle)

        # views --------------------------------------------------------------
        self.panes: dict[str, SlicePane] = {}
        self.sliders: dict[str, QSlider] = {}
        self._columns: dict[str, QWidget] = {}
        views_box = QWidget()
        views = QHBoxLayout(views_box)
        views.setContentsMargins(0, 0, 0, 0)
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

        # structures ---------------------------------------------------------
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Structure", "ml"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemDoubleClicked.connect(
            lambda item, _col=0: self.jump_to(int(item.data(0, VALUE_ROLE)))
        )
        self.layer_info = QLabel()
        self.layer_info.setWordWrap(True)
        self.layer_info.setStyleSheet("color: #7f8c8d")
        buttons = QHBoxLayout()
        for text, kind in (
            ("All", "all"),
            ("None", "none"),
            ("Bones", "bone"),
            ("Muscles", "muscle"),
        ):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, k=kind: self.show_only(k))
            buttons.addWidget(button)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(self.tree, 1)
        panel_layout.addLayout(buttons)
        panel_layout.addWidget(self.layer_info)
        self.panel = panel

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(views_box)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1220, 280])

        self.status_label = QLabel()
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hint = QLabel(
            "Left button: crosshair · wheel: slice · right drag: window/level · "
            "double click: one view / all · double click a structure: go to it · L: labels"
        )
        hint.setStyleSheet("color: #7f8c8d")

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(labels_row)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status_label)
        layout.addWidget(hint)
        self._enable(False)
        self._show_layer_info()

    # ------------------------------------------------------------------ loading

    def open(
        self,
        path: str | Path,
        layers: Sequence[sl.LayerSource] | None = None,
        select: int | None = None,
    ) -> None:
        """Show ``path`` (read in the background unless it is already shown).

        ``layers``: label maps offered for this image (``None`` keeps the current ones for
        the same image and clears them for another); ``select``: index of the one to show.
        """
        path = Path(path).resolve()
        stat = path.stat()
        source = (path, (stat.st_mtime_ns, stat.st_size))
        same = source == self._source and (self._loading or self.volume is not None)
        if layers is not None or not same:
            self.set_sources(layers or [], select)
        elif select is not None:
            self.layer_combo.setCurrentIndex(select + 1)
        if same:
            return
        self._source = source
        self._request += 1
        self._loading = True
        self.volume, self.error = None, None
        self._plane_cache.clear()
        self.setWindowTitle(f"{path.name} - image viewer")
        self.file_label.setText(path.name)
        self.status_label.setText("")
        for pane in self.panes.values():
            pane.set_message("Loading…")
        self._enable(False)
        self._pool.start(_LoadTask(self._request, load_volume, path, self._signals))

    def open_run(self, run_dir: str | Path) -> bool:
        """Input image of a run folder with its label maps (unified label map shown)."""
        image, sources = sl.run_sources(run_dir)
        if image is None:
            QMessageBox.warning(self, "Open run", f"No input image in {run_dir}.")
            return False
        self.open(image, layers=sources, select=0 if sources else None)
        return True

    def is_loading(self) -> bool:
        """The volume or a label map is being read."""
        return self._loading or self._layer_loading

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
        self._show_layer_info()
        self.apply_preset()
        self.loaded.emit()

    def _enable(self, on: bool) -> None:
        for w in (self.preset_combo, self.level_spin, self.width_spin, *self.sliders.values()):
            w.setEnabled(on)

    # ------------------------------------------------------------------ label maps

    def set_sources(self, sources: Sequence[sl.LayerSource], select: int | None = None) -> None:
        """Label maps offered in the combo box; ``select`` = index of the one to show."""
        self.sources = list(sources)
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.addItem("None", None)
        for i, src in enumerate(self.sources):
            self.layer_combo.addItem(src.title, i)
            self.layer_combo.setItemData(i + 1, str(src.path), Qt.ItemDataRole.ToolTipRole)
        self.layer_combo.setCurrentIndex(0)
        self.layer_combo.blockSignals(False)
        self._set_layer(None)
        if select is not None and 0 <= select < len(self.sources):
            self.layer_combo.setCurrentIndex(select + 1)

    def add_labels(self, path: str | Path) -> None:
        """Offer a label file (named from the run files next to it) and show it."""
        path = Path(path).resolve()
        for i, src in enumerate(self.sources):
            if src.path.resolve() == path:
                self.layer_combo.setCurrentIndex(i + 1)
                return
        source = sl.source_for_file(path)
        self.sources.append(source)
        self.layer_combo.addItem(source.title, len(self.sources) - 1)
        self.layer_combo.setItemData(len(self.sources), str(path), Qt.ItemDataRole.ToolTipRole)
        self.layer_combo.setCurrentIndex(len(self.sources))

    def _browse_run(self) -> None:
        start = str(self._source[0].parent) if self._source else str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, "Run folder", start)
        if folder:
            self.open_run(folder)

    def _browse_labels(self) -> None:
        start = str(self._source[0].parent) if self._source else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self, "Label map", start, "NIfTI (*.nii *.nii.gz);;All files (*)"
        )
        if path:
            self.add_labels(path)

    def _layer_chosen(self, _index: int = 0) -> None:
        i = self.layer_combo.currentData()
        self._set_layer(None)
        if i is None:
            return
        source = self.sources[i]
        self._layer_request += 1
        self._layer_loading = True
        self.layer_info.setText("Loading the label map…")
        self._pool.start(_LoadTask(self._layer_request, sl.load_layer, source, self._layer_signals))

    def _layer_done(self, request: int, result: object) -> None:
        if request != self._layer_request:
            return
        self._layer_loading = False
        if isinstance(result, sl.LabelLayer):
            self._set_layer(result)
        else:
            self._set_layer(None)
            self.layer_error = str(result)
            self._show_layer_info()
        self.layer_loaded.emit()

    def _set_layer(self, layer: sl.LabelLayer | None) -> None:
        if layer is None:  # also drops a read in progress
            self._layer_request += 1
            self._layer_loading = False
        self.layer, self.layer_error = layer, None
        self._plane_cache.clear()
        self._fill_tree()
        self._update_lut()
        self._show_layer_info()
        self.refresh()

    def _hidden_values(self) -> set[int]:
        if self.layer is None:
            return set()
        return self._hidden.setdefault(str(self.layer.source.path), set())

    def _fill_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        layer = self.layer
        if layer is not None:
            hidden = self._hidden_values()
            for info in layer.labels.values():
                item = QTreeWidgetItem([info.name, f"{info.volume_ml:.1f}"])
                item.setData(0, VALUE_ROLE, info.value)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    0, Qt.CheckState.Unchecked if info.value in hidden else Qt.CheckState.Checked
                )
                pixmap = QPixmap(12, 12)
                pixmap.fill(QColor(*info.color))
                item.setIcon(0, QIcon(pixmap))
                item.setTextAlignment(
                    1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                item.setToolTip(
                    0,
                    f"{info.name}: value {info.value}, {info.kind}, {info.voxels} voxels, "
                    f"{info.volume_ml:.1f} ml",
                )
                self.tree.addTopLevelItem(item)
        self.tree.blockSignals(False)

    def _item(self, value: int) -> QTreeWidgetItem | None:
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, VALUE_ROLE) == value:
                return item
        return None

    def _item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self.layer is None:
            return
        value = int(item.data(0, VALUE_ROLE))
        hidden = self._hidden_values()
        if item.checkState(0) == Qt.CheckState.Checked:
            hidden.discard(value)
        else:
            hidden.add(value)
        self._update_lut()
        self.refresh()

    def show_only(self, which: str) -> None:
        """``all``, ``none`` or a kind (``bone``, ``muscle``): structures to show."""
        if self.layer is None:
            return
        hidden = self._hidden_values()
        hidden.clear()
        for info in self.layer.labels.values():
            shown = which == "all" or info.kind == which
            if not shown:
                hidden.add(info.value)
        self._fill_tree()
        self._update_lut()
        self.refresh()

    def visible_values(self) -> list[int]:
        if self.layer is None:
            return []
        hidden = self._hidden_values()
        return [v for v in self.layer.labels if v not in hidden]

    def _update_lut(self) -> None:
        layer = self.layer
        n = (layer.max_value if layer else 0) + 1
        self._lut = np.zeros((n, 3), np.uint8)
        self._alpha = np.zeros(n, np.float32)
        if layer is None:
            return
        hidden = self._hidden_values()
        for value, info in layer.labels.items():
            self._lut[value] = info.color
            self._alpha[value] = 0.0 if value in hidden else 1.0

    def jump_to(self, value: int) -> None:
        """Crosshair on a voxel of structure ``value`` (shown if it was hidden)."""
        layer, vol = self.layer, self.volume
        if layer is None or vol is None or value not in layer.labels:
            return
        item = self._item(value)
        if item is not None and item.checkState(0) != Qt.CheckState.Checked:
            item.setCheckState(0, Qt.CheckState.Checked)  # -> _item_changed
        self.cursor = sl.to_canonical(layer, vol.affine, vol.shape, layer.locate(value))
        self.refresh()

    def _show_layer_info(self) -> None:
        layer = self.layer
        if self.layer_error:
            text = f"Cannot read the label map: {self.layer_error}"
        elif self._layer_loading:
            text = "Loading the label map…"
        elif layer is None:
            text = (
                "No label map shown. 'Open run…' shows the label maps of a run, "
                "'Add label map…' any integer NIfTI."
            )
        else:
            notes = list(layer.notes)
            vol = self.volume
            if vol is not None and not layer.same_grid(vol.file_shape, vol.file_affine):
                notes.append("grid differs from the image: nearest label voxel shown")
            text = f"{len(layer.labels)} structures · {layer.source.path.name}"
            if notes:
                text += " · " + " · ".join(notes)
        self.layer_info.setText(text)

    def _label_plane(self, view: str, index: int) -> np.ndarray:
        cached = self._plane_cache.get(view)
        if cached is not None and cached[0] == index:
            return cached[1]
        vol, layer = self.volume, self.layer
        assert vol is not None and layer is not None
        labels = sl.label_plane(layer, vol.affine, vol.shape, AXES[view], index)
        self._plane_cache[view] = (index, labels)
        return labels

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
        value = self.label_under_cursor()
        item = self._item(value) if value else None
        if item is not None:
            self.tree.setCurrentItem(item)
        else:
            self.tree.clearSelection()

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

    def label_under_cursor(self) -> int:
        if self.layer is None or self.volume is None:
            return 0
        return sl.label_at(self.layer, self.volume.affine, self.cursor)

    def refresh(self) -> None:
        vol = self.volume
        if vol is None:
            return
        low, high = self.window()
        overlay = self.layer is not None and self.show_check.isChecked()
        opacity = self.opacity_slider.value() / 100.0
        for view in VIEWS:
            s_axis, h_axis, v_axis = AXES[view]
            index = self.cursor[s_axis]
            values = plane(vol.data, view, index).astype(np.float32) * vol.slope + vol.inter
            image = to_gray(values, low, high)
            if overlay:
                labels = self._label_plane(view, index)
                image = sl.compose(
                    image, labels, self._lut, self._alpha, opacity, self.outline_check.isChecked()
                )
            cross = (
                vol.shape[h_axis] - 1 - self.cursor[h_axis],
                vol.shape[v_axis] - 1 - self.cursor[v_axis],
            )
            self.panes[view].set_slice(
                image,
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
        ]
        if self.layer is not None:
            value = self.label_under_cursor()
            info = self.layer.labels.get(value)
            parts.append(f"label {info.name} ({value})" if info else "no label")
        self.status_label.setText("  ·  ".join([*parts, *vol.notes]))

    # ------------------------------------------------------------------ closing

    def closeEvent(self, event) -> None:
        self._request += 1  # reads still in progress are dropped
        self._loading = False
        self.volume, self._source = None, None  # a CT can take gigabytes
        self._set_layer(None)
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


def main(target: Path | None = None, labels: Sequence[Path] = ()) -> int:
    """Standalone viewer (``mskpipe view``): a volume or a run folder, label maps."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    viewer = SliceViewer()
    viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    if target is not None and Path(target).is_dir():
        viewer.open_run(target)
        for path in labels:
            viewer.add_labels(path)
    elif target is not None:
        sources = [sl.source_for_file(p) for p in labels]
        viewer.open(target, layers=sources, select=0 if sources else None)
    viewer.show()
    return app.exec()
