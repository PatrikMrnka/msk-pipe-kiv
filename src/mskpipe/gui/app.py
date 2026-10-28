# SPDX-License-Identifier: Apache-2.0
"""Main window of the msk-pipe GUI (``mskpipe gui`` / ``mskpipe-gui``).

Input (NIfTI file or DICOM folder), modality, device and output folder at the top, the
configuration (built from the config schema) on the left, progress of the steps, the log
and the result on the right. The pipeline runs as a child process (:mod:`mskpipe.gui.process`).
"""

from __future__ import annotations

import html
import re
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import (
    QEvent,
    QObject,
    QProcess,
    QRunnable,
    QSettings,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mskpipe import __version__, api
from mskpipe.config.help import help_for
from mskpipe.config.loader import ConfigError, deep_merge, parse_override
from mskpipe.core.registry import Registry
from mskpipe.gui.config_form import ConfigForm
from mskpipe.gui.help_badge import HelpBadge, help_html, label_with_badge
from mskpipe.gui.process import ChildRequest, PipelineProcess, child_command, mskpipe_command
from mskpipe.gui.setup_dialog import SetupDialog, needs_attention
from mskpipe.io.dicom import DICOM_SUFFIXES

if TYPE_CHECKING:
    from mskpipe.core.device import DeviceReport
    from mskpipe.gui.qc_dialog import DefectDialog
    from mskpipe.io.dicom import DicomScan, DicomVolume
    from mskpipe.resources.status import SetupStatus

__all__ = ["STEP_LABELS", "MainWindow", "main"]

STEP_LABELS: dict[str, str] = {
    "segment": "Segmentation",
    "labelmap": "Label map",
    "mesh": "Surface meshes",
    "skeleton": "Skeletal model",
    "attachments": "Attachment areas",
    "export_mw2": "Muscle Wrapping input",
}
STATUS_COLORS: dict[str, str] = {
    "pending": "#7f8c8d",
    "running": "#2471a3",
    "completed": "#1e8449",
    "cached": "#7f8c8d",
    "failed": "#c0392b",
    "interrupted": "#ca6f1e",
    "skipped": "#7f8c8d",
}
LOG_COLORS = {"WARNING": "#ca6f1e", "ERROR": "#c0392b", "CRITICAL": "#c0392b"}
NIFTI_FILTER = "NIfTI volumes (*.nii *.nii.gz);;All files (*)"
INPUT_FORMATS = {"nifti": "NIfTI file", "dicom": "DICOM folder"}
FORMAT_TIPS = {
    "nifti": "One volume: .nii or .nii.gz",
    "dicom": "A folder with the DICOM files of a CT or MRI (subfolders included)",
}
PLACEHOLDERS = {
    "nifti": "CT or MRI volume (.nii / .nii.gz)",
    "dicom": "folder with the DICOM files of the CT or MRI",
}
_TOOLS = {"totalsegmentator": "TotalSegmentator", "musclemap": "MuscleMap"}
RED, AMBER, GREEN, GREY = "#c0392b", "#ca6f1e", "#1e8449", "#7f8c8d"
MAX_LISTED = 6  # unusable DICOM series listed under the image
CONFIG_FILTER = "YAML config (*.yaml *.yml);;All files (*)"
LOG_LINES = 20_000
AUTO = api.AUTO
DEVICES_GPU = (("Auto", "auto"), ("CPU", "cpu"), ("GPU (NVIDIA)", "gpu"))
DEVICES_CPU = (("CPU", "cpu"),)
NONRIGID = ("attachments", "params", "nonrigid")


class MainWindow(QMainWindow):
    def __init__(
        self,
        settings: QSettings | None = None,
        *,
        registry: Registry | None = None,
        command: Sequence[str] | None = None,
        probe_device: bool = True,
    ) -> None:
        super().__init__()
        self.settings = settings or QSettings("KIV ZCU", "msk-pipe")
        self.command = list(command or mskpipe_command())  # python -m mskpipe
        self.process = PipelineProcess(self)
        self.process.event.connect(self._on_event)
        self.process.text.connect(self._on_text)
        self.process.finished.connect(self._on_finished)
        self.summary: api.RunSummary | None = None
        self._steps: list[str] = []
        self._step_started: dict[str, float] = {}
        self._run_started: float | None = None
        self._stderr_tail: list[str] = []
        self._cancelling = False
        self.setup_dialog: SetupDialog | None = None
        self._setup_opened_at_start = False
        self._run_active = False
        # device: what the user wants (settings, config file, combo) and what the probe
        # found; GPU choices are offered only when a GPU run works here
        self._wanted_device = "auto"
        self.gpu_report: DeviceReport | None = None
        self._gpu_probing = False
        self.blockers: list[str] = []  # why Run is disabled (rich text), see run_blockers

        self.setWindowTitle(f"msk-pipe {__version__}")
        self.resize(1280, 820)
        self.form = ConfigForm(registry)
        self.form.changed.connect(self._schedule_validation)
        self._validate_timer = QTimer(self)
        self._validate_timer.setSingleShot(True)
        self._validate_timer.setInterval(250)
        self._validate_timer.timeout.connect(self._validate)
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick)
        self._device_proc = QProcess(self)
        self._device_proc.finished.connect(self._device_detected)
        self._modalities: dict[str, Any] = {}  # input key -> ModalityGuess | error
        self._detecting: set[str] = set()
        self._pool = QThreadPool(self)
        self._modality_signals = _ModalitySignals(self)
        self._modality_signals.done.connect(self._modality_detected)
        self._scans: dict[str, Any] = {}  # resolved DICOM folder -> DicomScan | error message
        self._scanning: set[str] = set()
        self._dicom_signals = _ModalitySignals(self)
        self._dicom_signals.done.connect(self._dicom_scanned)
        self._paths = dict.fromkeys(INPUT_FORMATS, "")  # last path per input format
        self._format = "nifti"
        self._wanted_series: str | None = None  # series to select once the folder is read

        self._build_ui()
        self._restore()
        self._set_running(False)
        self._validate()
        if probe_device:
            self.detect_device()

    # ------------------------------------------------------------------ layout

    def _build_ui(self) -> None:
        # input -------------------------------------------------------------
        self.format_group = QButtonGroup(self)
        self.format_buttons: dict[str, QRadioButton] = {}
        for value, label in INPUT_FORMATS.items():
            button = QRadioButton(label)
            button.setToolTip(FORMAT_TIPS[value])
            self.format_group.addButton(button)
            self.format_buttons[value] = button
        self.format_buttons["nifti"].setChecked(True)
        self.format_group.buttonToggled.connect(
            lambda _button, checked: checked and self._format_changed()
        )
        self.image_edit = QLineEdit()
        self.image_edit.setPlaceholderText(PLACEHOLDERS["nifti"])
        self.image_edit.editingFinished.connect(self._image_changed)
        browse_image = QPushButton("Browse…")
        browse_image.clicked.connect(self._browse_image)
        self.view_image_button = QPushButton("View…")
        self.view_image_button.setToolTip("Show the volume in axial, coronal and sagittal slices")
        self.view_image_button.setEnabled(False)
        self.view_image_button.clicked.connect(self.view_image)
        self.image_edit.textChanged.connect(lambda _=None: self._update_view_button())
        self.slice_viewer = None  # SliceViewer window, created on first use
        self.image_info = QLabel()
        self.image_info.setWordWrap(True)
        self.image_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.series_combo = QComboBox()
        self.series_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.series_combo.currentIndexChanged.connect(lambda _=None: self._series_changed())

        self.modality_combo = QComboBox()
        self.modality_combo.addItem("Auto (from image)", AUTO)
        self.modality_combo.addItem("CT", "ct")
        self.modality_combo.addItem("MRI", "mri")
        self.modality_combo.currentIndexChanged.connect(lambda _=None: self._show_modality())
        self.modality_combo.currentIndexChanged.connect(lambda _=None: self._schedule_validation())
        self.modality_label = QLabel()
        self.modality_label.setWordWrap(True)
        self.modality_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("from the file name")

        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(lambda _=None: self._device_chosen())
        self.device_label = QLabel()
        self.device_label.setWordWrap(True)
        self.device_label.linkActivated.connect(lambda _link: self.open_setup())

        self.runs_edit = QLineEdit()
        browse_runs = QPushButton("Browse…")
        browse_runs.clicked.connect(self._browse_runs)
        self.until_combo = QComboBox()
        for step in api.PIPELINE_STEPS:
            self.until_combo.addItem(STEP_LABELS[step], step)
        self.until_combo.setCurrentIndex(len(api.PIPELINE_STEPS) - 1)
        self.verbose_check = QCheckBox("Tool output in the log")

        self._badges: dict[str, HelpBadge] = {}

        inputs = QGroupBox("Input")
        grid = QFormLayout(inputs)
        formats = _row(*self.format_buttons.values(), stretch=None)
        formats.layout().addStretch(1)
        grid.addRow(self._label("Image", "gui.image"), formats)
        grid.addRow("", _row(self.image_edit, browse_image, self.view_image_button))
        grid.addRow("", self.image_info)
        self.series_row = _row(self.series_combo, stretch=None)
        self.series_row.layout().addStretch(1)
        grid.addRow(self._label("Series", "gui.series"), self.series_row)
        self._input_grid = grid
        grid.addRow(
            self._label("Modality", "gui.modality"),
            _row(self.modality_combo, self.modality_label, stretch=1),
        )
        grid.addRow(self._label("Subject", "gui.subject"), self.subject_edit)
        grid.addRow(
            self._label("Device", "gui.device"),
            _row(self.device_combo, self.device_label, stretch=1),
        )
        grid.addRow(self._label("Output folder", "gui.runs_dir"), _row(self.runs_edit, browse_runs))
        grid.addRow(
            self._label("Run until", "gui.until"),
            _row(self.until_combo, self.verbose_check, stretch=1),
        )
        verbose_help = help_for("gui.verbose")
        if verbose_help is not None:
            self.verbose_check.setToolTip(help_html(verbose_help))

        # configuration -----------------------------------------------------
        load = QPushButton("Load…")
        load.clicked.connect(self._load_config)
        save = QPushButton("Save…")
        save.clicked.connect(self._save_config)
        reset = QPushButton("Defaults")
        reset.setToolTip("Reset every setting to its default")
        reset.clicked.connect(self.form.reset)
        config_box = QGroupBox("Settings (bold = changed)")
        config_layout = QVBoxLayout(config_box)
        config_layout.addWidget(self.form, 1)
        config_layout.addLayout(_hbox(load, save, reset, stretch_first=True))

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(inputs)
        left_layout.addWidget(config_box, 1)

        # progress ----------------------------------------------------------
        self.steps_tree = QTreeWidget()
        self.steps_tree.setHeaderLabels(["Step", "Status", "Time"])
        self.steps_tree.setRootIsDecorated(False)
        self.steps_tree.setUniformRowHeights(True)
        header = self.steps_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.run_dir_label = QLabel()
        self.run_dir_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(LOG_LINES)
        self.log_view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self.run_button = QPushButton("Run")
        self.run_button.setDefault(True)
        font = QFont(self.run_button.font())
        font.setBold(True)
        self.run_button.setFont(font)
        self.run_button.clicked.connect(self.start_run)
        self.block_label = QLabel()  # why Run is disabled, with links to the fix
        self.block_label.setWordWrap(True)
        self.block_label.setTextFormat(Qt.TextFormat.RichText)
        self.block_label.linkActivated.connect(self._block_link)
        self.block_label.hide()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_run)
        self.open_run_button = QPushButton("Open run folder")
        self.open_run_button.clicked.connect(lambda: self._open(self.process.run_dir))
        self.open_mw2_button = QPushButton("Open Muscle Wrapping input")
        self.open_mw2_button.setToolTip("06_mw2_input: setup XML, model, meshes, attachments")
        self.open_mw2_button.clicked.connect(
            lambda: self._open(self.summary.mw2_dir if self.summary else None)
        )
        self.qc_button = QPushButton("Geometry check…")
        self.qc_button.setToolTip(
            "Bone holes, gaps and unusable muscle meshes found in the run, in a 3D view"
        )
        self.qc_button.clicked.connect(self.open_qc)
        self.qc_dialog: DefectDialog | None = None  # 3D view of the defects, deleted on close
        self.seg_button = QPushButton("Segmentation…")
        self.seg_button.setToolTip(
            "Input image with the label maps of the run in axial, coronal and sagittal slices"
        )
        self.seg_button.clicked.connect(self.view_segmentation)
        self.setup_button = QPushButton("Setup…")
        self.setup_button.setToolTip("Licence key, attachment atlas, model weights and GPU support")
        self.setup_button.clicked.connect(lambda: self.open_setup())

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        progress_box = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_box)
        progress_layout.addWidget(self.steps_tree)
        progress_layout.addWidget(self.progress)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.run_dir_label)
        log_box = QGroupBox("Log")
        QVBoxLayout(log_box).addWidget(self.log_view)
        right_layout.addWidget(progress_box)
        right_layout.addWidget(log_box, 1)
        right_layout.addWidget(self.block_label)
        right_layout.addLayout(
            _hbox(
                self.setup_button,
                self.open_run_button,
                self.open_mw2_button,
                self.qc_button,
                self.seg_button,
                self.cancel_button,
                self.run_button,
                stretch_first=False,
                stretch_at=5,
            )
        )

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(splitter)
        self.setCentralWidget(central)
        self._reset_steps(api.PIPELINE_STEPS)

    # ------------------------------------------------------------------ settings

    def _restore(self) -> None:
        s = self.settings
        self._paths["nifti"] = str(s.value("image", ""))
        self._paths["dicom"] = str(s.value("dicom_folder", ""))
        self._wanted_series = str(s.value("series", "")) or None
        fmt = str(s.value("input_format", "nifti"))
        self._format = fmt if fmt in INPUT_FORMATS else "nifti"
        self.format_group.blockSignals(True)
        self.format_buttons[self._format].setChecked(True)
        self.format_group.blockSignals(False)
        self.image_edit.setPlaceholderText(PLACEHOLDERS[self._format])
        self.image_edit.setText(self._paths[self._format])
        _select(self.modality_combo, s.value("modality", AUTO))
        self._wanted_device = str(s.value("device", "auto"))
        self._fill_devices()
        self.runs_edit.setText(str(s.value("runs_dir", str(Path.cwd() / "runs"))))
        _select(self.until_combo, s.value("until", api.PIPELINE_STEPS[-1]))
        self.verbose_check.setChecked(str(s.value("verbose", "false")).lower() == "true")
        self.form.set_show_advanced(str(s.value("show_advanced", "false")).lower() == "true")
        overrides = s.value("overrides", []) or []
        if isinstance(overrides, str):
            overrides = [overrides]
        try:
            data: dict[str, Any] = {}
            for item in overrides:
                data = deep_merge(data, parse_override(str(item)))
            self.form.set_values(data)
        except ConfigError:  # settings from an older version: start from the defaults
            self.form.reset()
        self._image_changed()

    def _store(self) -> None:
        s = self.settings
        self._paths[self._format] = self.image_edit.text()
        s.setValue("input_format", self._format)
        s.setValue("image", self._paths["nifti"])
        s.setValue("dicom_folder", self._paths["dicom"])
        vol = self.current_dicom()
        s.setValue("series", vol.id if vol is not None else (self._wanted_series or ""))
        s.setValue("modality", self.modality_combo.currentData())
        s.setValue("device", self._wanted_device)
        s.setValue("runs_dir", self.runs_edit.text())
        s.setValue("until", self.until_combo.currentData())
        s.setValue("verbose", "true" if self.verbose_check.isChecked() else "false")
        s.setValue("show_advanced", "true" if self.form.show_advanced else "false")
        s.setValue("overrides", self.form.overrides())
        s.sync()

    # ------------------------------------------------------------------ input

    def input_format(self) -> str:
        """``nifti`` (a file) or ``dicom`` (a folder of DICOM files)."""
        return self._format

    def set_input(self, path: str | Path, fmt: str | None = None) -> None:
        """Choose the input (format switch, path) and check it, as the user would."""
        if fmt is not None and fmt != self._format:
            self.set_format(fmt)
        self.image_edit.setText(str(path))
        self._image_changed()

    def set_format(self, fmt: str) -> None:
        """Switch between ``nifti`` and ``dicom`` (each keeps its own last path)."""
        self.format_buttons[fmt].setChecked(True)  # toggled -> _format_changed

    def _format_changed(self) -> None:
        fmt = next(k for k, b in self.format_buttons.items() if b.isChecked())
        if fmt == self._format:
            return
        self._paths[self._format] = self.image_edit.text()
        self._format = fmt
        self.image_edit.setPlaceholderText(PLACEHOLDERS[fmt])
        self.image_edit.setText(self._paths[fmt])
        self._image_changed()

    def _browse_image(self) -> None:
        text = self.image_edit.text().strip()
        start = text or str(Path.cwd())
        if self._format == "dicom":
            path = QFileDialog.getExistingDirectory(self, "DICOM folder", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Input volume", start, NIFTI_FILTER)
        if path:
            self.image_edit.setText(path)
            self._image_changed()

    def _browse_runs(self) -> None:
        start = self.runs_edit.text() or str(Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "Output folder", start)
        if path:
            self.runs_edit.setText(path)

    def _image_changed(self) -> None:
        """Check the input when the path or the format changed (DICOM: in the background)."""
        self._schedule_validation()  # the weights a run needs depend on the modality
        text = self.image_edit.text().strip()
        self.image_info.setVisible(bool(text))
        self._show_series_row()
        if not text:
            self.image_info.setText("")
            self.subject_edit.setPlaceholderText("from the file or folder name")
            self._show_modality()
            self._update_view_button()
            return
        path = Path(text)
        if self._format == "dicom":
            self._dicom_changed(path)
        else:
            self._nifti_changed(path)
        self._update_view_button()

    def _nifti_changed(self, path: Path) -> None:
        error = None
        if path.is_dir():
            error = "This is a folder; for DICOM images switch to 'DICOM folder'"
        elif not path.is_file():
            error = "File not found"
        elif path.suffix.lower() in DICOM_SUFFIXES:
            error = "This is a DICOM file; switch to 'DICOM folder' and choose its folder"
        elif not path.name.lower().endswith((".nii", ".nii.gz")):
            error = "Not a NIfTI file (.nii or .nii.gz)"
        if error is not None:
            self.image_info.setText(_colored(error, RED))
            self._show_modality()
            return
        self._detect_modality(path)
        lines = []
        try:
            lines.append(html.escape(api.describe_image(path)))
        except Exception as exc:  # any unreadable header
            lines.append(_colored(f"Cannot read the header: {exc}", RED))
        for issue in api.check_image(path):
            lines.append(_colored(issue.message, RED if issue.severity == "error" else AMBER))
        self.image_info.setText("<br>".join(lines))
        self.subject_edit.setPlaceholderText(_stem(path))

    def _update_view_button(self) -> None:
        self.view_image_button.setEnabled(self._view_path() is not None)

    def _view_path(self) -> Path | None:
        """NIfTI shown by View…: the file, or the converted DICOM series."""
        if self._format == "dicom":
            vol = self.current_dicom()
            return vol.nifti_path if vol is not None and vol.nifti_path.is_file() else None
        text = self.image_edit.text().strip()
        path = Path(text) if text else None
        return path if path is not None and path.is_file() and _is_nifti(path) else None

    def view_image(self) -> QWidget | None:
        """Show the input volume in axial, coronal and sagittal slices (own window)."""
        path = self._view_path()
        if path is None:
            QMessageBox.warning(self, "View image", "Choose an existing input volume first.")
            return None
        viewer = self._viewer()
        viewer.open(path)
        return viewer

    # ------------------------------------------------------------------ DICOM

    def _dicom_changed(self, path: Path) -> None:
        error = None
        if path.is_file():
            error = (
                "This is a file; choose the folder with the DICOM files"
                if path.suffix.lower() in DICOM_SUFFIXES or not _is_nifti(path)
                else "This is a NIfTI file; switch to 'NIfTI file'"
            )
        elif not path.is_dir():
            error = "Folder not found"
        self.subject_edit.setPlaceholderText(path.name or "from the folder name")
        if error is not None:
            self._fill_series(None)
            self.image_info.setText(_colored(error, RED))
            self._show_modality()
            return
        key = str(path.resolve())
        if key in self._scans:
            self._show_dicom()
            return
        self._fill_series(None)
        self.image_info.setText(_colored("Reading the DICOM folder (dcm2niix)…", GREY))
        self._show_modality()
        if key not in self._scanning:
            self._scanning.add(key)
            self._pool.start(_DicomTask(key, self._dicom_signals))

    def _dicom_scanned(self, key: str, result: Any) -> None:
        self._scanning.discard(key)
        self._scans[key] = result
        if self._format == "dicom" and self._folder_key() == key:
            self._show_dicom()

    def _folder_key(self) -> str | None:
        text = self.image_edit.text().strip()
        if self._format != "dicom" or not text or not Path(text).is_dir():
            return None
        return str(Path(text).resolve())

    def dicom_scan(self, *, wait: bool = False) -> DicomScan | str | None:
        """Result of reading the current DICOM folder (an error message if it failed);
        None while it is read, unless ``wait`` (then it is read here)."""
        key = self._folder_key()
        if key is None:
            return None
        if key not in self._scans and wait:
            try:
                self._scans[key] = api.scan_dicom(key)
            except api.SetupError as exc:
                self._scans[key] = str(exc)
            self._scanning.discard(key)
            self._show_dicom()
        return self._scans.get(key)

    def current_dicom(self) -> DicomVolume | None:
        """The DICOM series chosen under Series (None: no usable one, or not read yet)."""
        result = self.dicom_scan()
        if result is None or isinstance(result, str):
            return None
        return result.get(self.series_combo.currentData() or "")

    def _fill_series(self, result: DicomScan | None) -> None:
        wanted = self.series_combo.currentData() or self._wanted_series
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        for vol in result.usable if result is not None else []:
            self.series_combo.addItem(_series_text(vol), vol.id)
        if wanted is not None:
            _select(self.series_combo, wanted)
        self.series_combo.setEnabled(self.series_combo.count() > 1 and not self._run_active)
        self.series_combo.blockSignals(False)
        self._show_series_row()

    def _show_series_row(self) -> None:
        visible = self._format == "dicom" and self.series_combo.count() > 0
        self._input_grid.setRowVisible(self.series_row, visible)

    def _show_dicom(self) -> None:
        """Info under the image: chosen series, conversion warnings, unusable series."""
        result = self.dicom_scan()
        if result is None:
            return
        if isinstance(result, str):
            self._fill_series(None)
            self.image_info.setText(_colored(result.splitlines()[0], RED))
            self._show_modality()
            self._update_view_button()
            return
        if self.series_combo.count() == 0 or self.series_combo.currentData() is None:
            self._fill_series(result)
        self._series_changed()

    def _series_changed(self) -> None:
        self._schedule_validation()
        result = self.dicom_scan()
        if result is None or isinstance(result, str):
            return
        if self.series_combo.currentData():
            self._wanted_series = self.series_combo.currentData()
        usable = result.usable
        vol = self.current_dicom()
        lines = []
        if not usable:
            lines.append(_colored("No usable 3D CT or MR series in this folder", RED))
        elif len(usable) > 1:
            lines.append(_colored(f"{len(usable)} usable series: choose one under Series", AMBER))
        if vol is not None:
            self._modalities[self._current_key() or ""] = api.dicom_modality(vol)
            text = f"Series {vol.id}: "
            try:
                text += api.describe_image(vol.nifti_path)
            except Exception as exc:  # converted file removed from the cache
                text += f"cannot read the converted volume ({exc})"
            lines.append(html.escape(f"{text}, {vol.files} DICOM files"))
            for warning in vol.warnings:
                lines.append(_colored(f"dcm2niix: {warning}", AMBER))
            for issue in api.check_image(vol.nifti_path):
                lines.append(_colored(issue.message, RED if issue.severity == "error" else AMBER))
        unusable = [v for v in result.volumes if not v.usable and not v.replaced_by]
        for v in unusable[:MAX_LISTED]:
            desc = f" '{v.description}'" if v.description else ""
            lines.append(_colored(f"Not usable: series {v.id}{desc}: {v.status()}", GREY))
        if len(unusable) > MAX_LISTED:
            lines.append(_colored(f"… and {len(unusable) - MAX_LISTED} more", GREY))
        self.image_info.setText("<br>".join(lines))
        self._show_modality()
        self._update_view_button()

    def view_segmentation(self) -> QWidget | None:
        """Input image of the last run with its label maps (unified label map shown)."""
        run_dir = self.process.run_dir
        if run_dir is None or not self._has_segmentation():
            return None
        viewer = self._viewer()
        return viewer if viewer.open_run(run_dir) else None

    def _viewer(self):
        """The slice viewer window, created on first use and brought to the front."""
        from mskpipe.gui.slice_viewer import SliceViewer

        if self.slice_viewer is None:
            # the window's thread pool: closing the main window waits for a read in progress
            self.slice_viewer = SliceViewer(self, pool=self._pool)
        self.slice_viewer.show()
        self.slice_viewer.raise_()
        self.slice_viewer.activateWindow()
        return self.slice_viewer

    def _has_segmentation(self) -> bool:
        from mskpipe.gui.slice_labels import run_sources

        run_dir = self.process.run_dir
        if run_dir is None or not Path(run_dir).is_dir():
            return False
        image, sources = run_sources(run_dir)
        return image is not None and bool(sources)

    # ------------------------------------------------------------------ modality

    def _detect_modality(self, path: Path) -> None:
        """Detect CT/MRI in a background thread (reads the volume); result is cached."""
        key = str(path.resolve())
        if key in self._modalities or key in self._detecting:
            self._show_modality()
            return
        self._detecting.add(key)
        self._show_modality()
        task = _ModalityTask(key, self._modality_signals)
        self._pool.start(task)

    def _modality_detected(self, key: str, guess: Any) -> None:
        self._detecting.discard(key)
        self._modalities[key] = guess
        self._show_modality()
        self._schedule_validation()  # the weights a run needs depend on the modality

    def _current_key(self) -> str | None:
        """Key of the current input in the modality cache (DICOM: folder and series)."""
        if self._format == "dicom":
            folder = self._folder_key()
            series = self.series_combo.currentData()
            return f"{folder}|{series}" if folder and series else None
        text = self.image_edit.text().strip()
        if not text or not Path(text).is_file():
            return None
        return str(Path(text).resolve())

    def _show_modality(self) -> None:
        key = self._current_key()
        if key is None:
            self.modality_label.setText("")
            return
        if key in self._detecting:
            self.modality_label.setText("detecting…")
            return
        guess = self._modalities.get(key)
        if guess is None:
            self.modality_label.setText("")
            return
        if isinstance(guess, str):  # detection failed with an error
            self.modality_label.setText(_colored(f"cannot read the volume: {guess}", "#c0392b"))
            return
        chosen = self.modality_combo.currentData()
        if guess.modality is None:
            color = "#c0392b" if chosen == AUTO else "#7f8c8d"
            text = f"not detected ({guess.reason}); choose CT or MRI"
        elif chosen not in (AUTO, guess.modality):
            color = "#ca6f1e"
            text = f"the image looks like {guess.modality.upper()} ({guess.reason})"
        else:
            color = "#1e8449"
            text = f"detected {guess.modality.upper()} ({guess.source}: {guess.reason})"
        self.modality_label.setText(_colored(text, color))

    def effective_modality(self, image: Path | None = None) -> str:
        """CT/MRI chosen by the user, or detected for Auto (synchronously if not done yet).

        ``image``: a NIfTI file; a DICOM input uses the modality of its chosen series."""
        chosen = self.modality_combo.currentData()
        if chosen != AUTO:
            return chosen
        if self._format == "dicom":
            vol = self.current_dicom()
            guess = api.dicom_modality(vol) if vol is not None else None
            if guess is None or guess.modality is None:
                raise ValueError(
                    "Cannot detect the modality of the DICOM series. Choose CT or MRI."
                )
            return guess.modality
        if image is None:
            image = Path(self.image_edit.text().strip())
        key = str(image.resolve())
        guess = self._modalities.get(key)
        if guess is None or isinstance(guess, str):
            try:
                guess = api.detect_modality(image)
            except Exception as exc:  # unreadable volume: reported as the input error
                raise ValueError(f"Cannot detect the modality of {image.name}: {exc}") from None
            self._modalities[key] = guess
            self._show_modality()
        if guess.modality is None:
            raise ValueError(
                f"Cannot detect the modality of {image.name} ({guess.reason}). Choose CT or MRI."
            )
        return guess.modality

    # ------------------------------------------------------------------ setup

    def setup_status(self, modalities: Sequence[str] | None = None) -> SetupStatus:
        """What a run with the current settings needs (:mod:`mskpipe.resources.status`)."""
        from mskpipe.resources.status import collect

        config = api.resolved_config(None, tuple(self.form.overrides()))
        return collect(config, modalities or self.setup_modalities())

    def run_blockers(self, modalities: Sequence[str] | None = None) -> list[str]:
        """Why a run cannot start now (rich text with links to the fix); empty = it can.

        Required: the TotalSegmentator licence key (only when a licensed task runs, i.e.
        bones from TotalSegmentator), the attachment atlas (atlas_registration), the model
        weights of the configured segmenters and Fast-RNRR when nonrigid = fast_rnrr. GPU
        support is optional (without it only the CPU is offered). ``modalities``: those of
        the run; by default the current input's, or both while it is not known.
        """
        if self.setup_running() and self.setup_dialog is not None:
            task = html.escape(self.setup_dialog.blocked_reason() or "")
            return [
                f"<b>A setup task is running</b> ({task}). The run can start when it has "
                "finished or was cancelled in "
                '<a href="setup">Setup…</a>.'
            ]
        if not self.form.error_label.isHidden():
            return ["<b>Invalid settings</b> - see the message below the settings."]
        mods = tuple(modalities or self.setup_modalities())
        try:
            status = self.setup_status(mods)
        except Exception as exc:  # e.g. a setting the form does not check
            return [f"<b>Cannot check what the run needs:</b> {html.escape(str(exc))}"]
        out: list[str] = []
        lic = status.licence
        if lic.needed and (not lic.present or lic.problem):
            tasks = self._licensed_tasks(mods)
            state = "is not set" if not lic.present else f"is invalid ({lic.problem})"
            out.append(
                f"<b>TotalSegmentator licence key {html.escape(state)}.</b> With "
                "<i>Segmentation → Bones = totalsegmentator</i> the tibia and fibula come "
                f"from the licensed task {html.escape(', '.join(tasks) or 'appendicular_bones')}"
                '. Enter the key in <a href="setup">Setup…</a>, or '
                '<a href="musclemap">take all bones from MuscleMap</a> (no key needed).'
            )
        if status.atlas.needed and not status.atlas.ok:
            out.append(
                "<b>The attachment atlas is missing.</b> <i>Attachments = "
                "atlas_registration</i> transfers the attachment areas from it. Install it "
                'in <a href="setup">Setup…</a>.'
            )
        rnrr = status.fast_rnrr
        if rnrr is not None and rnrr.needed and not rnrr.ok:
            out.append(
                "<b>Fast-RNRR is not built on this computer</b>, but <i>Non-rigid = "
                "fast_rnrr</i> is set (e.g. by a loaded configuration). Build it in "
                '<a href="setup">Setup…</a> or <a href="nonrigid">use coherent_icp</a>.'
            )
        missing = self._missing_weights(status, mods)
        unknown = [w for w in missing if w.status == "unknown"]
        if unknown:
            out.append(
                f"<b>The model weights cannot be checked:</b> {html.escape(unknown[0].detail)}."
            )
        elif missing:
            names = ", ".join(f"{_TOOLS.get(w.tool, w.tool)} {w.name}" for w in missing)
            out.append(
                f"<b>Model weights are missing:</b> {html.escape(names)}. Download them in "
                '<a href="setup">Setup…</a> (Download now; needs internet).'
            )
        return out

    def _licensed_tasks(self, modalities: Sequence[str]) -> list[str]:
        """TotalSegmentator tasks of the current settings that need the licence key."""
        from mskpipe.plugins.segmenters.totalsegmentator import LICENSED_TASKS
        from mskpipe.resources.weights import required_ts_tasks

        config = api.resolved_config(None, tuple(self.form.overrides()))
        return sorted(set(required_ts_tasks(config, modalities)) & LICENSED_TASKS)

    def _missing_weights(self, status: SetupStatus, modalities: Sequence[str]) -> list:
        """Weights a run needs that are not present. While the modality of the input is
        not known (both modalities asked), the weights of one complete modality suffice;
        the start of the run checks the actual modality."""
        missing = [w for w in status.weights if w.status != "ok"]
        if not missing or len(modalities) < 2:
            return missing
        from mskpipe.resources.weights import weights_status

        config = api.resolved_config(None, tuple(self.form.overrides()))
        for modality in modalities:
            if all(w.status == "ok" for w in weights_status(config, (modality,))):
                return []
        return missing

    def _wait_reason(self) -> str | None:
        """A short wait before Run is possible (not an error)."""
        if self._gpu_probing and self._wanted_device != "cpu":
            return "Detecting whether the GPU can be used…"
        return None

    def _update_run_state(self) -> None:
        """Enable Run only when nothing blocks it; say why not above the buttons."""
        running = self._run_active or self.process.is_running()
        self.blockers = [] if running else self.run_blockers()
        wait = None if running else self._wait_reason()
        self._offer_nonrigid()
        self.run_button.setEnabled(not running and not self.blockers and wait is None)
        items = "".join(f"<li>{b}</li>" for b in self.blockers)
        text = ""
        if items:
            text = (
                f'<span style="color:{RED}">The run cannot start:</span>'
                f'<ul style="margin:0 0 0 -24px">{items}</ul>'
            )
        if wait:
            text += _colored(wait, GREY)
        self.block_label.setText(text)
        self.block_label.setVisible(bool(text))
        self.run_button.setToolTip("\n".join(_plain(b) for b in self.blockers) or (wait or ""))

    def _offer_nonrigid(self) -> None:
        """Offer nonrigid = fast_rnrr only where Fast-RNRR is available (optional)."""
        from mskpipe.resources.fast_rnrr import locate

        params = self.form.values().get("attachments", {}).get("params", {})
        try:
            path, _source = locate(params.get("fast_rnrr_exe"))
        except OSError:
            path = None
        reason = "Fast-RNRR is not built on this computer (Setup… → Fast-RNRR)"
        self.form.set_unavailable(NONRIGID, {} if path is not None else {"fast_rnrr": reason})

    def _block_link(self, link: str) -> None:
        if link == "musclemap":
            self.apply_setting(("segmentation", "bones"), "musclemap")
        elif link == "nonrigid":
            self.apply_setting(NONRIGID, "coherent_icp")
        else:
            self.open_setup()

    def setup_modalities(self) -> tuple[str, ...]:
        """CT/MRI of the current input if known (no detection here), else both."""
        chosen = self.modality_combo.currentData()
        if chosen in ("ct", "mri"):
            return (chosen,)
        key = self._current_key()
        guess = self._modalities.get(key) if key else None
        modality = getattr(guess, "modality", None)
        return (modality,) if modality in ("ct", "mri") else ("ct", "mri")

    def open_setup(self, *, at_start: bool = False) -> SetupDialog:
        """Show the setup window. There is one for the life of the main window: a task
        started in it keeps running when it is closed, and reopening shows its progress
        (a new window would allow a second download next to the first one)."""
        show_at_start = str(self.settings.value("setup_at_start", "true")).lower() == "true"
        dialog = self.setup_dialog
        if dialog is None:
            dialog = SetupDialog(
                self.setup_status,
                command=self.command,
                modalities=self.setup_modalities(),
                overrides_fn=self.form.overrides,
                busy_fn=self._setup_blocked,
                parent=self,
            )
            dialog.use_musclemap.connect(
                lambda: self.apply_setting(("segmentation", "bones"), "musclemap")
            )
            dialog.gpu_changed.connect(self.detect_device)
            dialog.task_changed.connect(self._setup_task_changed)
            dialog.finished.connect(lambda _code: self._setup_closed())
            self.setup_dialog = dialog
        if not dialog.isVisible():  # settings of the main window may have changed
            dialog.modalities = self.setup_modalities()
            dialog.set_at_start(show_at_start if at_start else None)
            self._setup_opened_at_start = at_start
            dialog.refresh()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _setup_closed(self) -> None:
        dialog = self.setup_dialog
        if dialog is not None and self._setup_opened_at_start:
            value = "true" if dialog.at_start_check.isChecked() else "false"
            self.settings.setValue("setup_at_start", value)
            self.settings.sync()

    def setup_running(self) -> bool:
        return self.setup_dialog is not None and self.setup_dialog.is_running()

    def _setup_blocked(self) -> str | None:
        return "A pipeline run is in progress" if self._run_active else None

    def _setup_task_changed(self, running: bool) -> None:
        if running and self.setup_dialog is not None:
            self.statusBar().showMessage(f"Setup: {self.setup_dialog.blocked_reason()}")
        else:
            self.statusBar().clearMessage()
        self._set_running(self._run_active)

    def check_setup(self) -> SetupDialog | None:
        """At start: open the setup window if a run would stop on a missing licence key or
        atlas (unless the user turned that off)."""
        if str(self.settings.value("setup_at_start", "true")).lower() != "true":
            return None
        try:
            missing = needs_attention(self.setup_status())
        except Exception:  # invalid remembered settings: the form reports them
            return None
        if not missing:
            return None
        self.statusBar().showMessage(f"Missing: {', '.join(missing)}", 10_000)
        return self.open_setup(at_start=True)

    def apply_setting(self, path: Sequence[str], value: Any) -> None:
        """Set one config value in the form (e.g. from the setup window)."""
        data = self.form.values()
        node = data
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
        self.form.set_values(data)
        self.statusBar().showMessage(f"{'.'.join(path)} = {value}", 5000)

    # ------------------------------------------------------------------ help

    def _label(self, text: str, key: str) -> QWidget:
        """Form label with a '?' showing the explanation of ``key`` from help.yaml."""
        info = help_for(key)
        label = QLabel(text)
        if info is None:
            return label
        tip = help_html(info)
        label.setToolTip(tip)
        badge = HelpBadge(tip)
        self._badges[key] = badge
        return label_with_badge(label, badge)

    # ------------------------------------------------------------------ device

    @property
    def gpu_usable(self) -> bool:
        """A GPU run works here: NVIDIA GPU, driver and a CUDA build of PyTorch (GPU
        support installed) - found by :meth:`detect_device`."""
        return self.gpu_report is not None and self.gpu_report.selected == "cuda"

    def detect_device(self) -> None:
        """Probe whether a GPU run works (``mskpipe device --device auto --json``, child
        process: NVML / nvidia-smi and a PyTorch CUDA test). Until it answers, and when
        no GPU is usable, only the CPU is offered."""
        if self._device_proc.state() != QProcess.ProcessState.NotRunning:
            self._device_proc.kill()
            self._device_proc.waitForFinished(2000)
        self._gpu_probing = True
        self.device_label.setText(_colored("detecting the GPU…", GREY))
        argv = [*self.command, "device", "--device", "auto", "--json"]
        self._device_proc.start(argv[0], argv[1:])
        self._update_run_state()

    def _device_detected(self, code: int, _status: QProcess.ExitStatus) -> None:
        out = bytes(self._device_proc.readAllStandardOutput()).decode("utf-8", "replace")
        err = bytes(self._device_proc.readAllStandardError()).decode("utf-8", "replace")
        self._gpu_probing = False
        self.gpu_report = None
        text = None
        if code == 0:
            from mskpipe.core.device import DeviceReport

            with suppress(ValueError):
                self.gpu_report = DeviceReport.model_validate_json(out)
        report = self.gpu_report
        if report is not None and self.gpu_usable and report.gpu is not None:
            text = _colored(f"GPU available: cuda:{report.gpu.index} ({report.gpu.name})", GREEN)
        elif report is not None:
            reason = report.reason.removeprefix("auto: ")
            text = _colored(f"CPU only - GPU not available: {reason}", GREY)
            if "mskpipe setup gpu" in report.reason:
                text += ' - <a href="setup">install GPU support…</a>'
        else:
            message = err.strip().removeprefix("Error: ") or "GPU detection failed"
            text = _colored(f"CPU only - {message}", RED)
        self.device_label.setText(text)
        self._fill_devices()
        self._update_run_state()

    def _fill_devices(self) -> None:
        """Auto / CPU / GPU when a GPU run works here, else CPU only (GPU support is
        optional). The wanted device is kept and selected once it is offered."""
        combo = self.device_combo
        options = DEVICES_GPU if self.gpu_usable else DEVICES_CPU
        combo.blockSignals(True)
        try:
            combo.clear()
            for label, value in options:
                combo.addItem(label, value)
            i = combo.findData(self._wanted_device)
            combo.setCurrentIndex(i if i >= 0 else combo.findData("cpu"))
        finally:
            combo.blockSignals(False)
        combo.setToolTip("" if self.gpu_usable else "GPU not available here: the run uses the CPU")

    def _device_chosen(self) -> None:
        value = self.device_combo.currentData()
        if value is not None:
            self._wanted_device = str(value)
        self._update_run_state()

    # ------------------------------------------------------------------ config

    def _schedule_validation(self) -> None:
        self._validate_timer.start()

    def _validate(self) -> bool:
        ok = self.form.validate() is None
        if not self.process.is_running():
            self._update_run_state()
            dialog = self.setup_dialog
            if dialog is not None and dialog.isVisible() and not dialog.is_running():
                dialog.modalities = self.setup_modalities()  # e.g. MuscleMap <-> TS bones
                dialog.refresh()
        return ok

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load configuration", "", CONFIG_FILTER)
        if not path:
            return
        try:
            data = self.form.load(path)
        except api.SetupError as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return
        runtime = data.get("runtime", {})
        if runtime.get("device"):
            self._wanted_device = str(runtime["device"])
            self._fill_devices()
        if runtime.get("runs_dir"):
            runs = Path(runtime["runs_dir"])
            self.runs_edit.setText(str(runs if runs.is_absolute() else Path(path).parent / runs))
        note = ""
        if self._wanted_device != "cpu" and not self.gpu_usable:
            note = " - GPU not available here, the run uses the CPU"
        self.statusBar().showMessage(f"Loaded {path}{note}", 8000)

    def _save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save configuration", "", CONFIG_FILTER)
        if not path:
            return
        extra = {"runtime": {"device": self.device_combo.currentData()}}
        try:
            self.form.save(path, extra)
        except OSError as exc:
            QMessageBox.critical(self, "Cannot save", str(exc))
            return
        self.statusBar().showMessage(f"Saved {path}", 5000)

    # ------------------------------------------------------------------ run

    def child_request(self) -> ChildRequest:
        """The run described by the window; raises ValueError for missing input."""
        image = self.image_edit.text().strip()
        series = None
        if self._format == "dicom":
            if not image:
                raise ValueError("Choose the folder with the DICOM files.")
            if not Path(image).is_dir():
                raise ValueError(f"DICOM folder not found: {image}")
            result = self.dicom_scan(wait=True)
            if isinstance(result, str):
                raise ValueError(result)
            vol = self.current_dicom()
            if vol is None:
                raise ValueError(f"No usable 3D CT or MR series in {Path(image).name}.")
            series = vol.id
        elif not image:
            raise ValueError("Choose an input volume (.nii or .nii.gz).")
        runs = self.runs_edit.text().strip()
        if not runs:
            raise ValueError("Choose an output folder.")
        until = self.until_combo.currentData()
        return ChildRequest(
            image=Path(image),
            series=series,
            modality=self.effective_modality(None if series else Path(image)),
            runs_dir=Path(runs),
            device=self.device_combo.currentData(),
            subject=self.subject_edit.text().strip() or None,
            until_step=None if until == api.PIPELINE_STEPS[-1] else until,
            overrides=tuple(self.form.overrides()),
            verbose=self.verbose_check.isChecked(),
        )

    def start_run(self) -> bool:
        """Check the input and settings, run the pre-run checks and start the child."""
        if self.process.is_running():
            return False
        if self.setup_running():
            QMessageBox.warning(
                self,
                "Setup task running",
                "A setup task (download or installation) is running. Wait until it has "
                "finished or cancel it in the Setup… window, then start the run.",
            )
            self.open_setup()
            return False
        if not self._validate():
            QMessageBox.warning(self, "Invalid settings", self.form.error_label.text())
            return False
        try:
            child = self.child_request()
        except (ValueError, api.SetupError) as exc:
            QMessageBox.warning(self, "Cannot start", str(exc))
            return False
        blockers = self.run_blockers((child.modality,))  # the actual modality
        if blockers:
            self._update_run_state()
            QMessageBox.warning(
                self,
                "Cannot start",
                "The run cannot start:\n\n" + "\n\n".join(_plain(b) for b in blockers),
            )
            return False
        try:
            prepared = api.prepare_run(
                api.RunRequest(
                    image=child.image,
                    modality=child.modality,
                    subject=child.subject,
                    series=child.series,
                    overrides=child.overrides,
                    device=child.device,
                    runs_dir=child.runs_dir,
                    until_step=child.until_step,
                )
            )
        except (ValueError, api.SetupError) as exc:
            QMessageBox.warning(self, "Cannot start", str(exc))
            return False
        issues = api.preflight(prepared)
        errors = [i for i in issues if i.severity == "error"]
        if errors and not self._confirm_errors(errors):
            return False
        self._store()
        self.log_view.clear()
        for issue in issues:
            self._append_log(issue.severity.upper(), f"{issue.where}: {issue.message}")
        self.summary = None
        self._stderr_tail = []
        self._cancelling = False
        self._reset_steps(prepared.steps)
        self.status_label.setText(
            "Starting… (device detection, missing model weights, copying the input)"
        )
        self.run_dir_label.setText("")
        self._run_started = time.monotonic()
        self._set_running(True)
        self.process.start(child_command(child, self.command))
        self._clock.start()
        return True

    def _confirm_errors(self, errors: Sequence[api.Issue]) -> bool:
        text = "\n\n".join(f"{i.where}: {i.message}" for i in errors)
        answer = QMessageBox.warning(
            self,
            "Pre-run checks failed",
            f"The run would probably fail:\n\n{text}\n\n"
            "Missing licence key, atlas or weights can be set up with Setup….\n\nStart anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def cancel_run(self) -> None:
        if not self.process.is_running():
            return
        self._cancelling = True
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling… (the running tool is being stopped)")
        self.process.cancel()

    # ------------------------------------------------------------------ events

    def _on_event(self, event: api.RunEvent) -> None:
        if event.kind == "run_started":
            if event.steps:
                self._reset_steps(event.steps)
            if event.run_dir:
                self.run_dir_label.setText(f"Run folder: {event.run_dir}")
                self.open_run_button.setEnabled(True)
            self.status_label.setText("Running")  # the device is in the runner's log
        elif event.kind == "step" and event.step:
            self._set_step(event.step, event.status or "", event.wall_s)
        elif event.kind == "log":
            self._append_log(event.level or "INFO", event.message or "")
        elif event.kind == "run_finished":
            self._append_log("INFO", f"Run {event.status} ({event.exit_code})")

    def _on_text(self, line: str, channel: str) -> None:
        if channel == "stderr":
            self._stderr_tail = [*self._stderr_tail[-30:], line]
        level = "ERROR" if line.startswith(("Error", "Traceback")) else "INFO"
        if line.startswith(("WARNING", "ERROR")):
            level = line.split(":", 1)[0]
        self._append_log(level, line)

    def _on_finished(self, code: int, killed: bool) -> None:
        self._clock.stop()
        run_dir = self.process.run_dir
        if killed and run_dir is not None:
            try:
                api.finalize_stale(run_dir, "killed by the GUI after a cancel timeout")
            except (OSError, ValueError, RuntimeError) as exc:
                self._append_log("WARNING", f"Cannot finalise the manifest: {exc}")
        self.summary = None
        if run_dir is not None:
            try:
                self.summary = api.summarize(run_dir)
            except (OSError, ValueError, RuntimeError) as exc:
                self._append_log("WARNING", f"Cannot read the run summary: {exc}")
        self._set_running(False)
        self._show_result(code, killed)

    def _show_result(self, code: int, killed: bool) -> None:
        s = self.summary
        if s is None:
            detail = "\n".join(self._stderr_tail[-12:]) or f"exit code {code}"
            if self._cancelling:
                self.status_label.setText(
                    _colored("Cancelled before the run started", STATUS_COLORS["interrupted"])
                )
                return
            self.status_label.setText(_colored("The run did not start", "#c0392b"))
            if not self._cancelling:
                QMessageBox.critical(self, "The run did not start", detail)
            return
        for step in s.steps:
            if step.name in self._steps:
                wall = step.wall_s if step.status == "completed" else None
                self._set_step(step.name, step.status, wall)
        done = sum(st.status in ("completed", "cached") for st in s.steps)
        self.progress.setValue(done)
        if s.status == "completed":
            text = f"Completed in {_duration(s.executed_wall_s)}"
            if s.mw2_dir:
                text += f" - Muscle Wrapping input: {s.mw2_dir}"
            self.status_label.setText(_colored(text, STATUS_COLORS["completed"]))
        elif s.status == "interrupted":
            how = " (process killed)" if killed else ""
            self.status_label.setText(
                _colored(f"Cancelled in '{s.failed_step}'{how}", STATUS_COLORS["interrupted"])
            )
        else:
            error = html.escape(s.error or "see the log")
            self.status_label.setText(
                _colored(f"Failed in '{s.failed_step}': ", STATUS_COLORS["failed"]) + error
            )
        self.open_mw2_button.setEnabled(s.mw2_dir is not None)
        self.qc_button.setEnabled(self._has_defects())
        self.seg_button.setEnabled(self._has_segmentation())
        if self._has_defects():
            self.warn_defects()

    # ------------------------------------------------------------------ geometry check

    def _has_defects(self) -> bool:
        from mskpipe.qc.defects import collect

        run_dir = self.process.run_dir
        return run_dir is not None and bool(collect(Path(run_dir)).defects)

    def warn_defects(self) -> None:
        """Message box with the defects of the finished run; 'Show in 3D' opens the view."""
        from mskpipe.gui.qc_dialog import warn_defects

        if warn_defects(self, Path(self.process.run_dir)):
            self.open_qc()

    def open_qc(self) -> None:
        """3D view of the defects of the last run: one window, brought to the front if it
        is open; a window of an earlier run is closed (it frees its OpenGL context)."""
        from mskpipe.gui.qc_dialog import DefectDialog

        if self.process.run_dir is None:
            return
        run_dir = Path(self.process.run_dir)
        dialog = self.qc_dialog
        if dialog is not None and dialog.run_dir != run_dir:
            self.qc_dialog = None
            dialog.close()  # deleted on close
            dialog = None
        if dialog is None:
            try:
                dialog = DefectDialog(run_dir, self)
            except ImportError as exc:  # VTK Qt support missing
                QMessageBox.warning(self, "Geometry check", f"3D view not available: {exc}")
                return
            dialog.destroyed.connect(lambda _=None, d=dialog: self._qc_dialog_gone(d))
            self.qc_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _qc_dialog_gone(self, dialog: QWidget) -> None:
        if self.qc_dialog is dialog:
            self.qc_dialog = None

    # ------------------------------------------------------------------ steps

    def _reset_steps(self, steps: Sequence[str]) -> None:
        self._steps = list(steps)
        self._step_started = {}
        self.steps_tree.clear()
        for step in api.PIPELINE_STEPS:
            item = QTreeWidgetItem([STEP_LABELS.get(step, step), "", ""])
            item.setData(0, Qt.ItemDataRole.UserRole, step)
            item.setDisabled(step not in self._steps)
            self.steps_tree.addTopLevelItem(item)
            if step in self._steps:
                self._paint(item, "pending")
        self.progress.setRange(0, max(1, len(self._steps)))
        self.progress.setValue(0)

    def _item(self, step: str) -> QTreeWidgetItem | None:
        for i in range(self.steps_tree.topLevelItemCount()):
            item = self.steps_tree.topLevelItem(i)
            if item.data(0, Qt.ItemDataRole.UserRole) == step:
                return item
        return None

    def _set_step(self, step: str, status: str, wall_s: float | None) -> None:
        item = self._item(step)
        if item is None:
            return
        if status == "running":
            self._step_started[step] = time.monotonic()
        self._paint(item, status)
        if wall_s is not None:
            item.setText(2, _duration(wall_s))
        elif status in ("cached", "pending"):
            item.setText(2, "")
        done = sum(
            1
            for i in range(self.steps_tree.topLevelItemCount())
            if self.steps_tree.topLevelItem(i).text(1) in ("completed", "cached")
        )
        self.progress.setValue(done)

    def _paint(self, item: QTreeWidgetItem, status: str) -> None:
        item.setText(1, status)
        color = QColor(STATUS_COLORS.get(status, "#000000"))
        item.setForeground(1, color)
        font = QFont(item.font(0))
        font.setBold(status == "running")
        item.setFont(0, font)
        item.setFont(1, font)

    def _tick(self) -> None:
        now = time.monotonic()
        for step, started in self._step_started.items():
            item = self._item(step)
            if item is not None and item.text(1) == "running":
                item.setText(2, _duration(now - started))
        if self._run_started is not None and self.process.is_running():
            self.statusBar().showMessage(f"Elapsed {_duration(now - self._run_started)}")

    # ------------------------------------------------------------------ helpers

    def _append_log(self, level: str, message: str) -> None:
        text = html.escape(message)
        color = LOG_COLORS.get(level.upper())
        stamp = time.strftime("%H:%M:%S")
        line = f"{stamp} {level.upper():<7} {text}"
        self.log_view.appendHtml(
            f'<span style="color:{color}">{line}</span>' if color else f"<span>{line}</span>"
        )

    def _set_running(self, running: bool) -> None:
        self._run_active = running
        for widget in (
            *self.format_buttons.values(),
            self.image_edit,
            self.modality_combo,
            self.subject_edit,
            self.device_combo,
            self.runs_edit,
            self.until_combo,
            self.verbose_check,
            self.form,
        ):
            widget.setEnabled(not running)
        self.series_combo.setEnabled(not running and self.series_combo.count() > 1)
        self.cancel_button.setEnabled(running)
        self._update_run_state()
        if self.setup_dialog is not None and not self.setup_running():
            self.setup_dialog.refresh()  # its actions are disabled while a run is active
        self.open_run_button.setEnabled(not running and self.process.run_dir is not None)
        self.open_mw2_button.setEnabled(
            not running and self.summary is not None and self.summary.mw2_dir is not None
        )
        self.qc_button.setEnabled(not running and self._has_defects())
        self.seg_button.setEnabled(not running and self._has_segmentation())

    def _open(self, path: Path | None) -> None:
        if path is not None and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def changeEvent(self, event: QEvent) -> None:
        """Back in the window (e.g. after ``mskpipe setup ...`` in a console): check again
        what a run needs."""
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self._schedule_validation()
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.setup_running():
            answer = QMessageBox.question(
                self,
                "Setup task running",
                "A download or installation is running. Cancel it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self.process.is_running():
            answer = QMessageBox.question(
                self,
                "Run in progress",
                "Cancel the run and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel_run()
            if not self.process.wait(self.process.cancel_grace_ms):
                self.process.kill()
                self.process.wait(5000)
        if self.setup_dialog is not None:
            self.setup_dialog.shutdown()  # a download must not outlive the window
        if self.slice_viewer is not None:
            self.slice_viewer.close()  # drops a read in progress and frees the volume
        if self.qc_dialog is not None:
            self.qc_dialog.close()  # frees its OpenGL context
        self._pool.waitForDone(30_000)  # a modality detection still reading the volume
        self._store()
        event.accept()


class _ModalitySignals(QObject):
    done = Signal(str, object)  # (resolved path, ModalityGuess | error message)


class _ModalityTask(QRunnable):
    def __init__(self, key: str, signals: _ModalitySignals) -> None:
        super().__init__()
        self.key = key
        self.signals = signals

    def run(self) -> None:
        try:
            result: Any = api.detect_modality(Path(self.key))
        except Exception as exc:  # any unreadable volume
            result = str(exc) or type(exc).__name__
        self.signals.done.emit(self.key, result)


class _DicomTask(QRunnable):
    """Read (convert and check) a DICOM folder; emits the DicomScan or an error message."""

    def __init__(self, key: str, signals: _ModalitySignals) -> None:
        super().__init__()
        self.key = key
        self.signals = signals

    def run(self) -> None:
        try:
            result: Any = api.scan_dicom(self.key)
        except Exception as exc:  # SetupError, or anything unexpected while converting
            result = str(exc) or type(exc).__name__
        self.signals.done.emit(self.key, result)


# ---------------------------------------------------------------------------- utilities


def _is_nifti(path: Path) -> bool:
    return path.name.lower().endswith((".nii", ".nii.gz"))


def _stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _series_text(vol: DicomVolume) -> str:
    """Series in the combo box: id, description, modality, voxels, spacing."""
    desc = f" '{vol.description}'" if vol.description else ""
    shape = " x ".join(str(n) for n in vol.shape[:3])
    spacing = " x ".join(f"{z:.3g}" for z in vol.spacing[:3])
    return f"{vol.id}{desc} - {vol.dicom_modality or '?'}, {shape}, {spacing} mm"


def _row(*widgets: QWidget, stretch: int | None = 0) -> QWidget:
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    for i, w in enumerate(widgets):
        layout.addWidget(w, 1 if (stretch is not None and i == stretch) else 0)
    return box


def _hbox(*widgets: QWidget, stretch_first: bool, stretch_at: int | None = None) -> QHBoxLayout:
    layout = QHBoxLayout()
    if stretch_first:
        layout.addStretch(1)
    for i, w in enumerate(widgets):
        if stretch_at is not None and i == stretch_at:
            layout.addStretch(1)
        layout.addWidget(w)
    return layout


def _select(combo: QComboBox, value: Any) -> None:
    i = combo.findData(value)
    if i >= 0:
        combo.setCurrentIndex(i)


def _plain(text: str) -> str:
    """Rich text of a blocker as plain text (message boxes, tooltips)."""
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def _colored(text: str, color: str) -> str:
    return f'<span style="color:{color}">{html.escape(text)}</span>'


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes} min {sec:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def main(argv: Sequence[str] | None = None) -> int:
    """Start the GUI; returns the exit code of the Qt event loop."""
    app = QApplication.instance() or QApplication(list(argv if argv is not None else sys.argv))
    app.setApplicationName("msk-pipe")
    app.setOrganizationName("KIV ZCU")
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window.check_setup)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
