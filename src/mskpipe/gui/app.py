# SPDX-License-Identifier: Apache-2.0
"""Main window of the msk-pipe GUI (``mskpipe gui`` / ``mskpipe-gui``).

Input volume, modality, device and output folder at the top, the configuration (built
from the config schema) on the left, progress of the steps, the log and the result on
the right. The pipeline runs as a child process (:mod:`mskpipe.gui.process`).
"""

from __future__ import annotations

import html
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
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
CONFIG_FILTER = "YAML config (*.yaml *.yml);;All files (*)"
LOG_LINES = 20_000
AUTO = api.AUTO


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
        self._modalities: dict[str, Any] = {}  # resolved image path -> ModalityGuess | error
        self._detecting: set[str] = set()
        self._pool = QThreadPool(self)
        self._modality_signals = _ModalitySignals(self)
        self._modality_signals.done.connect(self._modality_detected)

        self._build_ui()
        self._restore()
        self._set_running(False)
        self._validate()
        if probe_device:
            self.detect_device()

    # ------------------------------------------------------------------ layout

    def _build_ui(self) -> None:
        # input -------------------------------------------------------------
        self.image_edit = QLineEdit()
        self.image_edit.setPlaceholderText("CT or MRI volume (.nii / .nii.gz)")
        self.image_edit.editingFinished.connect(self._image_changed)
        browse_image = QPushButton("Browse…")
        browse_image.clicked.connect(self._browse_image)
        self.image_info = QLabel()
        self.image_info.setWordWrap(True)
        self.image_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.modality_combo = QComboBox()
        self.modality_combo.addItem("Auto (from image)", AUTO)
        self.modality_combo.addItem("CT", "ct")
        self.modality_combo.addItem("MRI", "mri")
        self.modality_combo.currentIndexChanged.connect(lambda _=None: self._show_modality())
        self.modality_label = QLabel()
        self.modality_label.setWordWrap(True)
        self.modality_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("from the file name")

        self.device_combo = QComboBox()
        for label, value in (("Auto", "auto"), ("CPU", "cpu"), ("GPU (NVIDIA)", "gpu")):
            self.device_combo.addItem(label, value)
        self.device_combo.currentIndexChanged.connect(lambda _=None: self.detect_device())
        self.device_label = QLabel()
        self.device_label.setWordWrap(True)

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
        grid.addRow(self._label("Image", "gui.image"), _row(self.image_edit, browse_image))
        grid.addRow("", self.image_info)
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
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_run)
        self.open_run_button = QPushButton("Open run folder")
        self.open_run_button.clicked.connect(lambda: self._open(self.process.run_dir))
        self.open_mw2_button = QPushButton("Open Muscle Wrapping input")
        self.open_mw2_button.setToolTip("06_mw2_input: setup XML, model, meshes, attachments")
        self.open_mw2_button.clicked.connect(
            lambda: self._open(self.summary.mw2_dir if self.summary else None)
        )

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
        right_layout.addLayout(
            _hbox(
                self.open_run_button,
                self.open_mw2_button,
                self.cancel_button,
                self.run_button,
                stretch_first=False,
                stretch_at=2,
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
        self.image_edit.setText(str(s.value("image", "")))
        _select(self.modality_combo, s.value("modality", AUTO))
        _select(self.device_combo, s.value("device", "auto"))
        self.runs_edit.setText(str(s.value("runs_dir", str(Path.cwd() / "runs"))))
        _select(self.until_combo, s.value("until", api.PIPELINE_STEPS[-1]))
        self.verbose_check.setChecked(str(s.value("verbose", "false")).lower() == "true")
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
        s.setValue("image", self.image_edit.text())
        s.setValue("modality", self.modality_combo.currentData())
        s.setValue("device", self.device_combo.currentData())
        s.setValue("runs_dir", self.runs_edit.text())
        s.setValue("until", self.until_combo.currentData())
        s.setValue("verbose", "true" if self.verbose_check.isChecked() else "false")
        s.setValue("overrides", self.form.overrides())
        s.sync()

    # ------------------------------------------------------------------ input

    def _browse_image(self) -> None:
        start = self.image_edit.text() or str(Path.cwd())
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
        text = self.image_edit.text().strip()
        self.image_info.setVisible(bool(text))
        if not text:
            self.image_info.setText("")
            self.subject_edit.setPlaceholderText("from the file name")
            self._show_modality()
            return
        path = Path(text)
        if not path.is_file():
            self.image_info.setText(_colored("File not found", "#c0392b"))
            self._show_modality()
            return
        self._detect_modality(path)
        lines = []
        try:
            lines.append(html.escape(api.describe_image(path)))
        except Exception as exc:  # any unreadable header
            lines.append(_colored(f"Cannot read the header: {exc}", "#c0392b"))
        for issue in api.check_image(path):
            color = "#c0392b" if issue.severity == "error" else "#ca6f1e"
            lines.append(_colored(issue.message, color))
        self.image_info.setText("<br>".join(lines))
        name = path.name
        for suffix in (".nii.gz", ".nii"):
            if name.lower().endswith(suffix):
                name = name[: -len(suffix)]
        self.subject_edit.setPlaceholderText(name)

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

    def _current_key(self) -> str | None:
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

    def effective_modality(self, image: Path) -> str:
        """CT/MRI chosen by the user, or detected for Auto (synchronously if not done yet)."""
        chosen = self.modality_combo.currentData()
        if chosen != AUTO:
            return chosen
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

    def detect_device(self) -> None:
        """Probe the device a run would use (``mskpipe device --json``, child process)."""
        if self._device_proc.state() != QProcess.ProcessState.NotRunning:
            self._device_proc.kill()
            self._device_proc.waitForFinished(2000)
        self.device_label.setText("detecting…")
        requested = self.device_combo.currentData()
        argv = [*self.command, "device", "--device", requested, "--json"]
        self._device_proc.start(argv[0], argv[1:])

    def _device_detected(self, code: int, _status: QProcess.ExitStatus) -> None:
        out = bytes(self._device_proc.readAllStandardOutput()).decode("utf-8", "replace")
        err = bytes(self._device_proc.readAllStandardError()).decode("utf-8", "replace")
        if code == 0:
            from mskpipe.core.device import DeviceReport

            try:
                self.device_label.setText(DeviceReport.model_validate_json(out).summary())
                return
            except ValueError:
                pass
        message = err.strip().removeprefix("Error: ") or "device detection failed"
        self.device_label.setText(_colored(message, "#c0392b"))

    # ------------------------------------------------------------------ config

    def _schedule_validation(self) -> None:
        self._validate_timer.start()

    def _validate(self) -> bool:
        ok = self.form.validate() is None
        if not self.process.is_running():
            self.run_button.setEnabled(ok)
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
        _select(self.device_combo, runtime.get("device"))
        if runtime.get("runs_dir"):
            runs = Path(runtime["runs_dir"])
            self.runs_edit.setText(str(runs if runs.is_absolute() else Path(path).parent / runs))
        self.statusBar().showMessage(f"Loaded {path}", 5000)

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
        if not image:
            raise ValueError("Choose an input volume (.nii or .nii.gz).")
        runs = self.runs_edit.text().strip()
        if not runs:
            raise ValueError("Choose an output folder.")
        until = self.until_combo.currentData()
        return ChildRequest(
            image=Path(image),
            modality=self.effective_modality(Path(image)),
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
        if not self._validate():
            QMessageBox.warning(self, "Invalid settings", self.form.error_label.text())
            return False
        try:
            child = self.child_request()
            prepared = api.prepare_run(
                api.RunRequest(
                    image=child.image,
                    modality=child.modality,
                    subject=child.subject,
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
        self.status_label.setText("Starting… (device detection, copying the input)")
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
            f"The run would probably fail:\n\n{text}\n\nStart anyway?",
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
        for widget in (
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
        self.cancel_button.setEnabled(running)
        self.run_button.setEnabled(not running and self.form.error_label.isHidden())
        self.open_run_button.setEnabled(not running and self.process.run_dir is not None)
        self.open_mw2_button.setEnabled(
            not running and self.summary is not None and self.summary.mw2_dir is not None
        )

    def _open(self, path: Path | None) -> None:
        if path is not None and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def closeEvent(self, event: QCloseEvent) -> None:
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


# ---------------------------------------------------------------------------- utilities


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
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
