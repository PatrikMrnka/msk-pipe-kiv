# SPDX-License-Identifier: Apache-2.0
"""Setup window of the GUI: what a run needs besides the code (``mskpipe setup ...``).

The status is collected in the GUI process (:func:`mskpipe.resources.status.collect`, file
checks only); every action - storing the TotalSegmentator licence key, downloading model
weights, installing an atlas, installing or removing GPU support - runs as a child process
``python -m mskpipe setup ...`` so that the window stays responsive, the output goes to the
log and a download can be cancelled.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mskpipe.gui.process import kill_tree, mskpipe_command
from mskpipe.resources.licence import LICENCE_URL, format_problem
from mskpipe.resources.status import SetupStatus

__all__ = ["SetupDialog", "needs_attention", "parse_progress", "weights_args"]

OK = "#1e8449"
MISSING = "#c0392b"
NOTE = "#7f8c8d"
CANCELLED = 130
ATLAS_FILTER = "Atlas (*.zip);;All files (*)"
_PROGRESS = re.compile(r"^\s*(?P<name>[\w.-]+):\s+(?P<pct>\d+)\s*%")


def parse_progress(line: str) -> tuple[str, int] | None:
    """``"  torch:  35 % of 3.05 GiB"`` (``mskpipe setup gpu``) -> ``("torch", 35)``."""
    m = _PROGRESS.match(line)
    return (m["name"], int(m["pct"])) if m else None


def weights_args(modalities: Sequence[str], overrides: Sequence[str]) -> list[str]:
    """Arguments of ``mskpipe setup weights`` for the window's settings."""
    modality = modalities[0] if len(modalities) == 1 else "all"
    args = ["setup", "weights", "--modality", modality]
    for item in overrides:
        args += ["--set", item]
    return args


def needs_attention(status: SetupStatus) -> list[str]:
    """What stops a run with the current settings (licence key, atlas); missing weights
    are not listed, they are downloaded on the first run."""
    out = []
    lic = status.licence
    if lic.needed and (not lic.present or lic.problem):
        out.append("TotalSegmentator licence key")
    if status.atlas.needed and not status.atlas.ok:
        out.append("attachment atlas")
    return out


def _colored(text: str, color: str) -> str:
    return f'<span style="color:{color}">{html.escape(text)}</span>'


class SetupDialog(QDialog):
    """``status_fn`` returns the current :class:`SetupStatus` and ``overrides_fn`` the
    ``--set`` values of the main window (both are asked again after every change);
    ``command`` starts the CLI (``python -m mskpipe``). Signals: ``use_musclemap``
    (tibia/fibula from MuscleMap, no licence key), ``gpu_changed`` (re-detect the device)."""

    use_musclemap = Signal()
    gpu_changed = Signal()

    def __init__(
        self,
        status_fn: Callable[[], SetupStatus],
        *,
        command: Sequence[str] | None = None,
        modalities: Sequence[str] = ("ct", "mri"),
        overrides_fn: Callable[[], Sequence[str]] = tuple,
        show_at_start: bool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("msk-pipe setup")
        self.resize(820, 640)
        self.status_fn = status_fn
        self.command = list(command or mskpipe_command())
        self.modalities = tuple(modalities)
        self.overrides_fn = overrides_fn
        self.status: SetupStatus | None = None
        self.last_exit: int | None = None
        self._task: str | None = None
        self._after: Callable[[], None] | None = None
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._read)
        self._proc.finished.connect(self._finished)
        self._proc.errorOccurred.connect(self._failed_to_start)
        self._buffer = b""
        self._cancelled = False
        self._build(show_at_start)
        self.refresh()

    # ------------------------------------------------------------------ layout

    def _build(self, show_at_start: bool | None) -> None:
        self.home_label = QLabel()
        self.home_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        open_home = QPushButton("Open")
        open_home.clicked.connect(self._open_home)

        # licence
        self.licence_label = QLabel()
        self.licence_label.setWordWrap(True)
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("aca_XXXXXXXXXXXXXX")
        self.key_edit.returnPressed.connect(self.save_licence)
        self.key_button = QPushButton("Save key")
        self.key_button.clicked.connect(self.save_licence)
        self.musclemap_button = QPushButton("Use MuscleMap instead")
        self.musclemap_button.setToolTip(
            "Take tibia and fibula from MuscleMap (no key needed): "
            "segmentation.tibia_fibula = musclemap"
        )
        self.musclemap_button.clicked.connect(self._use_musclemap)
        get_key = QLabel(
            f'Free for non-commercial use: <a href="{LICENCE_URL}">get a licence key</a>'
        )
        get_key.setOpenExternalLinks(True)

        # atlas
        self.atlas_label = QLabel()
        self.atlas_label.setWordWrap(True)
        self.atlas_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.atlas_button = QPushButton("Install from zip…")
        self.atlas_button.clicked.connect(self._pick_atlas)

        # weights
        self.weights_label = QLabel()
        self.weights_label.setWordWrap(True)
        self.weights_button = QPushButton("Download now")
        self.weights_button.setToolTip(
            "Download the model weights in advance (otherwise on the first run; needs internet)"
        )
        self.weights_button.clicked.connect(self.download_weights)

        # gpu
        self.gpu_label = QLabel()
        self.gpu_label.setWordWrap(True)
        self.gpu_install = QPushButton("Install (~3.5 GB)")
        self.gpu_install.clicked.connect(self.install_gpu)
        self.gpu_remove = QPushButton("Remove")
        self.gpu_remove.clicked.connect(self.remove_gpu)

        box = QGroupBox("What a run needs")
        form = QFormLayout(box)
        form.addRow("Data folder", _row(self.home_label, open_home))
        form.addRow("TotalSegmentator licence", self.licence_label)
        form.addRow("", _row(self.key_edit, self.key_button, self.musclemap_button))
        form.addRow("", get_key)
        form.addRow("Attachment atlas", _row(self.atlas_label, self.atlas_button))
        form.addRow("Model weights", _row(self.weights_label, self.weights_button))
        form.addRow("GPU support", _row(self.gpu_label, self.gpu_install, self.gpu_remove))

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.progress = QProgressBar()
        self.progress.setMinimumWidth(280)
        self.progress.setVisible(False)
        self.task_label = QLabel()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        self.at_start_check = QCheckBox("Open at start when something is missing")
        self.at_start_check.setChecked(True if show_at_start is None else show_at_start)
        self.at_start_check.setVisible(show_at_start is not None)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)

        log_box = QGroupBox("Output")
        log_layout = QVBoxLayout(log_box)
        log_layout.addWidget(self.log_view, 1)
        log_layout.addWidget(_row(self.task_label, self.progress, self.cancel_button))

        bottom = QHBoxLayout()
        bottom.addWidget(self.at_start_check)
        bottom.addStretch(1)
        bottom.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(log_box, 1)
        layout.addLayout(bottom)

    # ------------------------------------------------------------------ status

    def refresh(self) -> None:
        try:
            self.status = self.status_fn()
        except Exception as exc:  # an invalid config in the form must not break the window
            self.status = None
            self.log(f"Cannot read the setup status: {exc}")
        self._show()

    def _show(self) -> None:
        s = self.status
        busy = self._task is not None
        if s is None:
            for button in self._actions():
                button.setEnabled(False)
            return
        self.home_label.setText(s.home)

        lic = s.licence
        if lic.present and not lic.problem:
            text = _colored("set", OK)
        elif lic.problem:
            text = _colored(f"invalid: {lic.problem}", MISSING)
        else:
            text = _colored("not set", MISSING if lic.needed else NOTE)
        if not lic.needed:
            text += _colored(" - not needed with the current settings", NOTE)
        self.licence_label.setText(text)
        self.key_button.setEnabled(not busy)
        self.musclemap_button.setEnabled(lic.needed and not busy)

        atlas = s.atlas
        if not atlas.needed:
            self.atlas_label.setText(_colored("not needed with the current settings", NOTE))
        elif atlas.ok:
            source = {"config": "settings", "env": "release", "home": "installed"}
            where = f"{atlas.path} ({source.get(atlas.source, atlas.source)})"
            self.atlas_label.setText(_colored("ok", OK) + " - " + html.escape(where))
        else:
            self.atlas_label.setText(_colored("missing", MISSING))
        self.atlas_button.setEnabled(not busy)

        total = len(s.weights)
        ok = sum(w.status == "ok" for w in s.weights)
        unknown = [w for w in s.weights if w.status == "unknown"]
        if unknown:
            text = _colored(unknown[0].detail, NOTE)
        elif ok == total:
            text = _colored(f"all {total} present", OK)
        else:
            text = _colored(f"{ok} of {total} present", MISSING) + _colored(
                " - the rest is downloaded on the first run (needs internet)", NOTE
            )
        self.weights_label.setText(text)
        self.weights_label.setToolTip(
            "<br>".join(
                html.escape(
                    f"{w.tool} {w.name}: {w.status}" + (f" ({w.detail})" if w.detail else "")
                )
                for w in s.weights
            )
        )
        self.weights_button.setEnabled(not busy and not unknown and ok < total)

        gpu = s.gpu
        if gpu is None:
            self.gpu_label.setText(_colored("unknown", NOTE))
            self.gpu_install.setEnabled(False)
            self.gpu_remove.setEnabled(False)
        elif gpu.installed:
            self.gpu_label.setText(_colored(f"installed ({gpu.tag})", OK))
            self.gpu_install.setEnabled(False)
            self.gpu_remove.setEnabled(not busy and gpu.source == "overlay")
        elif not gpu.supported:
            self.gpu_label.setText(
                _colored("only in the Windows release; from the sources use pixi -e gpu", NOTE)
            )
            self.gpu_install.setEnabled(False)
            self.gpu_remove.setEnabled(False)
        else:
            self.gpu_label.setText(
                _colored(f"not installed - optional, NVIDIA driver 570+ ({gpu.tag})", NOTE)
            )
            self.gpu_install.setEnabled(not busy)
            self.gpu_remove.setEnabled(False)

    def _actions(self) -> list[QPushButton]:
        return [
            self.key_button,
            self.musclemap_button,
            self.atlas_button,
            self.weights_button,
            self.gpu_install,
            self.gpu_remove,
        ]

    # ------------------------------------------------------------------ actions

    def save_licence(self) -> bool:
        key = self.key_edit.text().strip()
        problem = format_problem(key) if key else "Enter the key"
        if problem:
            self.log(f"Licence key: {problem}")
            return False
        return self.run_task("Saving the licence key", ["setup", "licence", key])

    def _use_musclemap(self) -> None:
        self.use_musclemap.emit()
        self.log("Tibia and fibula will be taken from MuscleMap (no licence key needed).")
        self.refresh()

    def _pick_atlas(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Attachment atlas", "", ATLAS_FILTER)
        if path:
            self.install_atlas(path)

    def install_atlas(self, source: str | Path) -> bool:
        args = ["setup", "atlas", str(source)]
        if self.status is not None and Path(self.status.home, "atlas", "lhdl").exists():
            args.append("--force")
        return self.run_task("Installing the atlas", args)

    def download_weights(self) -> bool:
        return self.run_task(
            "Downloading model weights", weights_args(self.modalities, self.overrides_fn())
        )

    def install_gpu(self) -> bool:
        return self.run_task(
            "Installing GPU support", ["setup", "gpu"], after=self.gpu_changed.emit
        )

    def remove_gpu(self) -> bool:
        return self.run_task(
            "Removing GPU support", ["setup", "gpu", "--remove"], after=self.gpu_changed.emit
        )

    # ------------------------------------------------------------------ child process

    def is_running(self) -> bool:
        return self._task is not None

    def run_task(
        self, title: str, args: Sequence[str], *, after: Callable[[], None] | None = None
    ) -> bool:
        if self._task is not None:
            return False
        self._task = title
        self._after = after
        self.last_exit = None
        self._buffer = b""
        self._cancelled = False
        self.log(f"== {title}")
        self.task_label.setText(title + "…")
        self.progress.setRange(0, 0)  # busy until a percentage arrives
        self.progress.setVisible(True)
        self.cancel_button.setEnabled(True)
        self._show()
        argv = [*self.command, *args]
        self._proc.start(argv[0], argv[1:])
        return True

    def cancel(self) -> None:
        if self._task is None:
            return
        self.log("Cancelling…")
        self._cancelled = True
        pid = self._proc.processId()
        if pid:
            kill_tree(pid)
        self._proc.kill()

    def wait(self, msecs: int = 30_000) -> bool:
        return self._proc.waitForFinished(msecs)

    def _read(self) -> None:
        self._buffer += bytes(self._proc.readAllStandardOutput())
        *lines, self._buffer = self._buffer.split(b"\n")
        for raw in lines:
            self._line(raw.decode("utf-8", "replace").rstrip("\r"))

    def _line(self, text: str) -> None:
        progress = parse_progress(text)
        if progress is not None:
            name, pct = progress
            self.progress.setRange(0, 100)
            self.progress.setValue(pct)
            self.task_label.setText(f"{self._task}: {name}")
            return
        if text.strip():
            self.log(text)

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if self._buffer:
            self._line(self._buffer.decode("utf-8", "replace"))
            self._buffer = b""
        if self._cancelled:
            code = CANCELLED
        elif status == QProcess.ExitStatus.CrashExit and code == 0:
            code = 1
        self._done(code)

    def _failed_to_start(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and self._task is not None:
            self.log(f"Cannot start {self.command[0]}: {self._proc.errorString()}")
            self._done(127)

    def _done(self, code: int) -> None:
        title, after = self._task, self._after
        self.last_exit = code
        self._task, self._after = None, None
        result = {0: "done", CANCELLED: "cancelled"}.get(code, f"failed (exit code {code})")
        self.log(f"== {title}: {result}")
        self.task_label.setText("")
        self.progress.setVisible(False)
        self.cancel_button.setEnabled(False)
        if code == 0:
            self.key_edit.clear()
        if after is not None:
            after()
        self.refresh()

    # ------------------------------------------------------------------ helpers

    def log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _open_home(self) -> None:
        if self.status is not None:
            Path(self.status.home).mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.status.home))

    def closeEvent(self, event) -> None:
        self.cancel()
        super().closeEvent(event)

    def reject(self) -> None:
        self.cancel()
        super().reject()


def _row(*widgets: QWidget) -> QWidget:
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    for i, w in enumerate(widgets):
        layout.addWidget(w, 1 if i == 0 else 0)
    return box
