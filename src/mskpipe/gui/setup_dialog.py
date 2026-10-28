# SPDX-License-Identifier: Apache-2.0
"""Setup window of the GUI: what a run needs besides the code (``mskpipe setup ...``).

The status is collected in the GUI process (:func:`mskpipe.resources.status.collect`, file
checks only); every action - storing the TotalSegmentator licence key, downloading model
weights, installing an atlas, installing or removing GPU support, building Fast-RNRR - runs
as a child process
``python -m mskpipe setup ...`` so that the window stays responsive, the output goes to the
log and a download can be cancelled.

One window per main window, for the life of the GUI: closing it while a task runs keeps
the task running in the background (reopening shows its progress), and only one task runs
at a time. Tasks are refused while a pipeline run is active (``busy_fn``), and the child
``mskpipe setup`` holds the setup lock (:mod:`mskpipe.resources.lock`) against other
msk-pipe windows and consoles.
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
    QMessageBox,
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
BUSY = 3  # mskpipe.cli.EXIT_BUSY: another msk-pipe process holds the setup lock
ATLAS_FILTER = "Atlas (*.zip);;All files (*)"
_PROGRESS = re.compile(r"^\s*(?P<name>[\w.-]+):\s+(?P<pct>\d+)\s*%")
_COMPILED = re.compile(r"^\s*compiled (?P<done>\d+)/(?P<total>\d+)")  # mskpipe setup fast-rnrr


def parse_progress(line: str) -> tuple[str, int] | None:
    """``"  torch:  35 % of 3.05 GiB"`` (``mskpipe setup gpu``) -> ``("torch", 35)``;
    ``"  compiled 3/30: x.cc"`` (``mskpipe setup fast-rnrr``) -> ``("compiling", 10)``."""
    m = _PROGRESS.match(line)
    if m:
        return m["name"], int(m["pct"])
    m = _COMPILED.match(line)
    if m and int(m["total"]):
        return "compiling", 100 * int(m["done"]) // int(m["total"])
    return None


def weights_args(modalities: Sequence[str], overrides: Sequence[str]) -> list[str]:
    """Arguments of ``mskpipe setup weights`` for the window's settings."""
    modality = modalities[0] if len(modalities) == 1 else "all"
    args = ["setup", "weights", "--modality", modality]
    for item in overrides:
        args += ["--set", item]
    return args


def needs_attention(status: SetupStatus) -> list[str]:
    """What stops a run with the current settings: the licence key (bones from
    TotalSegmentator), the attachment atlas, the model weights and Fast-RNRR when
    nonrigid = fast_rnrr. GPU support is optional (without it runs use the CPU)."""
    out = []
    lic = status.licence
    if lic.needed and (not lic.present or lic.problem):
        out.append("TotalSegmentator licence key")
    if status.atlas.needed and not status.atlas.ok:
        out.append("attachment atlas")
    if any(w.status != "ok" for w in status.weights):
        out.append("model weights")
    rnrr = status.fast_rnrr
    if rnrr is not None and rnrr.needed and not rnrr.ok:
        out.append("Fast-RNRR")
    return out


def _colored(text: str, color: str) -> str:
    return f'<span style="color:{color}">{html.escape(text)}</span>'


class SetupDialog(QDialog):
    """``status_fn`` returns the current :class:`SetupStatus` and ``overrides_fn`` the
    ``--set`` values of the main window (both are asked again after every change);
    ``command`` starts the CLI (``python -m mskpipe``); ``busy_fn`` returns why tasks are
    not allowed now (e.g. a pipeline run is active) or None. Signals: ``use_musclemap``
    (tibia/fibula from MuscleMap, no licence key), ``gpu_changed`` (re-detect the device),
    ``task_changed`` (a task started / finished)."""

    FAST_RNRR_QUESTION = (
        "{notice}\n\nmsk-pipe downloads its source from the authors' repository, OpenMesh, "
        "Eigen and the Zig compiler (~110 MB) and builds it on this computer (a few minutes).\n\n"
        "Do you use Fast-RNRR for research purposes only?"
    )

    use_musclemap = Signal()
    gpu_changed = Signal()
    task_changed = Signal(bool)

    def __init__(
        self,
        status_fn: Callable[[], SetupStatus],
        *,
        command: Sequence[str] | None = None,
        modalities: Sequence[str] = ("ct", "mri"),
        overrides_fn: Callable[[], Sequence[str]] = tuple,
        show_at_start: bool | None = None,
        busy_fn: Callable[[], str | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("msk-pipe setup")
        self.resize(820, 640)
        self.status_fn = status_fn
        self.command = list(command or mskpipe_command())
        self.modalities = tuple(modalities)
        self.overrides_fn = overrides_fn
        self.busy_fn = busy_fn or (lambda: None)
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
        self.ready_label = QLabel()  # can a run start with the current settings?
        self.ready_label.setWordWrap(True)
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
            "Take all bones from MuscleMap (no key needed): segmentation.bones = musclemap"
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
            "Download the model weights the current settings need (required before a run)"
        )
        self.weights_button.clicked.connect(self.download_weights)

        # gpu
        self.gpu_label = QLabel()
        self.gpu_label.setWordWrap(True)
        self.gpu_install = QPushButton("Install (~3.5 GB)")
        self.gpu_install.clicked.connect(self.install_gpu)
        self.gpu_remove = QPushButton("Remove")
        self.gpu_remove.clicked.connect(self.remove_gpu)

        # fast-rnrr
        self.rnrr_label = QLabel()
        self.rnrr_label.setWordWrap(True)
        self.rnrr_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.rnrr_build = QPushButton("Build…")
        self.rnrr_build.setToolTip(
            "Optional non-rigid registration of attachments (nonrigid = fast_rnrr); "
            "research use only, built on this computer"
        )
        self.rnrr_build.clicked.connect(lambda: self.build_fast_rnrr())
        self.rnrr_remove = QPushButton("Remove")
        self.rnrr_remove.clicked.connect(self.remove_fast_rnrr)

        box = QGroupBox("What a run needs")
        form = QFormLayout(box)
        form.addRow("Data folder", _row(self.home_label, open_home))
        form.addRow("TotalSegmentator licence", self.licence_label)
        form.addRow("", _row(self.key_edit, self.key_button, self.musclemap_button))
        form.addRow("", get_key)
        form.addRow("Attachment atlas", _row(self.atlas_label, self.atlas_button))
        form.addRow("Model weights", _row(self.weights_label, self.weights_button))
        form.addRow("GPU support", _row(self.gpu_label, self.gpu_install, self.gpu_remove))
        form.addRow("Fast-RNRR", _row(self.rnrr_label, self.rnrr_build, self.rnrr_remove))

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
        self.blocked_label = QLabel()
        self.blocked_label.setWordWrap(True)
        self.blocked_label.setVisible(False)
        self.at_start_check = QCheckBox("Open at start when something is missing")
        self.set_at_start(show_at_start)
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
        layout.addWidget(self.blocked_label)
        layout.addWidget(self.ready_label)
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

    def set_at_start(self, show_at_start: bool | None) -> None:
        """Show the 'open at start' check box (opened at start) with its value, or hide it."""
        self.at_start_check.setChecked(True if show_at_start is None else show_at_start)
        self.at_start_check.setVisible(show_at_start is not None)

    def blocked_reason(self) -> str | None:
        """Why no task may start now (another task here, or a pipeline run)."""
        if self._task is not None:
            return f"{self._task} is running"
        return self.busy_fn()

    def _show(self) -> None:
        s = self.status
        blocked = self.busy_fn()
        self.blocked_label.setText(
            _colored(f"{blocked} - setup changes are possible when it has finished.", MISSING)
            if blocked
            else ""
        )
        self.blocked_label.setVisible(bool(blocked))
        busy = self._task is not None or blocked is not None
        if s is None:
            for button in self._actions():
                button.setEnabled(False)
            self.ready_label.setText(
                _colored("Cannot read the setup status (see Output).", MISSING)
            )
            return
        self.home_label.setText(s.home)
        missing = needs_attention(s)
        if missing:
            self.ready_label.setText(
                _colored(f"A run cannot start yet - missing: {', '.join(missing)}.", MISSING)
                + _colored(" GPU support and Fast-RNRR are optional.", NOTE)
            )
        else:
            self.ready_label.setText(
                _colored("Everything a run needs is set up.", OK)
                + _colored(" GPU support and Fast-RNRR are optional.", NOTE)
            )

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
                " - required: download them now (needs internet)", NOTE
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
                _colored(
                    f"not installed - optional, runs use the CPU; for NVIDIA driver 570+ "
                    f"({gpu.tag})",
                    NOTE,
                )
            )
            self.gpu_install.setEnabled(not busy)
            self.gpu_remove.setEnabled(False)
        self._show_fast_rnrr(busy)

    def _show_fast_rnrr(self, busy: bool) -> None:
        rnrr = self.status.fast_rnrr if self.status is not None else None
        build = rnrr.build if rnrr is not None else None
        if rnrr is None or build is None:
            self.rnrr_label.setText(_colored("unknown", NOTE))
            self.rnrr_build.setEnabled(False)
            self.rnrr_remove.setEnabled(False)
            return
        if rnrr.ok and (rnrr.source != "build" or build.built):
            where = "built here" if rnrr.source == "build" else rnrr.source
            text = _colored("ok", OK) + " - " + html.escape(f"{rnrr.path} ({where})")
        elif rnrr.needed:
            text = _colored("missing", MISSING) + _colored(
                " - needed by nonrigid = fast_rnrr; research use only", NOTE
            )
        else:
            text = _colored(
                "not built - optional, research use only; nonrigid = fast_rnrr is offered "
                "once it is built",
                NOTE,
            )
        if not build.supported:
            text += _colored(" (building is not supported on this system)", NOTE)
        self.rnrr_label.setText(text)
        self.rnrr_build.setEnabled(not busy and build.supported and not build.built)
        self.rnrr_remove.setEnabled(not busy and build.built)

    def _actions(self) -> list[QPushButton]:
        return [
            self.key_button,
            self.musclemap_button,
            self.atlas_button,
            self.weights_button,
            self.gpu_install,
            self.gpu_remove,
            self.rnrr_build,
            self.rnrr_remove,
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
        self.log("All bones will be taken from MuscleMap (no licence key needed).")
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

    def build_fast_rnrr(self, *, confirmed: bool = False) -> bool:
        """Ask for the research-use confirmation, then build (``mskpipe setup fast-rnrr``)."""
        if not confirmed:
            from mskpipe.resources.fast_rnrr import load_spec

            answer = QMessageBox.question(
                self,
                "Fast-RNRR - research use only",
                self.FAST_RNRR_QUESTION.format(notice=load_spec().notice),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.log("Fast-RNRR: not built (research use not confirmed).")
                return False
        return self.run_task(
            "Building Fast-RNRR", ["setup", "fast-rnrr", "--accept-research-licence"]
        )

    def remove_fast_rnrr(self) -> bool:
        return self.run_task("Removing Fast-RNRR", ["setup", "fast-rnrr", "--remove"])

    # ------------------------------------------------------------------ child process

    def is_running(self) -> bool:
        return self._task is not None

    def run_task(
        self, title: str, args: Sequence[str], *, after: Callable[[], None] | None = None
    ) -> bool:
        reason = self.blocked_reason()
        if reason is not None:
            self.log(f"Not started: {reason}.")
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
        self.task_changed.emit(True)
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
        result = {
            0: "done",
            CANCELLED: "cancelled",
            BUSY: "not started, another msk-pipe window or console is changing the setup",
        }.get(code, f"failed (exit code {code})")
        self.log(f"== {title}: {result}")
        self.task_label.setText("")
        self.progress.setVisible(False)
        self.cancel_button.setEnabled(False)
        if code == 0:
            self.key_edit.clear()
        if after is not None:
            after()
        self.refresh()
        self.task_changed.emit(False)

    # ------------------------------------------------------------------ helpers

    def log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _open_home(self) -> None:
        if self.status is not None:
            Path(self.status.home).mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.status.home))

    def done(self, result: int) -> None:
        """Close / Esc / the window's X: a running task keeps running in the background."""
        if self._task is not None:
            self.log(f"{self._task} continues in the background; reopen Setup… to follow it.")
        super().done(result)

    def shutdown(self, msecs: int = 10_000) -> None:
        """Cancel a running task and wait for its process (the GUI is closing)."""
        if self._task is not None:
            self.cancel()
            self.wait(msecs)


def _row(*widgets: QWidget) -> QWidget:
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    for i, w in enumerate(widgets):
        layout.addWidget(w, 1 if i == 0 else 0)
    return box
