# SPDX-License-Identifier: Apache-2.0
"""Geometry check in the GUI: warning after the run and an interactive 3D view.

:func:`warn_defects` is called by the main window when a run ends: if ``qc/defects.json``
lists defects it shows a warning (red = not fixed, the output needs attention) with a
"Show in 3D" button. :class:`DefectDialog` lists the defects; selecting one points the
camera at it (rotate/pan/zoom with the mouse, n/p = next/previous).

The dialog owns an OpenGL context. It is released once when the dialog is closed (window
button, Esc, Close) and the dialog is deleted with it: a closed but living dialog that is
repainted later makes VTK draw into the released context ("wglMakeCurrent failed").
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from mskpipe.qc.defects import collect

FIXED_COLOR = "#1e8449"
UNFIXED_COLOR = "#c0392b"

__all__ = ["DefectDialog", "warn_defects"]


def warn_defects(parent: QWidget, run_dir: Path) -> bool:
    """Warning box after a run with defects; True if the 3D view was requested."""
    report = collect(run_dir)
    if not report.defects:
        return False
    unfixed = report.unfixed
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical if unfixed else QMessageBox.Icon.Warning)
    box.setWindowTitle("Geometry check")
    lines = [d.message() for d in (unfixed or report.defects)[:8]]
    if unfixed:
        head = (
            f"{len(unfixed)} defect(s) could not be repaired. The meshes are not usable for "
            "Muscle Wrapping as they are - check them in the 3D view."
        )
    else:
        head = (
            f"{len(report.defects)} defect(s) found and repaired automatically. Repaired parts "
            "are interpolated, not measured - check them in the 3D view."
        )
    box.setText(head)
    box.setDetailedText("\n".join(d.message() + "\n    " + d.detail for d in report.defects))
    box.setInformativeText("\n".join(lines))
    show = box.addButton("Show in 3D", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Close)
    box.exec()
    return box.clickedButton() is show


class DefectDialog(QDialog):
    def __init__(self, run_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        from mskpipe.qc.viewer import load_scene

        self.run_dir = run_dir
        self.released = False  # OpenGL context of the 3D view given back
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Geometry check - {run_dir.name}")
        self.resize(1300, 820)
        self.scene = load_scene(run_dir)

        self.list = QListWidget()
        for d in self.scene.report.defects:
            item = QListWidgetItem(("✔ " if d.fixed else "✖ ") + d.message())
            item.setToolTip(d.detail)
            item.setForeground(QBrush(QColor(FIXED_COLOR if d.fixed else UNFIXED_COLOR)))
            self.list.addItem(item)
        self.list.currentRowChanged.connect(self.focus)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.context = QCheckBox("Show other structures solid")
        self.context.toggled.connect(self._context)
        overview = QPushButton("Whole model")
        overview.clicked.connect(self.overview)
        images = QPushButton("Save images")
        images.clicked.connect(self.save_images)
        folder = QPushButton("Open run folder")
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(run_dir))))
        hint = QLabel("Mouse: left = rotate, middle = pan, wheel = zoom.  Keys: n / p.")
        hint.setStyleSheet("color:#7f8c8d")

        left = QWidget()
        lay = QVBoxLayout(left)
        lay.addWidget(QLabel(f"<b>{len(self.scene.report.defects)} defect(s)</b>"))
        lay.addWidget(self.list, 1)
        lay.addWidget(self.detail)
        lay.addWidget(self.context)
        row = QHBoxLayout()
        for b in (overview, images, folder):
            row.addWidget(b)
        lay.addLayout(row)
        lay.addWidget(hint)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        window = self.vtk_widget.GetRenderWindow()
        window.AddRenderer(self.scene.renderer)
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

        self.interactor = window.GetInteractor()
        self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self.interactor.AddObserver("KeyPressEvent", self._key)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.vtk_widget)
        split.setSizes([380, 920])
        outer = QVBoxLayout(self)
        outer.addWidget(split)

        self.vtk_widget.Initialize()
        first = next((i for i, d in enumerate(self.scene.report.defects) if not d.fixed), 0)
        if self.scene.report.defects:
            self.list.setCurrentRow(first)
        else:
            self.overview()

    # ------------------------------------------------------------------ actions
    def focus(self, row: int) -> None:
        if row < 0:
            return
        self.scene.focus(row)
        self.detail.setText(self.scene.report.defects[row].detail)
        self._render()

    def overview(self) -> None:
        self.list.clearSelection()
        self.scene.overview()
        self._render()

    def save_images(self) -> None:
        from mskpipe.qc.viewer import snapshots

        files = snapshots(self.run_dir)
        QMessageBox.information(self, "Images saved", f"{len(files)} PNG files in qc/.")

    def _context(self, solid: bool) -> None:
        self.scene.context_opacity = 1.0 if solid else 0.25
        if self.scene.current is not None:
            self.scene.focus(self.scene.current)
        self._render()

    def _key(self, obj, _event) -> None:
        n = self.list.count()
        if not n:
            return
        key = obj.GetKeySym().lower()
        row = self.list.currentRow()
        if key == "n":
            self.list.setCurrentRow((row + 1) % n)
        elif key == "p":
            self.list.setCurrentRow((row - 1) % n)

    def _render(self) -> None:
        if not self.released:
            self.vtk_widget.GetRenderWindow().Render()

    def release(self) -> None:
        """Give back the OpenGL context of the 3D view (once; the dialog is closing)."""
        if not self.released:
            self.released = True
            self.vtk_widget.Finalize()

    def done(self, result: int) -> None:  # Esc and reject(): Qt eats their close event
        self.release()
        super().done(result)

    def closeEvent(self, event) -> None:  # window button, close()
        self.release()
        super().closeEvent(event)
