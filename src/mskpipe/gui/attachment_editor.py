# SPDX-License-Identifier: Apache-2.0
"""Attachment editor: draw or correct the attachment outlines of a run in 3D.

Opened from the main window (Attachments…) for the last run. It starts from the outlines
the run computed (usually the atlas registration) and saves the complete set as a
folder for the method ``manual``; *Save and run* also switches the settings to that
method and starts the run again (earlier steps come from the cache).

Mouse in the 3D view: click on the bone = add a point at the end of the outline, click on
the outline between two points = insert a point there, drag a point = move it along the
bone, right click on a point = remove it; elsewhere the mouse turns, pans and zooms as
usual. Keys: Delete = remove the selected point, Ctrl+Z = undo, N / P = next / previous
area.

The dialog owns an OpenGL context, released once when it closes (as the geometry check).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mskpipe.gui.attachment_model import AttachmentSession, Key
from mskpipe.gui.muscles import pretty
from mskpipe.io.manual_areas import KINDS, AreaFile, match_folder, write_mapping

KIND_LABELS = {"Ori": "Origin", "Ins": "Insertion"}
STATE_COLORS = {
    "auto": "#7f8c8d",
    "edited": "#2471a3",
    "imported": "#7d3c98",
    "new": "#2471a3",
    "missing": "#c0392b",
}
OK_COLOR = "#1e8449"
WARN_COLOR = "#b9770e"
NONE = "—"
CLICK_PX = 4  # a press and release closer than this is a click, not a rotation
# check results as shown (Muscle Wrapping sees the outline projected onto the muscle)
PROBLEM_TEXT = {
    "collapsed": "collapses on the muscle (the export inflates it)",
    "crossing": "crosses itself on the muscle",
    "patch too large": "encloses half of the muscle or more",
}


def _problem_text(problems: list[str]) -> str:
    return ", ".join(PROBLEM_TEXT.get(p, p) for p in problems)


__all__ = ["AttachmentEditor", "ImportDialog"]


# =============================================================================== import


class ImportDialog(QDialog):
    """Assign the files of a folder (e.g. exported from NMSBuilder) to muscles and areas."""

    def __init__(
        self,
        side: str,
        muscles: list[str],
        parent: QWidget | None = None,
        folder: str = "",
        guess: Callable[[list[Path]], Any] | None = None,
    ) -> None:
        """``guess``: files -> :class:`~mskpipe.io.manual_areas.FrameGuess` (or None);
        the coordinate frame is then preselected when a folder is loaded."""
        super().__init__(parent)
        self._guess = guess
        self.setWindowTitle("Import attachment outlines")
        self.resize(820, 560)
        self.side = side
        self.muscles = sorted(muscles)
        self.folder: Path | None = None
        self._files: list[Path] = []

        self.folder_edit = QLineEdit(folder)
        self.folder_edit.setReadOnly(True)
        browse = QPushButton("Folder…")
        browse.clicked.connect(self._browse)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Muscle", "Area"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.frame_combo = QComboBox()
        self.frame_combo.addItem("RAS (meshes of the run)", "ras")
        self.frame_combo.addItem("LPS (DICOM, x and y flipped)", "lps")
        self.frame_combo.addItem("Voxel index x voxel size of the input image (no origin)", "voxel")
        self.frame_combo.setToolTip(
            "Voxel: meshes made from the voxel array alone (e.g. a raw volume "
            "in NMSBuilder/MAF); converted with the header of the run's input image"
        )
        self.frame_note = QLabel()
        self.frame_note.setWordWrap(True)
        self.frame_note.setVisible(False)
        self.order_combo = QComboBox()
        self.order_combo.addItem("as in the file", "keep")
        self.order_combo.addItem("around the centre (unordered points)", "angular")
        self.save_csv = QCheckBox("Save the assignment as areas.csv in that folder")
        self.save_csv.setChecked(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        top = QHBoxLayout()
        top.addWidget(self.folder_edit, 1)
        top.addWidget(browse)
        options = QFormLayout()
        options.addRow("Coordinates", self.frame_combo)
        options.addRow("", self.frame_note)
        options.addRow("Point order", self.order_combo)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.info)
        lay.addLayout(options)
        lay.addWidget(self.save_csv)
        lay.addWidget(self.buttons)
        if folder:
            self.load_folder(Path(folder))
        self._validate()

    def _browse(self) -> None:
        start = self.folder_edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Folder with outline files", start)
        if chosen:
            self.load_folder(Path(chosen))

    def load_folder(self, folder: Path) -> None:
        """Fill the table: recognised files preselected, the rest to be assigned."""
        self.folder = folder
        self.folder_edit.setText(str(folder))
        match = match_folder(folder, self.side)
        known = {a.path: a for a in match.areas.values()}
        dup = {p: key for key, paths in match.duplicates.items() for p in paths}
        self._files = sorted([*known, *dup, *match.unmatched])
        self.table.setRowCount(len(self._files))
        for row, path in enumerate(self._files):
            item = QTableWidgetItem(match.rel(path))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)
            muscle_box = QComboBox()
            muscle_box.addItem(NONE, None)
            for m in self.muscles:
                muscle_box.addItem(pretty(m), m)
            kind_box = QComboBox()
            for k in KINDS:
                kind_box.addItem(KIND_LABELS[k], k)
            area = known.get(path)
            key = (area.muscle, area.kind) if area else dup.get(path)
            if key is not None:
                i = muscle_box.findData(key[0])
                if i >= 0:
                    muscle_box.setCurrentIndex(i)
                else:
                    item.setToolTip(f"{key[0]}: not a meshed muscle of side {self.side}")
                kind_box.setCurrentIndex(KINDS.index(key[1]))
            muscle_box.currentIndexChanged.connect(lambda _=None: self._validate())
            kind_box.currentIndexChanged.connect(lambda _=None: self._validate())
            self.table.setCellWidget(row, 1, muscle_box)
            self.table.setCellWidget(row, 2, kind_box)
        if match.errors:
            QMessageBox.warning(self, "areas.csv", "\n".join(match.errors))
        self._validate()
        self._preselect_frame()

    def _preselect_frame(self) -> None:
        """Coordinates: the frame that puts the files onto the bones of the run."""
        if self._guess is None:
            return
        files = [a.path for a in self.assignments()] or self._files
        try:
            guess = self._guess(files)
        except Exception as exc:  # unreadable meshes etc.: leave the choice to the user
            guess = None
            self.frame_note.setText(f"Coordinates not detected: {exc}")
        if guess is not None:
            if guess.frame is not None:
                i = self.frame_combo.findData(guess.frame)
                if i >= 0:
                    self.frame_combo.setCurrentIndex(i)
                self.frame_note.setText(f"Detected: {guess.note}")
            else:
                self.frame_note.setText(
                    f"<span style='color:#b9770e'>Coordinates {guess.note}</span>"
                )
        self.frame_note.setVisible(bool(self.frame_note.text()))

    def assignments(self) -> list[AreaFile]:
        """Assigned files; files left at '—' are skipped."""
        from mskpipe.io.manual_areas import default_stem

        out = []
        for row, path in enumerate(self._files):
            muscle = self.table.cellWidget(row, 1).currentData()
            kind = self.table.cellWidget(row, 2).currentData()
            if muscle:
                out.append(AreaFile(path, muscle, kind, default_stem(muscle), "gui"))
        return out

    def _validate(self) -> bool:
        chosen = self.assignments()
        counts: dict[Key, int] = {}
        for a in chosen:
            counts[(a.muscle, a.kind)] = counts.get((a.muscle, a.kind), 0) + 1
        clash = {k for k, n in counts.items() if n > 1}
        for row in range(len(self._files)):
            muscle = self.table.cellWidget(row, 1).currentData()
            kind = self.table.cellWidget(row, 2).currentData()
            bad = (muscle, kind) in clash
            unset = muscle is None
            color = "#c0392b" if bad else ("#7f8c8d" if unset else "")
            self.table.item(row, 0).setForeground(QBrush(QColor(color or "black")))
        skipped = len(self._files) - len(chosen)
        text = f"{len(chosen)} file(s) assigned"
        if skipped:
            text += f", {skipped} skipped (—)"
        if clash:
            text += ". Two files for one area: " + ", ".join(
                f"{pretty(m)} {KIND_LABELS[k]}" for m, k in sorted(clash)
            )
        self.info.setText(text)
        ok = bool(chosen) and not clash
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)
        return ok

    def accept(self) -> None:
        if not self._validate():
            return
        if self.save_csv.isChecked() and self.folder is not None:
            try:
                write_mapping(self.folder, self.assignments())
            except OSError as exc:
                QMessageBox.warning(self, "areas.csv", f"Not saved: {exc}")
        super().accept()

    @property
    def frame(self) -> str:
        return self.frame_combo.currentData()

    @property
    def order(self) -> str:
        return self.order_combo.currentData()


# =============================================================================== editor


class AttachmentEditor(QDialog):
    #: folder saved, and whether the run should start again with it
    applied = Signal(str, bool)

    def __init__(
        self, run_dir: Path, parent: QWidget | None = None, session: AttachmentSession | None = None
    ) -> None:
        super().__init__(parent)
        from mskpipe.gui.attachment_scene import AttachmentScene

        self.run_dir = Path(run_dir)
        self.session = session or AttachmentSession.load(run_dir)
        self.released = False
        self.key: Key | None = None
        self._drag: int | None = None
        self._drag_moved = False
        self._press: tuple[int, int] | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Attachment areas - {self.run_dir.name}")
        self.resize(1360, 860)

        # ------------------------------------------------------------ left: areas
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Area", "State", "Check"])
        self.tree.setRootIsDecorated(True)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.items: dict[Key, QTreeWidgetItem] = {}
        for muscle in sorted(self.session.muscles):
            top = QTreeWidgetItem([pretty(muscle), "", ""])
            self.tree.addTopLevelItem(top)
            for kind in KINDS:
                child = QTreeWidgetItem([KIND_LABELS[kind], "", ""])
                child.setData(0, Qt.ItemDataRole.UserRole, (muscle, kind))
                top.addChild(child)
                self.items[(muscle, kind)] = child
            top.setExpanded(True)
        self.tree.currentItemChanged.connect(lambda cur, _prev: self._select_item(cur))
        import_button = QPushButton("Import…")
        import_button.setToolTip("Outline files from another program (e.g. NMSBuilder)")
        import_button.clicked.connect(self.import_files)
        origin = self.session.source_run
        if origin is None:
            text = "The run has no attachment outlines: draw them on the bones or import them."
        elif origin == self.run_dir:
            text = f"Starting point: outlines of this run ({self.session.method})."
        else:
            text = (
                f"Starting point: outlines of run {origin.name} ({self.session.method}), "
                "same meshes."
            )
        source = QLabel(text)
        source.setWordWrap(True)
        source.setStyleSheet("color:#7f8c8d")
        left = QWidget()
        llay = QVBoxLayout(left)
        llay.addWidget(self.tree, 1)
        llay.addWidget(source)
        llay.addWidget(import_button)

        # ------------------------------------------------------------ right: 3D
        self.scene = AttachmentScene(self.session.body_meshes())
        self.vtk_widget = self._view_widget()
        window = self.vtk_widget.GetRenderWindow()
        window.AddRenderer(self.scene.renderer)
        self.interactor = window.GetInteractor()
        self.style = self._make_style()
        self.interactor.SetInteractorStyle(self.style)

        self.show_muscle = QCheckBox("Muscle")
        self.show_muscle.setChecked(True)
        self.show_muscle.toggled.connect(self._toggle_muscle)
        self.show_auto = QCheckBox("Outline of the run (grey)")
        self.show_auto.setChecked(True)
        self.show_auto.toggled.connect(self._toggle_auto)
        hint = QLabel(
            "Click on the bone = add point · click on the outline = insert · drag a point = "
            "move · right click / Delete = remove · Ctrl+Z = undo · N / P = next / previous"
        )
        hint.setStyleSheet("color:#7f8c8d")
        hint.setWordWrap(True)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        self.reorder_button = QPushButton("Reorder")
        self.reorder_button.setToolTip("Order the points around their centre")
        self.reorder_button.clicked.connect(self.reorder)
        self.reset_button = QPushButton("Reset to run")
        self.reset_button.setToolTip("Back to the outline computed by the run")
        self.reset_button.clicked.connect(self.reset_area)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Remove all points of this area (draw it anew)")
        self.clear_button.clicked.connect(self.clear_area)
        self.check_label = QLabel()
        self.check_label.setWordWrap(True)

        self.folder_edit = QLineEdit(str(self.session.default_folder()))
        folder_button = QPushButton("…")
        folder_button.setFixedWidth(32)
        folder_button.setToolTip("Folder to save the outlines to")
        folder_button.clicked.connect(self._browse_folder)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(lambda: self.save(run=False))
        self.run_button = QPushButton("Save and run")
        self.run_button.setToolTip(
            "Save, use these outlines in the settings (method manual) and run again; "
            "earlier steps are reused from the cache"
        )
        self.run_button.clicked.connect(lambda: self.save(run=True))
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        view_row = QHBoxLayout()
        view_row.addWidget(self.show_muscle)
        view_row.addWidget(self.show_auto)
        view_row.addStretch(1)
        edit_row = QHBoxLayout()
        for b in (self.undo_button, self.reorder_button, self.reset_button, self.clear_button):
            edit_row.addWidget(b)
        edit_row.addWidget(self.check_label, 1)
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save to"))
        save_row.addWidget(self.folder_edit, 1)
        save_row.addWidget(folder_button)
        save_row.addWidget(self.save_button)
        save_row.addWidget(self.run_button)
        save_row.addWidget(close_button)
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addLayout(view_row)
        rlay.addWidget(self.vtk_widget, 1)
        rlay.addWidget(hint)
        rlay.addLayout(edit_row)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([360, 1000])
        outer = QVBoxLayout(self)
        outer.addWidget(split, 1)
        outer.addLayout(save_row)

        for seq, slot in (
            (QKeySequence.StandardKey.Undo, self.undo),
            (QKeySequence(Qt.Key.Key_Delete), self.remove_selected),
            (QKeySequence(Qt.Key.Key_N), lambda: self.step_area(1)),
            (QKeySequence(Qt.Key.Key_P), lambda: self.step_area(-1)),
        ):
            QShortcut(seq, self, activated=slot)

        for key in self.items:
            self._update_item(key)
        self.vtk_widget.Initialize()
        first = next(iter(self.items), None)
        if first is not None:
            self.tree.setCurrentItem(self.items[first])
        else:
            self.scene.overview()

    # ------------------------------------------------------------------ interaction

    def _view_widget(self) -> QWidget:
        """The OpenGL view (replaced in tests without a display)."""
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        return QVTKRenderWindowInteractor(self)

    def _make_style(self):
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

        style = vtkInteractorStyleTrackballCamera()
        style.AddObserver("LeftButtonPressEvent", lambda o, e: self._left_press(o))
        style.AddObserver("LeftButtonReleaseEvent", lambda o, e: self._left_release(o))
        style.AddObserver("MouseMoveEvent", lambda o, e: self._mouse_move(o))
        style.AddObserver("RightButtonPressEvent", lambda o, e: self._right_press(o))
        style.AddObserver("CharEvent", lambda o, e: None)  # no VTK key bindings (w, s, e, q)
        return style

    def _event_xy(self) -> tuple[int, int]:
        x, y = self.interactor.GetEventPosition()
        return int(x), int(y)

    def _left_press(self, style) -> None:
        x, y = self._event_xy()
        handle = self.scene.handle_at(x, y) if self.key else None
        if handle is not None:
            self._drag, self._drag_moved = handle, False
            self.scene.selected = handle
            self.refresh(check=False)
            return
        self._press = (x, y)
        style.OnLeftButtonDown()

    def _mouse_move(self, style) -> None:
        if self._drag is None or self.key is None:
            style.OnMouseMove()
            return
        hit = self.scene.pick_bone(*self._event_xy())
        if hit is None:
            return
        self.session.move_point(self.key, self._drag, hit[1], record=not self._drag_moved)
        self._drag_moved = True
        self.refresh(check=False)

    def _left_release(self, style) -> None:
        if self._drag is not None:
            self._drag = None
            self.refresh()
            return
        style.OnLeftButtonUp()
        press, self._press = self._press, None
        x, y = self._event_xy()
        if press is not None and abs(press[0] - x) + abs(press[1] - y) <= CLICK_PX:
            self.click(x, y)

    def _right_press(self, style) -> None:
        x, y = self._event_xy()
        handle = self.scene.handle_at(x, y) if self.key else None
        if handle is None:
            style.OnRightButtonDown()
            return
        self.session.remove_point(self.key, handle)
        self.scene.selected = None
        self.refresh()

    def click(self, x: float, y: float) -> None:
        """Add a point where the bone is clicked (between two points: inserted there)."""
        if self.key is None:
            return
        hit = self.scene.pick_bone(x, y)
        if hit is None:
            return
        index = self.scene.segment_at(x, y)
        i = self.session.add_point(self.key, hit[1], index)
        self.scene.selected = i
        self.refresh()

    # ------------------------------------------------------------------ actions

    def _select_item(self, item: QTreeWidgetItem | None) -> None:
        key = None if item is None else item.data(0, Qt.ItemDataRole.UserRole)
        if key is None:
            return
        self.key = tuple(key)
        self.scene.selected = None
        self._ensure_muscle(self.key[0])
        self.refresh()
        self.focus()

    def _ensure_muscle(self, muscle: str) -> None:
        if muscle not in self.scene.muscle_actors:
            from mskpipe.io.mesh_io import read_mesh

            self.scene.set_muscle_mesh(muscle, read_mesh(self.session.muscles[muscle]))

    def focus(self) -> None:
        if self.key is None:
            return
        pts = self.session.points(self.key)
        area = self.session.area(self.key)
        if not len(pts) and area is not None and area.auto is not None:
            pts = area.auto
        if not len(pts):
            other = self.session.points((self.key[0], "Ins" if self.key[1] == "Ori" else "Ori"))
            pts = other
        outward = None
        if len(pts):
            body = (area.body if area else None) or self.session.nearest_body(pts)
            surface = self.session.surfaces[body]
            outward = surface.outward(surface.closest(pts)[2])
        self.scene.focus(pts, outward)
        self._render()

    def step_area(self, delta: int) -> None:
        keys = list(self.items)
        if not keys:
            return
        i = keys.index(self.key) if self.key in keys else -delta
        self.tree.setCurrentItem(self.items[keys[(i + delta) % len(keys)]])

    def undo(self) -> None:
        key = self.session.undo()
        if key is None:
            return
        if key != self.key:
            self.tree.setCurrentItem(self.items[key])
        self.scene.selected = None
        self.refresh()

    def remove_selected(self) -> None:
        i = self.scene.selected
        if self.key is None or i is None or i >= len(self.session.points(self.key)):
            return
        self.session.remove_point(self.key, i)
        self.scene.selected = None
        self.refresh()

    def reorder(self) -> None:
        if self.key is not None and self.session.reorder(self.key):
            self.refresh()

    def reset_area(self) -> None:
        if self.key is not None and self.session.reset(self.key):
            self.scene.selected = None
            self.refresh()

    def clear_area(self) -> None:
        if self.key is not None and len(self.session.points(self.key)):
            self.session.clear(self.key)
            self.scene.selected = None
            self.refresh()

    def _toggle_muscle(self, on: bool) -> None:
        self.scene.show_muscle = on
        self.refresh(check=False)

    def _toggle_auto(self, on: bool) -> None:
        self.scene.show_auto = on
        self.refresh(check=False)

    def import_files(self) -> None:
        dialog = ImportDialog(
            self.session.side, list(self.session.muscles), self, guess=self.session.guess_frame
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            keys = self.session.import_files(dialog.assignments(), dialog.frame, dialog.order)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Import", str(exc))
            return
        for key in keys:
            self._update_item(key)
        if keys:
            self.tree.setCurrentItem(self.items[keys[0]])
        QMessageBox.information(self, "Import", f"{len(keys)} area(s) imported.")

    # ------------------------------------------------------------------ saving

    def _browse_folder(self) -> None:
        start = self.folder_edit.text() or str(self.run_dir)
        chosen = QFileDialog.getExistingDirectory(self, "Save the outlines to", start)
        if chosen:
            self.folder_edit.setText(chosen)

    def save(self, run: bool = False) -> Path | None:
        missing = [k for k in self.session.area_keys() if self.session.state(k) == "missing"]
        if missing and run:
            names = ", ".join(f"{pretty(m)} {KIND_LABELS[k]}" for m, k in missing[:6])
            more = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
            answer = QMessageBox.question(
                self,
                "Attachment areas",
                f"{len(missing)} area(s) have no outline and their muscles will not be "
                f"exported: {names}{more}. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None
        folder = Path(self.folder_edit.text().strip() or self.session.default_folder())
        try:
            self.session.save(folder)
        except OSError as exc:
            QMessageBox.warning(self, "Attachment areas", f"Not saved: {exc}")
            return None
        self.applied.emit(str(folder), run)
        if run:
            self.close()
        return folder

    # ------------------------------------------------------------------ display

    def _update_item(self, key: Key) -> None:
        item = self.items[key]
        state = self.session.state(key)
        item.setText(1, state)
        item.setForeground(1, QBrush(QColor(STATE_COLORS[state])))
        if state == "missing":
            item.setText(2, "")
            return
        problems = self.session.check(key)["problems"]
        item.setText(2, "⚠" if problems else "✔")
        item.setToolTip(2, _problem_text(problems) if problems else "ok for Muscle Wrapping")
        item.setForeground(2, QBrush(QColor(WARN_COLOR if problems else OK_COLOR)))

    def refresh(self, check: bool = True) -> None:
        """Redraw the current area (and re-run its checks unless ``check`` is False)."""
        self.undo_button.setEnabled(self.session.can_undo)
        if self.key is None:
            self._render()
            return
        muscle, kind = self.key
        area = self.session.area(self.key)
        pts = self.session.points(self.key)
        other = self.session.points((muscle, "Ins" if kind == "Ori" else "Ori"))
        body = area.body if area is not None else None
        if body is None and len(pts):
            body = self.session.nearest_body(pts)
        self.scene.show_area(muscle, body, pts, None if area is None else area.auto, other)
        self.reset_button.setEnabled(area is not None and area.auto is not None)
        self.reorder_button.setEnabled(len(pts) >= 3)
        self.clear_button.setEnabled(len(pts) > 0)
        title = f"{pretty(muscle)} - {KIND_LABELS[kind]}"
        if check:
            self._update_item(self.key)
            rep = self.session.check(self.key)
            if rep["problems"]:
                text = "⚠ " + _problem_text(rep["problems"])
                color = WARN_COLOR
            else:
                text = (
                    f"✔ {rep['n_points']} points, span {rep['span_mm']:.0f} mm, "
                    f"{rep['gap_mean_mm']:.0f} mm from the muscle"
                )
                color = OK_COLOR
            self.check_label.setText(text)
            self.check_label.setStyleSheet(f"color:{color}")
        self.scene.set_text(f"{title}  ({len(pts)} points, {body or 'no bone yet'})")
        self._render()

    def _render(self) -> None:
        if not self.released:
            self.vtk_widget.GetRenderWindow().Render()

    # ------------------------------------------------------------------ closing

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.vtk_widget.Finalize()

    def _confirm_close(self) -> bool:
        if not self.session.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Attachment areas",
            "The outlines were changed. Save them?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save(run=False) is not None
        return True

    def done(self, result: int) -> None:  # Esc
        if not self._confirm_close():
            return
        self.release()
        super().done(result)

    def closeEvent(self, event) -> None:
        if not self._confirm_close():
            event.ignore()
            return
        self.release()
        super().closeEvent(event)
