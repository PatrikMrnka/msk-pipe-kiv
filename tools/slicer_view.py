# SPDX-License-Identifier: Apache-2.0
"""Show a run in 3D Slicer: input volume, bone/muscle meshes, attachment outlines.

Runs inside Slicer's Python (no mskpipe import). All pipeline geometry is world RAS in mm,
so meshes are loaded with ``coordinateSystem=RAS`` (Slicer assumes LPS for STL/OBJ).

    & "C:/Program Files/Slicer 5.8.1/Slicer.exe" --python-script tools/slicer_view.py `
        runs/<run> --atlas data/reference-GT/lhdl_atlas

Loaded (subject hierarchy folders):
    Volume          00_input/*.nii.gz
    Bones           03_mesh/bones/*.stl (beige)
    Muscles         03_mesh/muscles/<muscle>.obj of the muscles with areas (hidden)
    Registration    05_attachments/registration/*.stl - atlas bones after registration
    Attachments     05_attachments/<muscle>/*_Ori.vtk (red), *_Ins.vtk (blue), closed curves
    Atlas           --atlas: the same areas taken from the atlas without registration
                    (Ori orange, Ins cyan)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import slicer

COLORS = {
    "bone": (0.89, 0.85, 0.79),
    "muscle": (0.80, 0.30, 0.30),
    "registration": (0.30, 0.70, 0.30),
    ("run", "Ori"): (0.90, 0.10, 0.10),
    ("run", "Ins"): (0.10, 0.30, 0.95),
    ("atlas", "Ori"): (1.00, 0.60, 0.00),
    ("atlas", "Ins"): (0.00, 0.85, 0.85),
}


def read_points(path: Path) -> np.ndarray:
    tokens = path.read_text(encoding="ascii", errors="replace").split()
    i = tokens.index("POINTS")
    n = int(tokens[i + 1])
    return np.asarray(tokens[i + 3 : i + 3 + 3 * n], dtype=float).reshape(n, 3)


class Scene:
    def __init__(self) -> None:
        self.sh = slicer.mrmlScene.GetSubjectHierarchyNode()
        self.folders: dict[str, int] = {}

    def put(self, node, folder: str) -> None:
        if folder not in self.folders:
            self.folders[folder] = self.sh.CreateFolderItem(self.sh.GetSceneItemID(), folder)
        self.sh.SetItemParent(self.sh.GetItemByDataNode(node), self.folders[folder])

    def model(self, path: Path, folder: str, color, opacity=1.0, visible=True):
        node = slicer.util.loadModel(str(path), properties={"coordinateSystem": "RAS"})
        disp = node.GetDisplayNode()
        disp.SetColor(*color)
        disp.SetOpacity(opacity)
        disp.SetVisibility(visible)
        disp.SetVisibility2D(visible)
        self.put(node, folder)
        return node

    def outline(self, path: Path, name: str, folder: str, color) -> None:
        pts = read_points(path)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode", name)
        node.SetCurveTypeToLinear()
        slicer.util.updateMarkupsControlPointsFromArray(node, pts)
        node.SetLocked(True)
        disp = node.GetDisplayNode()
        disp.SetColor(*color)
        disp.SetSelectedColor(*color)
        disp.SetPointLabelsVisibility(False)
        disp.SetGlyphScale(1.0)
        disp.SetLineThickness(0.3)
        self.put(node, folder)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="slicer_view.py")
    ap.add_argument("run", type=Path, help="run folder (runs/<run>)")
    ap.add_argument("--atlas", type=Path, help="atlas folder: also show unregistered areas")
    args = ap.parse_args(argv)
    run = args.run.resolve()
    scene = Scene()

    for image in sorted((run / "00_input").glob("*.nii*"))[:1]:
        volume = slicer.util.loadVolume(str(image))
        scene.put(volume, "Volume")

    for path in sorted((run / "03_mesh" / "bones").glob("*.stl")):
        scene.model(path, "Bones", COLORS["bone"])

    att_dir = run / "05_attachments"
    index_path = att_dir / "attachments.json"
    if not index_path.is_file():
        print(f"No attachments in {run}")
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))

    for muscle in sorted(index.get("muscles", {})):
        mesh = run / "03_mesh" / "muscles" / f"{muscle}.obj"
        if mesh.is_file():
            scene.model(mesh, "Muscles", COLORS["muscle"], opacity=0.4, visible=False)

    for path in sorted((att_dir / "registration").glob("*.stl")):
        scene.model(path, "Registration", COLORS["registration"], opacity=0.3)

    atlas_index = {}
    if args.atlas:
        atlas_index = json.loads((args.atlas / "atlas.json").read_text(encoding="utf-8"))
    for area in index.get("areas", []):
        if area.get("status") != "ok":
            continue
        muscle, kind = area["muscle"], area["kind"]
        name = f"{muscle}_{kind}"
        scene.outline(att_dir / area["file"], name, "Attachments", COLORS[("run", kind)])
        ref = atlas_index.get("muscles", {}).get(muscle, {}).get(kind, {})
        if "file" in ref:
            scene.outline(
                args.atlas / ref["file"], f"atlas_{name}", "Atlas", COLORS[("atlas", kind)]
            )

    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.resetThreeDViews()
    print(f"Loaded {run}")


main(sys.argv[1:])
