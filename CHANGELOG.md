# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Project skeleton: pixi environments (`default`, `cpu`, `gpu`, `dev-cpu`), `mskpipe` CLI, CI on Windows.
- Step `mesh`: label map -> bone `.stl` / muscle `.obj` meshes in world coordinates (NIfTI RAS+, mm), Python VTK port of the BP `mask_to_mesh` (identical meshes with VTK 9.4.2), `meshes.json` index and per-structure metrics.
- `mskpipe.io` (NIfTI label maps, `labels.json` label table, STL/OBJ), `mskpipe.geometry` (surface extraction, mesh metrics, surface distances).
- Config `mesh.{bones,muscles}.min_component_fraction`.
- Step `labelmap`: raw tool segmentations (`01_segment/segmentation.json`) -> one cleaned uint8 label map (`labelmap.nii.gz`, `labels.json`, `labelmap.json` with per-structure metrics). Tool labels are mapped by name through `config/unified_labels.yaml`; TotalSegmentator `appendicular_bones` tibia/fibula are split by side; only the `skeleton.side` leg and the pelvis are built.
- Mask cleaning in mm (closing -> 3D hole filling -> opening -> components), growth only into background, one label per voxel (bones win over muscles). Validation against BP: `tools/labelmap_compare.py`, reference test `tests/steps/test_labelmap_reference.py`.

### Changed
- Config `labelmap.*`: `closing_radius_mm`, `opening_radius_mm` (default 1.5 mm = BP SimpleITK radius 1 voxel), `min_component_fraction` replace `closing_kernel`, `opening_kernel`, `keep_largest`; `bone_subtraction_dilation_radius` removed (bones always win over muscles).

### Fixed (vs. BP pipeline)
- BP `VotingBinaryIterativeHoleFilling` grew masks instead of filling cavities; BP binarization was correct only for uint8 masks; BP cleaned masks could overlap (e.g. femur/hip).
