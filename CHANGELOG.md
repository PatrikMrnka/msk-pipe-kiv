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
- Step `segment`: TotalSegmentator (`total`/`total_mr` by modality, `--roi_subset`, optional `appendicular_bones[_mr]`) and MuscleMap whole-body model (pinned 1.4) write raw multi-label images on the input grid plus `segmentation.json`; label maps are read from the tools (TotalSegmentator NIfTI extension, MuscleMap model JSON), never hard-coded. Both legs are segmented, so the step is reused for either side. Metrics: input grid, per-tool time and device, voxels per requested label, missing/empty labels.
- Cancellation of running tools (`ctx.cancel`): the tool and all its child processes are terminated, the step and run are recorded as `interrupted`.
- Skeleton: the STAPLE tibia body geometry is tibia + fibula of the same leg (BP convention, `skeleton.include_fibula`, merged mesh in `04_skeleton/bodies/`).
- `tools/skeleton_compare.py` (`mskpipe.validation.skeleton.compare_anatomical`): compare models built from different bones in the pelvis ACS (joint centres in mm, frame rotations in deg, markers, QC); `skeleton_parity.py` stays the same-bones parity check.
- `tools/run_steps.py`: run the implemented steps on one volume (until `mskpipe run`).
- Config `segmentation.musclemap.overlap` (90 as BP), `segmentation.musclemap.chunk_size`.

### Changed
- Step fingerprints include the input modality (segmentation depends on it); caches of earlier runs are not reused.
- `biceps_femoris_*` = MuscleMap `biceps_femoris_long_head_*` (as BP).
- Config `labelmap.*`: `closing_radius_mm`, `opening_radius_mm` (default 1.5 mm = BP SimpleITK radius 1 voxel), `min_component_fraction` replace `closing_kernel`, `opening_kernel`, `keep_largest`; `bone_subtraction_dilation_radius` removed (bones always win over muscles).

### Fixed (vs. BP pipeline)
- BP `VotingBinaryIterativeHoleFilling` grew masks instead of filling cavities; BP binarization was correct only for uint8 masks; BP cleaned masks could overlap (e.g. femur/hip).
- MuscleMap exits with code 0 on errors and stores labels with the input header (8-bit inputs scale label IDs lossily): the output is checked and 8-bit inputs get a float32 copy.
