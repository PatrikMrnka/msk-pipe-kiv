# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Project skeleton: pixi environments (`default`, `cpu`, `gpu`, `dev-cpu`), `mskpipe` CLI, CI on Windows.
- Step `mesh`: label map -> bone `.stl` / muscle `.obj` meshes in world coordinates (NIfTI RAS+, mm), Python VTK port of the BP `mask_to_mesh` (identical meshes with VTK 9.4.2), `meshes.json` index and per-structure metrics.
- `mskpipe.io` (NIfTI label maps, `labels.json` label table, STL/OBJ), `mskpipe.geometry` (surface extraction, mesh metrics, surface distances).
- Config `mesh.{bones,muscles}.min_component_fraction`.
