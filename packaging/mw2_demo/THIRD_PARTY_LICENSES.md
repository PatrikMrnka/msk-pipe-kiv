# Third-party licences of the Muscle Wrapping 2.x demo

Licence texts: `licenses/<component>/`. Versions and checksums: `demo.json`.

| Files in `bin/` | Component | Licence | Source |
|---|---|---|---|
| `OsimMuscleGeneratorTool.dll`, `vtkVisDbgCore.dll` | Muscle Wrapping 2.x, vtkVisualDebugger | Apache-2.0 | University of West Bohemia; Modenese & Kohout 2020 |
| `opensim-cmd.exe`, `osim*.dll` | OpenSim 4.5 (with docopt.cpp, spdlog, Lepton) | Apache-2.0 (docopt.cpp MIT / BSL-1.0, spdlog MIT) | [opensim-org/opensim-core](https://github.com/opensim-org/opensim-core) |
| `SimTK*.dll` | Simbody 3.9 | Apache-2.0 | [simbody/simbody](https://github.com/simbody/simbody) |
| `vtk*-9.4.dll` | VTK 9.4 and its bundled libraries (expat, zlib, png, jpeg, tiff, freetype, lz4, lzma, pugixml, fmt, …) | BSD-3-Clause; bundled libraries under their own permissive licences | [vtk.org](https://vtk.org) |
| `Discregrid.dll` | Discregrid | MIT | [InteractiveComputerGraphics/Discregrid](https://github.com/InteractiveComputerGraphics/Discregrid) |
| (compiled in) | Eigen | MPL-2.0 | [libeigen/eigen](https://gitlab.com/libeigen/eigen) |
| `libblas.dll`, `liblapack.dll` | Reference BLAS / LAPACK (prebuilt, from Simbody `Platform/Windows/lib_x64`) | BSD-3-Clause | [netlib.org/lapack](https://www.netlib.org/lapack/) |
| `libgfortran-3.dll`, `libgcc_s_sjlj-1.dll` | GCC runtime of MinGW-w64 GCC 4.7.0 (snapshot 20111220), from Simbody | GPL-3.0 with GCC Runtime Library Exception 3.1 | [gcc.gnu.org](https://gcc.gnu.org) |
| `libquadmath-0.dll` | libquadmath of the same GCC | LGPL-2.1-or-later | [gcc.gnu.org](https://gcc.gnu.org) |
| `libomp140.x86_64.dll` | LLVM OpenMP runtime 22.1.8 (`libomp.dll` of conda-forge `llvm-openmp`, renamed) | Apache-2.0 WITH LLVM-exception | [llvm/llvm-project](https://github.com/llvm/llvm-project) |
| `vcruntime140*.dll`, `msvcp140*.dll` | Microsoft Visual C++ runtime 14.51 (conda-forge `vc14_runtime`) | Microsoft Visual C++ Redistributable licence | [conda-forge](https://anaconda.org/conda-forge/vc14_runtime) |

Corresponding source of the GCC runtime libraries (GPL / LGPL): the GCC source of the
version above, [gcc.gnu.org/snapshots](https://gcc.gnu.org/pub/gcc/snapshots/), and the
binaries as shipped in the Simbody repository. The ALGLIB library in the Muscle Wrapping
source tree (GPL-2.0-or-later) is not linked into these binaries (checked at packaging).

Example data (`example/`): derived from LHDL Cadaver #1 — CC BY-NC-SA 2.0 BE, see
`example/LICENSE`.
