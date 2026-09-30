# Third-party licenses

msk-pipe itself is licensed under [Apache-2.0](LICENSE). It relies on the components below, each under its own license.

> [!IMPORTANT]
> **pystaple is licensed under CC BY-NC 4.0.** The complete pipeline and the Windows release that includes it may therefore be used for **non-commercial purposes only**.

## Core pipeline

| Component | License | Role | Source |
|---|---|---|---|
| pystaple | CC BY-NC 4.0 | Skeletal model generation (Python port of STAPLE) | [PatrikMrnka/pystaple](https://github.com/PatrikMrnka/pystaple) |
| TotalSegmentator | Apache-2.0 | Bone and muscle segmentation | [wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator) |
| nnU-Net | Apache-2.0 | Segmentation framework used by TotalSegmentator | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| MuscleMap | MIT | Muscle segmentation | [MuscleMap/MuscleMap](https://github.com/MuscleMap/MuscleMap) |
| PyTorch | BSD-3-Clause | Deep-learning runtime | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| VTK | BSD-3-Clause | Surface extraction and mesh processing | [vtk.org](https://vtk.org) |
| PySide6 | LGPL-3.0 | Graphical user interface (unmodified, shipped as separate libraries) | [qt.io](https://www.qt.io/qt-for-python) |

The Windows release additionally bundles further Python and conda packages. Their complete list and license texts are generated automatically at release time and shipped in the `licenses/` folder of the release archive.

## Demo only

| Component | License | Role |
|---|---|---|
| OpenSim | Apache-2.0 | Musculoskeletal simulation |
| Muscle Wrapping 2.x | Apache-2.0 | Muscle fibre generation |

## Not redistributed

- **Model weights** of TotalSegmentator and MuscleMap are not part of this repository or of the release. They are downloaded on first use from their official sources and are subject to their own terms.
- **TotalSegmentator `appendicular_bones`** is an optional task that requires a separate license key, free for non-commercial use ([academic license](https://backend.totalsegmentator.com/license-academic/)).
- **Datasets** (e.g. LHDL, TLEM 2.0) are not included. Obtain them from their original providers under their terms.

## Required citations

If you use msk-pipe, please also cite the tools it builds on:

- **STAPLE** (via pystaple): Modenese L., Renault J.-B. (2021). Automatic generation of personalised skeletal models of the lower limb from three-dimensional bone geometries. *Journal of Biomechanics* 116, 110186. https://doi.org/10.1016/j.jbiomech.2020.110186
- **TotalSegmentator**: Wasserthal J. et al. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiology: Artificial Intelligence*. https://doi.org/10.1148/ryai.230024
- **MuscleMap**: McKay M.J. et al. (2024). MuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle. *J Imaging* 10(11), 262. https://doi.org/10.3390/jimaging10110262

See the respective repositories for further publications of the specific algorithms and models used.
