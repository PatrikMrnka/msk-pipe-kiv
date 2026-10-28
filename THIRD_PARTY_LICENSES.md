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
| dcm2niix | BSD-2-Clause (bundled parts: public domain, MIT, BSD) | DICOM to NIfTI conversion of DICOM input folders (unmodified executable, conda-forge) | [rordenlab/dcm2niix](https://github.com/rordenlab/dcm2niix) |

The Windows release additionally bundles further Python and conda packages. Their complete list and license texts are generated automatically at release time and shipped in the `licenses/` folder of the release archive.

## Not included (read the output of msk-pipe)

| Component | License | Role |
|---|---|---|
| OpenSim | Apache-2.0 | Musculoskeletal simulation |
| Muscle Wrapping 2.x | Apache-2.0 | Muscle fibre generation |

## Data in the release

| Component | License | Role |
|---|---|---|
| Attachment atlas `lhdl` (derived from LHDL Cadaver #1, IOR + ULB, EU FP6 IST-2004-026932) | [CC BY-NC-SA 2.0 BE](https://creativecommons.org/licenses/by-nc-sa/2.0/be/) | Attachment areas transferred by bone registration; folder `atlas/lhdl` of the release and asset `msk-pipe-atlas-lhdl-v1.zip`, with its own `LICENSE` and `README.md` (attribution, changes). Not covered by Apache-2.0; never committed to this repository. |

## Not redistributed

- **Model weights** of TotalSegmentator and MuscleMap are not part of this repository or of the release. They are downloaded on first use from their official sources and are subject to their own terms.
- **TotalSegmentator `appendicular_bones`** is an optional task that requires a separate license key, free for non-commercial use ([academic license](https://backend.totalsegmentator.com/license-academic/)).
- **Fast-RNRR** (optional non-rigid registration, `nonrigid = fast_rnrr`) is protected under patent and may be used **for research purposes only**; its repository has no licence that allows redistribution. Neither its source nor a binary is part of this repository or the release. With the user's explicit consent `mskpipe setup fast-rnrr --accept-research-licence` (GUI: *Setup… → Fast-RNRR*) downloads the pinned source files from the [authors' repository](https://github.com/Juyong/Fast_RNRR) and builds them on the user's computer, together with OpenMesh (BSD-3-Clause), Eigen (MPL-2.0) and the Zig compiler (MIT), which are downloaded for the build only. For commercial use contact the authors (juyong@ustc.edu.cn).
- **Datasets** (e.g. LHDL, TLEM 2.0) are not included (only the derived attachment atlas above). Obtain them from their original providers under their terms.

## Required citations

If you use msk-pipe, please also cite the tools it builds on:

- **STAPLE** (via pystaple): Modenese L., Renault J.-B. (2021). Automatic generation of personalised skeletal models of the lower limb from three-dimensional bone geometries. *Journal of Biomechanics* 116, 110186. https://doi.org/10.1016/j.jbiomech.2020.110186
- **TotalSegmentator**: Wasserthal J. et al. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiology: Artificial Intelligence*. https://doi.org/10.1148/ryai.230024
- **Fast-RNRR** (when `nonrigid = fast_rnrr`): Yao Y., Deng B., Xu W., Zhang J. (2020). Quasi-Newton Solver for Robust Non-Rigid Registration. *IEEE/CVF CVPR*. https://arxiv.org/abs/2004.04322
- **dcm2niix** (DICOM input): Li X., Morgan P.S., Ashburner J., Smith J., Rorden C. (2016). The first step for neuroimaging data analysis: DICOM to NIfTI conversion. *Journal of Neuroscience Methods* 264, 47–56. https://doi.org/10.1016/j.jneumeth.2016.03.001
- **MuscleMap**: McKay M.J. et al. (2024). MuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle. *J Imaging* 10(11), 262. https://doi.org/10.3390/jimaging10110262

See the respective repositories for further publications of the specific algorithms and models used.
