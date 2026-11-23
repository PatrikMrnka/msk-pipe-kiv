msk-pipe {version} for Linux (x86_64), command line
===================================================

CT or MRI (NIfTI or DICOM) -> input for Muscle Wrapping 2.x (.osim, .xml, .obj).

Start
  ./mskpipe --help
  ./mskpipe run ct.nii.gz -m ct          results in ./runs/<run>/06_mw2_input (-o to change)
  The first start unpacks the environment (1-3 min); nothing else has to be installed.
  The graphical interface is part of the Windows release only.

Needs Linux on x86_64 with glibc 2.28 or newer; tested on Ubuntu 24.04 and 26.04. If you
move this folder, the environment is unpacked again at the next start.

First run
  ./mskpipe setup status -m ct           shows what is missing
  ./mskpipe setup licence <key>          TotalSegmentator licence key (free for
                                         non-commercial use), or run with
                                         --set segmentation.bones=musclemap
  ./mskpipe setup weights -m ct          model weights (else downloaded on the first run)
  ./mskpipe setup gpu                    optional GPU support (NVIDIA driver 570+, ~4 GB)

Where data goes
  this folder                 program and environment (delete the folder to uninstall)
  ~/.local/share/msk-pipe     GPU support, MuscleMap weights, installed atlases
  ~/.totalsegmentator         TotalSegmentator weights and licence key

Licences
  msk-pipe: Apache-2.0 (LICENSE). The release includes pystaple (CC BY-NC 4.0), so it may
  be used for non-commercial purposes only (NOTICE). The attachment atlas in atlas/ is
  derived from the LHDL dataset and licensed CC BY-NC-SA 2.0 BE (atlas/lhdl/LICENSE).
  Licences of all bundled packages: licenses/.

https://github.com/kivzcu/msk-pipe
