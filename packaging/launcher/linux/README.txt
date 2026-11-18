msk-pipe {version} for Linux (x86_64)
=====================================

CT or MRI (NIfTI or DICOM) -> input for Muscle Wrapping 2.x (.osim, .xml, .obj).

Start
  Graphical interface:  ./msk-pipe
  Command line:         ./mskpipe --help   (e.g. ./mskpipe run ct.nii.gz -m ct)
  The first start unpacks the environment (1-3 min); nothing else has to be installed.

Needs Linux on x86_64 with glibc 2.28 or newer; tested on Ubuntu 24.04 and 26.04. If you
move this folder, the environment is unpacked again at the next start.

First run
  - Setup... in the GUI (or ./mskpipe setup status) shows what is missing:
    TotalSegmentator licence key (free for non-commercial use) or "Use MuscleMap instead",
    model weights (downloaded on the first run, needs internet), optional GPU support
    (NVIDIA driver 570+, ~3.5 GB; or ./mskpipe setup gpu).
  - Results: ~/Documents/msk-pipe/runs/<run>/06_mw2_input (GUI); the command line writes
    to runs/ in the current folder (-o to change).

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
