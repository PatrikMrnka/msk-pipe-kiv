msk-pipe {version} for Windows (x64)
====================================

CT or MRI (NIfTI) -> input for Muscle Wrapping 2.x (.osim, .xml, .obj).

Start
  Double-click msk-pipe.cmd. The first start unpacks the environment (1-3 min).
  Command line: mskpipe.cmd --help   (e.g. mskpipe.cmd run ct.nii.gz -m ct)

Keep this folder at a short path (e.g. C:\msk-pipe). If you move it, the environment is
unpacked again at the next start.

First run
  - Setup... in the GUI (or mskpipe.cmd setup status) shows what is missing:
    TotalSegmentator licence key (free for non-commercial use) or "Use MuscleMap instead",
    model weights (downloaded on the first run, needs internet), optional GPU support
    (NVIDIA driver 570+, ~3.5 GB).
  - Results: Documents\msk-pipe\runs\<run>\06_mw2_input

Where data goes
  this folder                    program and environment (delete the folder to uninstall)
  %LOCALAPPDATA%\msk-pipe        GPU support, MuscleMap weights, installed atlases
  %USERPROFILE%\.totalsegmentator  TotalSegmentator weights and licence key

Licences
  msk-pipe: Apache-2.0 (LICENSE). The release includes pystaple (CC BY-NC 4.0), so it may
  be used for non-commercial purposes only (NOTICE). The attachment atlas in atlas\ is
  derived from the LHDL dataset and licensed CC BY-NC-SA 2.0 BE (atlas\lhdl\LICENSE).
  Licences of all bundled packages: licenses\.

https://github.com/kivzcu/msk-pipe
