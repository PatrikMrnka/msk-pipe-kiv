**Windows (x64):** download `msk-pipe-<version>-win64.zip`, unzip it to a short path (e.g. `C:\msk-pipe`) and double-click `msk-pipe.cmd`. The first start prepares the environment (1–3 min); nothing else has to be installed. GPU support (NVIDIA driver 570+) is installed on demand from the *Setup…* window.

**Linux (x86_64, glibc 2.28+; tested on Ubuntu 24.04 and 26.04):** download `msk-pipe-<version>-linux-x86_64.tar.gz`, unpack it (`tar -xzf`) and start `./msk-pipe` (GUI) or `./mskpipe` (command line). Nothing else has to be installed for the command line; the GUI needs a desktop system. GPU support: *Setup…* or `./mskpipe setup gpu`.

**What the release does:** CT or MRI (NIfTI or DICOM) of the pelvis and thighs → bone and muscle surfaces, OpenSim skeleton and attachment areas as input for Muscle Wrapping 2.x. Right leg only; 17 of the 20 modelled muscles are exported by default (obturator internus and externus, tensor fasciae latae: their tendons are not segmented; they can be selected again in the export settings). Attachment areas come from the atlas [`atlas-lhdl-v4`](https://github.com/kivzcu/msk-pipe/releases/tag/atlas-lhdl-v4) or are drawn by the user.

**Licences:** msk-pipe is Apache-2.0; the release includes pystaple (CC BY-NC 4.0) and may be used for non-commercial purposes only. `msk-pipe-atlas-lhdl-v4.zip` (also bundled in the release folder) is derived from the LHDL dataset and licensed CC BY-NC-SA 2.0 BE. `SHA256SUMS.txt` lists the checksums.

**Muscle Wrapping demo:** a prebuilt Muscle Wrapping 2.x to try the output is a separate asset of the release [`mw2-demo-v1`](https://github.com/kivzcu/msk-pipe/releases/tag/mw2-demo-v1); it is not part of msk-pipe.
