# Release test in Windows Sandbox

Windows Sandbox (Windows 10/11 Pro: *Turn Windows features on or off* -> *Windows Sandbox*)
is a clean Windows without Python, pixi, Visual C++ redistributables or an NVIDIA driver -
the situation of a new user. Run before every tag.

1. Download `msk-pipe-<version>-win64.zip` (artifact `release` of the workflow run, or
   the release) and a test volume (e.g. a small CT `.nii.gz`) into `C:\msk-pipe-sandbox`.
2. Double-click `packaging\sandbox\msk-pipe.wsb`; in the sandbox copy the zip from
   `Desktop\shared` to `C:\` and unzip it there (*Extract all*).
3. Check:
   - [ ] `msk-pipe.cmd` (double-click): first start unpacks the environment, the GUI opens
         (no message about a missing `VCRUNTIME140.dll` or other DLL)
   - [ ] the Setup window opens by itself (no licence key); **Use MuscleMap instead** works
   - [ ] atlas shows `ok (release)`; GPU support shows *not installed* (no NVIDIA here)
   - [ ] Device *Auto* -> `cpu - auto: no NVIDIA GPU found`
   - [ ] a run with *Run until: Muscle Wrapping input* completes (CPU, ~20-30 min) and
         *Open Muscle Wrapping input* shows `setup_MuscleGeneratorTool.xml`, `.osim`, meshes
   - [ ] `mskpipe.cmd setup status` in a console lists the downloaded weights as `ok`
   - [ ] second start of `msk-pipe.cmd` opens the GUI without unpacking
4. Optional GPU test (host with an NVIDIA driver 570+ and 32 GB RAM): in `msk-pipe.wsb`
   set `<VGpu>Enable</VGpu>` and `<MemoryInMB>24576</MemoryInMB>`, then
   - [ ] Setup… -> GPU support -> Install; Device *GPU* -> `cuda:0 (...)`
   - [ ] a run on the GPU completes (slower than on the host: the GPU is paravirtualised)
   With 16 GB the sandbox runs out of commit memory in TotalSegmentator `total`
   (WinError 1455 / OSError 22); a real computer grows its page file instead.
5. Note the version, date and result in the release notes / CHANGELOG.

## Repeated tests without downloading again

`msk-pipe-cached.wsb` maps `C:\msk-pipe-sandbox-cache` into the sandbox as its
`%LOCALAPPDATA%\msk-pipe` (GPU support, MuscleMap, atlas) and `.totalsegmentator`
(weights, licence key), so they survive closing the sandbox:

```powershell
packaging\sandbox\prepare_cache.ps1 -FromHost      # once; -NoLicence to test the key dialog
Start-Process packaging\sandbox\msk-pipe-cached.wsb
```

Only the release itself is unpacked again in each session. This is not a first start
on a clean computer: before a tag, run the checklist above with `msk-pipe.wsb`.
