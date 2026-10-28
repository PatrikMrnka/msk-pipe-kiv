# Release test in Windows Sandbox

Windows Sandbox (Windows 10/11 Pro: *Turn Windows features on or off* -> *Windows Sandbox*)
is a clean Windows without Python, pixi, Visual C++ redistributables or an NVIDIA driver -
the situation of a new user. Run before every tag.

1. Download `msk-pipe-<version>-win64.zip` (draft release `ci-<run id>` of a manual
   *Release* run: `gh release download ci-<run id> -p "*.zip" -D C:\msk-pipe-sandbox`, or
   the release) and test data into `C:\msk-pipe-sandbox`: a CT `.nii.gz` and a DICOM
   folder (e.g. an MRI series).
2. Double-click `packaging\sandbox\msk-pipe.wsb`; in the sandbox copy the zip from
   `Desktop\shared` to `C:\` and unzip it there (*Extract all*).
3. Check:
   - [ ] `msk-pipe.cmd` (double-click): first start unpacks the environment, the GUI opens
         (no message about a missing `VCRUNTIME140.dll` or other DLL)
   - [ ] the Setup window opens by itself: *A run cannot start yet - missing:
         TotalSegmentator licence key, model weights*; atlas `ok (release)`
   - [ ] main window, *Run* disabled; above the buttons *The run cannot start* names the
         licence key (with *Bones = totalsegmentator*) and the missing weights
   - [ ] link *take all bones from MuscleMap*: the licence line disappears; *Settings ->
         Segmentation -> Bones = totalsegmentator* again: it is back, with the reason
   - [ ] Device offers only *CPU*, next to it *CPU only - GPU not available: ...*
         (with *install GPU support…* when an NVIDIA GPU is visible)
   - [ ] *Non-rigid* does not offer `fast_rnrr`
   - [ ] Setup… -> *Download now*: weights downloaded, the summary turns green and *Run*
         is enabled (with MuscleMap bones, or after saving a licence key)
   - [ ] a CT `.nii.gz` with *Run until: Muscle Wrapping input* completes (CPU,
         ~20-30 min); *Open Muscle Wrapping input* shows `setup_MuscleGeneratorTool.xml`,
         `.osim`, meshes; *Segmentation…* and *Geometry check…* open
   - [ ] *Image: DICOM folder*: the series are listed, the modality is detected and a run
         starts
   - [ ] `mskpipe.cmd setup status` in a console lists the weights as `ok`
   - [ ] second start of `msk-pipe.cmd` opens the GUI without unpacking
4. Optional (needs internet, a few minutes): Setup… -> Fast-RNRR -> *Build…*, confirm
   research use
   - [ ] the build finishes; *Non-rigid* now offers `fast_rnrr` and a run with it completes
5. Optional GPU test (host with an NVIDIA driver 570+ and 32 GB RAM; `msk-pipe.wsb` has
   `<VGpu>Enable</VGpu>` and `<MemoryInMB>24576</MemoryInMB>`):
   - [ ] Setup… -> GPU support -> Install; Device now offers *Auto / CPU / GPU* and the
         label shows `GPU available: cuda:0 (...)`
   - [ ] a run on the GPU completes (slower than on the host: the GPU is paravirtualised)
   With 16 GB the sandbox runs out of commit memory in TotalSegmentator `total`
   (WinError 1455 / OSError 22); a real computer grows its page file instead.
6. Note the version, date and result in the release notes / CHANGELOG.

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
