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
   - [ ] the Setup window opens by itself (no licence key); **Use MuscleMap instead** works
   - [ ] atlas shows `ok (release)`; GPU support shows *not installed* (no NVIDIA here)
   - [ ] Device *Auto* -> `cpu - auto: no NVIDIA GPU found`
   - [ ] a run with *Run until: Muscle Wrapping input* completes (CPU, ~20-30 min) and
         *Open Muscle Wrapping input* shows `setup_MuscleGeneratorTool.xml`, `.osim`, meshes
   - [ ] `mskpipe.cmd setup status` in a console lists the downloaded weights as `ok`
   - [ ] second start of `msk-pipe.cmd` opens the GUI without unpacking
4. Note the version, date and result in the release notes / CHANGELOG.
