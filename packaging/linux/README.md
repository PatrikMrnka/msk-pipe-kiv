# Linux release test

`smoke_test.sh` checks an unpacked release on a system with nothing installed; the workflow
`release.yml` (job `smoke-linux`) runs it in the Docker images `ubuntu:24.04` and
`ubuntu:26.04`. The same by hand, for a package from a draft release:

```bash
gh release download <tag or ci-<run id>> -p "msk-pipe-*-linux-x86_64.tar.gz" -D dist
mkdir release && tar -xzf dist/msk-pipe-<version>-linux-x86_64.tar.gz -C release
docker run --rm -v "$PWD/release:/release" -v "$PWD:/repo:ro" ubuntu:26.04 \
    bash /repo/packaging/linux/smoke_test.sh /release/msk-pipe-<version> <version> /repo
```

Not covered by the script (no GPU and no licence key on the runners) - check before a tag
on a computer with an NVIDIA GPU (WSL 2 works):

- [ ] `./mskpipe setup gpu`, then `./mskpipe device -d gpu` selects `cuda:0`
- [ ] `./mskpipe setup licence <key>`, `./mskpipe setup weights -m ct`
- [ ] `./mskpipe run <ct.nii.gz> -m ct -d gpu` completes, `06_mw2_input` has the `.osim`,
      `.xml` and meshes; Muscle Wrapping (Windows demo) reads it
- [ ] `./msk-pipe` opens the GUI on a desktop (Wayland and X11 session), *Setup…* shows
      the atlas `ok (release)`
