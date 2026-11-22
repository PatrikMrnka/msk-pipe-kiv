#!/usr/bin/env bash
# Test of the Linux release on a clean system (.github/workflows/release.yml: inside the
# Docker images ubuntu:24.04 and ubuntu:26.04, which have neither Python nor any graphics
# or X11 library, not even CA certificates).
#
#   smoke_test.sh RELEASE_FOLDER VERSION [REPO]
#
# As unpacked, nothing installed: first start, command line, libraries, relocation, and
# that the release is command line only (no PySide6; `mskpipe gui` says so).
# REPO (a checkout of msk-pipe) adds the skeleton parity check, which needs the network.
# SMOKE_LIGHT=1 skips the segmenters (to try this script with a reduced environment).
set -euo pipefail

ROOT="$(readlink -f "$1")"
VERSION="$2"
REPO="${3:-}"
PY="$ROOT/env/bin/python"
step() { printf '\n=== %s\n' "$*"; }
fail() { echo "FAILED: $*" >&2; exit 1; }

step "First start (unpacks the environment) and version"
out="$("$ROOT/mskpipe" --version)" # stdout only: the messages of the first start go to stderr
echo "$out"
[ "$out" = "mskpipe $VERSION" ] || fail "expected 'mskpipe $VERSION' on stdout, got '$out'"
[ ! -e /etc/ssl/certs/ca-certificates.crt ] && echo "(no system CA certificates: bin/cacert.pem was used)"

step "Command line"
"$ROOT/mskpipe" plugins
"$ROOT/mskpipe" device -d auto
"$ROOT/mskpipe" setup status -m ct || true
"$ROOT/mskpipe" config show >/dev/null
# (setup status exits with 1 while the licence key or the weights are missing)
("$ROOT/mskpipe" setup status --json || true) |
  "$PY" -c 'import json, sys; print("atlas:", json.load(sys.stdin)["atlas"]["source"])'

step "Tools and libraries in the environment"
"$PY" -c "import vtk; v = vtk.vtkVersion.GetVTKVersion(); print('vtk', v); assert v == '9.4.2', v"
"$PY" -c "import pystaple, scipy, nibabel, numpy; print('numpy', numpy.__version__)"
"$ROOT/env/bin/dcm2niix" -h | head -n 1
if [ -z "${SMOKE_LIGHT:-}" ]; then
  "$PY" -c "import torch, torchvision, totalsegmentator, nnunetv2, SimpleITK; from torchvision.ops import nms; print('torch', torch.__version__)"
  "$PY" -m totalsegmentator.bin.TotalSegmentator --version
  "$ROOT/env/bin/mm_segment" --help >/dev/null
fi

if [ -n "$REPO" ]; then
  step "Skeleton parity with MATLAB STAPLE"
  "$PY" "$REPO/tools/skeleton_parity.py" --fetch
fi

step "Moved folder is unpacked again (also: a path with a space)"
moved="$(dirname "$ROOT")/moved here/$(basename "$ROOT")"
mkdir -p "$(dirname "$moved")"
mv "$ROOT" "$moved"
[ "$("$moved/mskpipe" --version)" = "mskpipe $VERSION" ] || fail "relocated release does not start"
mv "$moved" "$ROOT"
"$ROOT/mskpipe" --version >/dev/null

step "Command line only: no Qt, the GUI command says so"
if "$PY" -c "import PySide6" 2>/dev/null; then fail "PySide6 is in the Linux environment"; fi
if msg="$("$ROOT/mskpipe" gui 2>&1)"; then fail "mskpipe gui did not fail"; fi
echo "$msg"
case "$msg" in *"Windows only"*) ;; *) fail "mskpipe gui: no 'Windows only' message" ;; esac

step "OK"
