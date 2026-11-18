#!/usr/bin/env bash
# Test of the Linux release on a clean system (.github/workflows/release.yml: inside the
# Docker images ubuntu:24.04 and ubuntu:26.04, which have neither Python nor any graphics
# or X11 library, not even CA certificates).
#
#   smoke_test.sh RELEASE_FOLDER VERSION [REPO]
#
# 1. as unpacked, nothing installed: first start, command line, libraries, relocation;
# 2. the GUI names the system packages it misses; with them installed (apt) the main
#    window opens on X11 (Xvfb) using the X11 libraries of the release.
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

packages=""
if [ -n "$("$PY" -c 'from mskpipe.gui.platform_libs import missing_hint; print(missing_hint() or "")')" ]; then
  step "GUI without the graphics libraries of a desktop: names the packages"
  if hint="$(DISPLAY=:99 "$ROOT/msk-pipe" 2>&1)"; then fail "the GUI started without a display"; fi
  echo "$hint"
  packages="$(echo "$hint" | sed -n 's/^ *sudo apt install //p')"
  [ -n "$packages" ] || fail "no 'sudo apt install ...' line in the message"
fi

step "GUI with them installed: main window on X11 (Xvfb)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null
# shellcheck disable=SC2086
apt-get install -y -q --no-install-recommends $packages xvfb xauth fonts-dejavu-core >/dev/null
QT_QPA_PLATFORM=offscreen "$PY" -c "import mskpipe.gui.app; from PySide6.QtWidgets import QApplication; QApplication([]); print('Qt starts (offscreen)')"
"$ROOT/mskpipe" gui --help >/dev/null
probe="$(mktemp --suffix=.py)"
cat >"$probe" <<'PYEOF'
import sys

from mskpipe.gui import platform_libs  # importing mskpipe.gui preloads the libraries

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from mskpipe.gui.app import MainWindow

app = QApplication(sys.argv)
window = MainWindow()
window.show()


def done():
    image = window.grab()
    print("platform:", app.platformName(), "window:", image.width(), "x", image.height())
    print("from the release:", " ".join(platform_libs.preload()) or "-")
    app.exit(0 if app.platformName() == "xcb" and window.isVisible() else 1)


QTimer.singleShot(1500, done)
sys.exit(app.exec())
PYEOF
(cd "$(mktemp -d)" && QT_QPA_PLATFORM=xcb xvfb-run -a "$PY" "$probe") || fail "the main window did not open on X11"

step "OK"
