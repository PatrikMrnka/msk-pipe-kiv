# msk-pipe release: prepare the environment (sourced by msk-pipe and mskpipe; bash).
# First start (or after the folder was moved): unpack environment.tar into env/.
# Sets MSKPIPE_ROOT, MSKPIPE_PYTHON and the variables of the release. Messages go to
# stderr, so the output of `mskpipe ... > file` stays clean.

MSKPIPE_ROOT="$(cd -P "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
MSKPIPE_ENV="$MSKPIPE_ROOT/env"
MSKPIPE_PYTHON="$MSKPIPE_ENV/bin/python"
MSKPIPE_VERSION="$(head -n 1 "$MSKPIPE_ROOT/version.txt" | tr -d '\r\n')"

mskpipe_fail() {
  echo "ERROR: $*" >&2
  return 1
}

mskpipe_unpacked() {
  # the environment is bound to its folder (conda prefix): unpack again after a move
  [ -f "$MSKPIPE_ENV/.mskpipe-unpacked" ] &&
    [ "$(cat "$MSKPIPE_ENV/.mskpipe-unpacked")" = "$MSKPIPE_VERSION*$MSKPIPE_ROOT" ]
}

mskpipe_check_system() {
  if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    mskpipe_fail "this release is for Linux on x86_64 (found $(uname -s) $(uname -m))."
    return 1
  fi
  # the environment needs glibc 2.28 or newer (Ubuntu 20.04, Debian 10, RHEL 8)
  local libc
  libc="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
  if [ -z "$libc" ] || [ "$(printf '%s\n' "2.28" "$libc" | sort -V | head -n 1)" != "2.28" ]; then
    mskpipe_fail "glibc 2.28 or newer is needed (found: ${libc:-none})."
    return 1
  fi
}

mskpipe_unpack() {
  echo "msk-pipe $MSKPIPE_VERSION: preparing the environment (first start, 1-3 minutes) ..." >&2
  mskpipe_check_system || return 1
  [ -w "$MSKPIPE_ROOT" ] ||
    { mskpipe_fail "cannot write to \"$MSKPIPE_ROOT\" - put msk-pipe into a folder of your own."; return 1; }
  rm -rf "$MSKPIPE_ENV" "$MSKPIPE_ROOT/activate.sh" ||
    { mskpipe_fail "cannot remove \"$MSKPIPE_ENV\" - close msk-pipe and try again."; return 1; }
  # pixi-unpack sets up a TLS client although it downloads nothing, and stops on a system
  # without CA certificates (minimal containers): it gets the bundle of the release
  SSL_CERT_FILE="${SSL_CERT_FILE:-$MSKPIPE_ROOT/bin/cacert.pem}" \
    "$MSKPIPE_ROOT/bin/pixi-unpack" "$MSKPIPE_ROOT/environment.tar" -o "$MSKPIPE_ROOT" -e env -s bash -q >&2 ||
    { mskpipe_fail "unpacking environment.tar failed. Is there enough free disk space (~5 GB)?"; return 1; }
  [ -x "$MSKPIPE_PYTHON" ] ||
    { mskpipe_fail "the environment has no bin/python after unpacking."; return 1; }
  printf '%s\n' "$MSKPIPE_VERSION*$MSKPIPE_ROOT" >"$MSKPIPE_ENV/.mskpipe-unpacked"
  echo "Environment ready: $MSKPIPE_ENV" >&2
}

if ! mskpipe_unpacked; then
  # one start at a time unpacks; a second one waits and finds the environment ready
  if command -v flock >/dev/null 2>&1 && [ -w "$MSKPIPE_ROOT" ]; then
    exec 9>"$MSKPIPE_ROOT/.unpack.lock"
    flock 9
    mskpipe_unpacked || mskpipe_unpack || return 1
    exec 9>&-
  else
    mskpipe_unpack || return 1
  fi
fi

# activate (not with the activate.sh of pixi-unpack: it breaks on a path with spaces)
export PATH="$MSKPIPE_ENV/bin:$PATH" CONDA_PREFIX="$MSKPIPE_ENV"
# the release is self-contained: no user site-packages or foreign PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUTF8=1
unset PYTHONPATH PYTHONHOME
if [ -z "${MSKPIPE_ATLAS_DIR:-}" ] && [ -f "$MSKPIPE_ROOT/atlas/lhdl/atlas.json" ]; then
  export MSKPIPE_ATLAS_DIR="$MSKPIPE_ROOT/atlas/lhdl"
fi
export MSKPIPE_ROOT MSKPIPE_PYTHON MSKPIPE_VERSION
