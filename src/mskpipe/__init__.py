# SPDX-License-Identifier: Apache-2.0
"""mskpipe: CT/MRI (NIfTI) -> Muscle Wrapping 2.x input."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mskpipe")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
