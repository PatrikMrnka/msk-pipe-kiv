# SPDX-License-Identifier: Apache-2.0
"""Graphical user interface (PySide6). Start with ``mskpipe gui`` or ``mskpipe-gui``.

Imported only by the ``gui`` command: the CLI and the API never import Qt.
"""

from mskpipe.gui.platform_libs import preload

# Linux: libraries Qt expects from the system and the environment has to bring; before any
# module of this package imports Qt
preload()
