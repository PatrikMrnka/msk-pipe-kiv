# SPDX-License-Identifier: Apache-2.0
import sys

from mskpipe.gui.platform_libs import missing_hint

try:
    from mskpipe.gui.app import main
except ImportError as exc:  # Linux without the graphics libraries of a desktop system
    hint = missing_hint()
    if hint is None:
        raise
    sys.exit(f"Error: the GUI cannot start ({exc}).\n{hint}")

sys.exit(main())
