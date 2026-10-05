#!/usr/bin/env python

"""
globato
~~~~~~~~~~~~~

Initialize API and version

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

try:
    from globato._version import __version__
except ImportError:
    # Fallback when using the package from source without installing
    # in editable mode with pip (nobody should do this):
    # <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>
    import warnings

    warnings.warn(
        "Importing 'globato' outside a proper installation."
        " It's highly recommended to install the package from a stable release or"
        " in editable mode.",
        stacklevel=2,
    )
    __version__ = "dev"

# --- API ----
from .api import build, read

__all__ = ["build", "read"]
