#!/usr/bin/env python

"""
globato.streams.readers.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Globato stream readers Base

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

from ..schema import ensure_schema
from .base import BaseGlobatoReader

__all__ = ["BaseGlobatoReader", "ensure_schema"]
