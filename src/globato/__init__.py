#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato
~~~~~~~~~~~~~

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import os
import inspect
import importlib
import logging

from fetchez.hooks.registry import HookRegistry
from fetchez.registry import FetchezRegistry
from fetchez.hooks import FetchHook

# --- Custom fetchez modules ---
from .modules.local_fs import LocalFS
from .modules.gebco import GEBCO_COG
from .modules.glob_dem import GlobDEM
from .modules.glob_coast import GlobCoast
from .modules.sources import GlobCopernicus, GlobFabDEM, GlobMultibeam, GlobBAG, GlobNOSXYZ

logger = logging.getLogger(__name__)
__version__ = "0.1.4"

def _auto_register_hooks():
    """Recursively scan the 'processors' directory and auto-register all FetchHooks."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processors_dir = os.path.join(current_dir, "processors")

    if not os.path.exists(processors_dir):
        return

    for root, dirs, files in os.walk(processors_dir):
        dirs[:] = [d for d in dirs if not d.startswith('_')]

        for f in files:
            if f.endswith(".py") and not f.startswith("_"):
                rel_dir = os.path.relpath(root, current_dir)
                mod_path = rel_dir.replace(os.sep, '.')
                mod_name = f[:-3]

                full_mod_name = f"globato.{mod_path}.{mod_name}"

                try:
                    mod = importlib.import_module(full_mod_name)
                    for name, obj in inspect.getmembers(mod):
                        if (inspect.isclass(obj) and
                            issubclass(obj, FetchHook) and
                            obj is not FetchHook):
                            HookRegistry.register_hook(obj)
                except Exception as e:
                    logger.warning(f"Failed to auto-load globato hook {full_mod_name}: {e}")


def setup_fetchez(registry_cls):
    """Register All globato capabilities with Fetchez."""

    _auto_register_hooks()

    registry_cls.register_module(
        'local_fs',
        LocalFS,
        metadata={
            'desc': 'Crawl, spatially filter, and process local directories of data.',
            'tags': ['local', 'datalist', 'folder', 'inf', 'cudem', 'globato'],
            'category': 'Globato',
        }
    )
    registry_cls.register_module(
        'gebco_cog',
        GEBCO_COG,
        metadata={
            "inherits": "gebco",
            'desc': 'Fetch GEBCO as a COG subset',
            'tags': ['gebco', 'bathymetry', 'global', 'tid', 'cog'],
            'category': 'Globato',
        }
    )
    registry_cls.register_module(
        'glob_dem',
        GlobDEM,
        metadata={
            'desc': 'Fetch and glob the best available DEMs',
            'tags': ['gebco', 'bathymetry', 'global', 'etopo', 'globato'],
            'category': 'Tools',
        }
    )
    registry_cls.register_module(
        'glob_coast',
        GlobCoast,
        metadata={
            'desc': 'Fetch and glob a coastline',
            'tags': ['global', 'globato', 'coastline', 'landmask'],
            'category': 'Tools',
        }
    )
    registry_cls.register_module(
        'copernicus_glob',
        GlobCopernicus,
        metadata={
            "inherits": "copernicus",
            "desc": "Copernicus Global/European Digital Elevation Models (COP-30/10)",
            "tags": ["satellite", "dsm", "radar", "global", "europe", "clean", "globato"],
            'category': 'Globato',
        }
    )
    registry_cls.register_module(
        'fabdem_glob',
        GlobFabDEM,
        metadata={
            "inherits": "fabdem",
            "tags": ["fabdem", "dem", "dtm", "copernicus", "global", "30m", "clean", "globato"],
            'category': 'Globato',
        }
    )
    registry_cls.register_module(
        'multibeam_glob',
        GlobMultibeam,
        metadata={
            "inherits": "multibeam",
            "tags": ["bathymetry", "multibeam", "ocean", "sonar", "noaa", "ncei", "globato"],
            'category': 'Globato',
        }
    )
    registry_cls.register_module(
        'bag_glob',
        GlobBAG,
        metadata={
            "inherits": "nos_hydro",
            "tags": ["bathymetry", "hydrography", "nos", "noaa", "bag", "soundings", "globato"],
            'category': 'Globato',
        }
    )
    registry_cls.register_module(
        'nos_xyz_glob',
        GlobNOSXYZ,
        metadata={
            "inherits": "nos_hydro",
            "tags": ["bathymetry", "nos", "noaa", "xyz", "legacy", "globato"],
            'category': 'Globato',
        }
    )

setup_fetchez(FetchezRegistry)
