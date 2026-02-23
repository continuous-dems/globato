#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.processors.formats.schema
~~~~~~~~~~~~~~~~~~~

Makes sure incoming format streams make the correct rec-array

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import numpy as np
from fetchez import utils

import numpy as np
import numpy.lib.recfunctions as rfn


def ensure_schema(stream, module_weight=1.0, module_unc=0.0):
    """Generator wrapper that guarantees the stream has standard columns.

    Standard Schema:
      - x, y, z (Required)
      - w (Weight): Defaults to module_weight
      - u (Uncertainty): Defaults to module_unc. If exists, combines with module_unc.
      - classification (uint8): Defaults to 0 (Unclassified)
      - confidence (int16): Defaults to 1 (Low/Exists)
    """

    for chunk in stream:
        if chunk is None or len(chunk) == 0:
            continue

        names = chunk.dtype.names
        if not names:
            yield chunk
            continue

        new_fields = {}

        # Metadata Defaults
        if "w" not in names:
            new_fields["w"] = np.full(len(chunk), module_weight, dtype=np.float32)

        if "u" not in names:
            new_fields["u"] = np.full(len(chunk), module_unc, dtype=np.float32)

        # Classification Defaults
        if "classification" not in names:
            new_fields["classification"] = np.zeros(len(chunk), dtype=np.uint8)

        if "confidence" not in names:
            new_fields["confidence"] = np.ones(len(chunk), dtype=np.int16)

        # Append missing fields
        if new_fields:
            chunk = rfn.append_fields(
                chunk,
                names=list(new_fields.keys()),
                data=list(new_fields.values()),
                usemask=False,
                asrecarray=True
            )

        # Apply Weight
        if "w" in names:
            chunk["w"] *= module_weight

        # Apply Uncertainty
        # sqrt(point_u^2 + module_u^2)
        if module_unc > 0:
            if "u" in names:
                chunk["u"] = np.sqrt(np.square(chunk["u"]) + np.square(module_unc))

        yield chunk
