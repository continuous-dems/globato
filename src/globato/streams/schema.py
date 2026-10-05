#!/usr/bin/env python

"""
globato.streams.schema
~~~~~~~~~~~~~~~~~~~

Makes sure incoming format streams make the correct rec-array

:copyright: (c) 2010-2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def ensure_schema(stream, module_weight=1.0, module_unc=0.0):
    """Generator wrapper that guarantees the stream has standard columns.

    Standard Schema:
      - x, y, z (Required)
      - w (Weight): Defaults to module_weight
      - u (Uncertainty): Defaults to module_unc
      - classification (uint8): Defaults to 0
      - confidence (int16): Defaults to 1
    """
    required_fields = {"x", "y", "z"}

    for chunk in stream:
        if chunk is None or len(chunk) == 0:
            continue

        names = chunk.dtype.names
        if not names:
            yield chunk
            continue

        missing_required = required_fields - set(names)
        if missing_required:
            err_msg = f"Raw chunk missing required elevation fields: {missing_required}. Available: {names}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        missing_fields = []
        if "w" not in names:
            missing_fields.append(("w", "f4"))
        if "u" not in names:
            missing_fields.append(("u", "f4"))
        if "classification" not in names:
            missing_fields.append(("classification", "u1"))
        if "confidence" not in names:
            missing_fields.append(("confidence", "i2"))

        # If perfectly formatted, just apply weight/unc scaling and yield
        if not missing_fields:
            if module_weight != 1.0:
                chunk["w"] *= module_weight
            if module_unc > 0.0:
                chunk["u"] = np.sqrt(np.square(chunk["u"]) + np.square(module_unc))
            yield chunk
            continue

        new_dtype = chunk.dtype.descr + missing_fields
        new_chunk = np.empty(len(chunk), dtype=new_dtype)

        for name in names:
            new_chunk[name] = chunk[name]

        if "w" in names:
            new_chunk["w"] *= module_weight
        else:
            new_chunk["w"] = module_weight

        if "u" in names:
            if module_unc > 0.0:
                new_chunk["u"] = np.sqrt(
                    np.square(new_chunk["u"]) + np.square(module_unc)
                )
        else:
            new_chunk["u"] = module_unc

        if "classification" not in names:
            new_chunk["classification"] = 0

        if "confidence" not in names:
            new_chunk["confidence"] = 1

        yield new_chunk
