"""Discoverable public Coast Mask source hook.

The scientific implementation lives in ``globato.coast_refine.integration``.
Fetchez discovers third-party hooks by scanning ``globato.hooks`` modules,
so simply defining ``CoastMask`` in ``coast_refine.integration`` does not
make the recipe name ``coast_mask`` available. This thin subclass is the
registration boundary; it adds no source masking or interpolation logic.
"""

from globato.coast_refine.integration import CoastMask as _CoastMask


class CoastMask(_CoastMask):
    """Expose the shared Coast Mask implementation to the Fetchez registry."""

    name = "coast_mask"
