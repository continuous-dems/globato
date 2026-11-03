"""Discoverable file-stage adapter for grid-native Coast Mask consumers.

Actual masking is performed once in CoastRefineFinalize, against an
unfilled, source-specific side-stack. No point filtering occurs here.
"""

from globato.coast_refine.integration import CoastMaskConsumer as _Consumer


class CoastMaskConsumer(_Consumer):
    name = "coast_mask_consumer"
