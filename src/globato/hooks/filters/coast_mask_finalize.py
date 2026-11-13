"""Registered collection boundary for standalone grid-native Coast Mask."""

from globato.coast_refine.integration import CoastRefineFinalize


class CoastMaskFinalize(CoastRefineFinalize):
    name = "coast_mask_finalize"

    def __init__(self, **kwargs):
        kwargs.setdefault("grid_native", True)
        super().__init__(**kwargs)
