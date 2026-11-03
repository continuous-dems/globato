"""Registered domain hooks; processing lives in globato.coast_refine."""

from globato.coast_refine import integration


class CoastRefine(integration.CoastRefine):
    pass


class CoastRefineCapture(integration.CoastRefineCapture):
    pass


class CoastRefineFinalize(integration.CoastRefineFinalize):
    pass


class CoastRefineReview(integration.CoastRefineReview):
    pass


__all__ = [
    "CoastRefine",
    "CoastRefineCapture",
    "CoastRefineFinalize",
    "CoastRefineReview",
]
