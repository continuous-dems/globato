"""Coast Mask policy binding for the native opt-in fresh side-stack contract.

Globato SideStackGenerate owns identity, gridding, materialization and receipts.
This adapter only freezes the Coast Mask request for later policy validation.
"""

from fetchez.utils import str2bool
from globato.hooks.sinks.side_stack import SideStackGenerate
from globato.hooks.sinks.side_stack import (
    source_processing_identity as native_source_identity,
)
from globato.coast_refine.source_grid_binding import (
    _source_request_digest,
    _source_request_signature,
)

__all__ = ["CoastMaskSideStack", "native_source_identity"]


class CoastMaskSideStack(SideStackGenerate):
    """Preserve the existing recipe name while consuming the native contract."""

    name = "coast_mask_side_stack"

    def __init__(self, *args, **kwargs):
        if "fresh" in kwargs and not str2bool(kwargs.pop("fresh")):
            raise ValueError("Coast Mask requires a fresh source side-stack")
        super().__init__(*args, fresh=True, **kwargs)

    def run(self, entries):
        for _, entry in entries:
            entry["_sidestack_context"] = {
                "request_sha256": _source_request_digest(entry),
                "request_signature": _source_request_signature(entry),
            }
        return super().run(entries)
