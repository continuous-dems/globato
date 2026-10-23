"""Seven-band FusionState streams with Stack QC rejection masks applied."""

import logging
import numpy as np
import rasterio

logger = logging.getLogger(__name__)


def masked_stream(path, mask_path=None):
    from globato.hooks.sinks.side_stack import read_fusion_state_stream

    stream = read_fusion_state_stream(path)
    if mask_path is None:
        yield from stream
        return
    with rasterio.open(mask_path) as mask:
        profile = next(stream)
        if (
            profile["height"],
            profile["width"],
            profile["transform"],
            rasterio.crs.CRS.from_user_input(profile["crs"]),
        ) != (mask.height, mask.width, mask.transform, mask.crs):
            raise ValueError("QC mask does not match the original seven-band grid")
        yield profile
        for item in stream:
            win, buffer, data, ndv, transform = item[:5]
            flags = mask.read(1, window=win)
            if data.ndim != 3 or data.shape[0] != 7 or flags.shape != data.shape[1:]:
                raise ValueError("Invalid FusionState chunk")
            if np.any((flags != 0) & (flags != 1)):
                raise ValueError("Invalid source rejection flags")
            if flags.any():
                data = data.copy()
                data[:, flags == 1] = 0.0
            yield (win, buffer, data, ndv, transform, *item[5:])
