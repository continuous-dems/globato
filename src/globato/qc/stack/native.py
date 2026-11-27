"""Seven-band FusionState streams with Stack QC rejection masks applied."""

import logging
import numpy as np
import rasterio

logger = logging.getLogger(__name__)


def _fusion_blocks(path, active_blocks, offset):
    """Like read_fusion_state_stream, reading only blocks where the source has data.

    ``active_blocks`` are "row,col" keys of 256-cell blocks on the QC stack grid
    (from the first working-stack build) and ``offset`` is the source's (row,
    col) offset on that grid. Blocks outside them hold only zeros.
    """
    active = {tuple(map(int, key.split(","))) for key in active_blocks}
    dy, dx = offset
    with rasterio.open(path) as src:
        yield src.profile.copy()
        for _, window in src.block_windows(1):
            y0, x0 = int(window.row_off) + dy, int(window.col_off) + dx
            y1, x1 = y0 + int(window.height) - 1, x0 + int(window.width) - 1
            if not any(
                (by, bx) in active
                for by in range(y0 // 256, y1 // 256 + 1)
                for bx in range(x0 // 256, x1 // 256 + 1)
            ):
                continue
            data = src.read(window=window).astype(np.float64, copy=False)
            yield window, window, data, None, src.window_transform(window)


def masked_stream(path, mask_path=None, *, active_blocks=None, source_offset=(0, 0)):
    """A FusionState stream with rejected cells zeroed.

    With ``active_blocks``, chunks without source data are skipped (they are
    all zeros), so only the blocks with measurements are read.
    """
    from globato.hooks.sinks.side_stack import read_fusion_state_stream

    stream = (
        read_fusion_state_stream(path)
        if active_blocks is None
        else _fusion_blocks(path, active_blocks, source_offset)
    )
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
