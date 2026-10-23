"""Internal native-grid cache and window utilities; no provider or region defaults."""

import hashlib
import json
import os
import logging
import tempfile
from pathlib import Path
import numpy as np


def log(s):
    logging.getLogger(__name__).info(s)


def read(p):
    return json.loads(Path(p).read_text())


def write(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=p.parent, prefix=p.name + ".", suffix=".partial", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(d, stream, indent=2, sort_keys=True, allow_nan=False)
        os.replace(temporary, p)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(d):
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def fingerprint(p):
    s = Path(p).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns)


def grid(ds):
    return dict(shape=list(ds.shape), crs=str(ds.crs), affine=list(ds.transform)[:6])


def windows(shape, core=256, halo=0):
    for y in range(0, shape[0], core):
        for x in range(0, shape[1], core):
            a = (slice(y, min(y + core, shape[0])), slice(x, min(x + core, shape[1])))
            b = (
                slice(max(0, y - halo), min(shape[0], a[0].stop + halo)),
                slice(max(0, x - halo), min(shape[1], a[1].stop + halo)),
            )
            inner = tuple(
                slice(i.start - j.start, i.stop - j.start) for i, j in zip(a, b)
            )
            yield a, b, inner


def raster_window(s):
    from rasterio.windows import Window

    return Window(
        s[1].start, s[0].start, s[1].stop - s[1].start, s[0].stop - s[0].start
    )


def mm(path, shape, dtype="float32", fill=0):
    a = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    a[:] = fill
    return a
