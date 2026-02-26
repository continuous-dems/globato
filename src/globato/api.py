#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
globato.api
~~~~~~~~~~~
High-level Python API for Globato.
Provides a fluent interface for streaming, processing, and accessing geospatial data
without needing to construct full Fetchez pipelines.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Union, List, Iterator, Optional

from globato.processors.formats.stream_factory import StreamFactory
from globato.processors.formats.schema import ensure_schema
from fetchez.core import FetchModule

logger = logging.getLogger(__name__)


class GlobatoStream:
    """A wrapper around a data stream generator.
    Allows for chaining processing steps (fluent API).
    """

    def __init__(self, iterator: Iterator[np.ndarray], src_srs: str = "EPSG:4326"):
        self._iterator = iterator
        self.src_srs = src_srs

    def __iter__(self):
        """Yields chunks of numpy rec-arrays."""

        yield from self._iterator

    def map(self, func, **kwargs):
        """Apply an arbitrary function to the stream.
        func(chunk, **kwargs) -> chunk
        """

        def _wrapper():
            for chunk in self._iterator:
                if chunk is not None and len(chunk) > 0:
                    yield func(chunk, **kwargs)

        self._iterator = _wrapper()
        return self

    def reproject(self, dst_srs: str):
        """Injects a reprojection step into the stream."""

        from globato.processors.transforms.reproject import stream_reproject_chunk

        def _repro_func(chunk):
            return stream_reproject_chunk(chunk, self.src_srs, dst_srs)

        return self.map(_repro_func)

    def crop(self, region: List[float]):
        """Injects a spatial crop step into the stream.
        region: [w, e, s, n]
        """

        from globato.processors.transforms.crop import stream_crop_chunk

        return self.map(stream_crop_chunk, region=region)

    def to_dataframe(self, limit: int = None) -> pd.DataFrame:
        """Consumes the stream and returns a Pandas DataFrame.
        Warning: This loads data into memory!
        """

        chunks = []
        count = 0

        for chunk in self._iterator:
            if chunk is None or len(chunk) == 0:
                continue

            chunks.append(pd.DataFrame(chunk))

            count += len(chunk)
            if limit and count >= limit:
                break

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks, ignore_index=True)
        if limit:
            df = df.head(limit)

        return df

    def to_numpy(self) -> np.recarray:
        """Consumes the stream and returns a single stacked numpy structured array."""

        chunks = list(self._iterator)
        if not chunks:
            return np.array([], dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f4')])
        return np.concatenate(chunks)


def read(source: Union[str, FetchModule], **kwargs) -> GlobatoStream:
    """The entry point for the Globato API.

    Args:
        source: A file path (str) OR a generic Fetchez Module instance.
        **kwargs: Arguments passed to the Reader (e.g. chunk_size, delimiter).

    Returns:
        GlobatoStream: An iterable stream object.
    """

    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source not found: {source}")

        reader = StreamFactory.get_reader(source, **kwargs)
        if not reader:
            raise ValueError(f"No valid reader found for {source}")

        raw_gen = reader.yield_chunks()

        src_srs = getattr(reader, "get_srs", lambda: "EPSG:4326")()

        w = kwargs.get("weight", 1.0)
        u = kwargs.get("uncertainty", 0.0)

        schema_gen = ensure_schema(raw_stream_wrapper(raw_gen), module_weight=w, module_unc=u)

        return GlobatoStream(schema_gen, src_srs=src_srs)

    elif isinstance(source, FetchModule):
        def _module_chain_gen():
            for entry in source.results:
                fn = entry.get('dst_fn')
                if fn and os.path.exists(fn):
                     try:
                         sub_stream = read(fn, **kwargs)
                         yield from sub_stream
                     except Exception as e:
                         logger.warning(f"Failed to stream {fn}: {e}")

        return GlobatoStream(_module_chain_gen(), src_srs="EPSG:4326") # Modules usually normalize to 4326?

    else:
        raise TypeError(f"Unknown source type: {type(source)}")

def raw_stream_wrapper(gen):
    """Helper to handle the difference between readers yielding plain tuples vs recarrays"""

    return gen
