#!/usr/bin/env python3
"""Globato user hook: fill enclosed ocean-mask holes to retain offshore rocks.

Install this file in PROJECT/.fetchez/hooks/ (or ~/.fetchez/hooks/).
Use after class selection, stream_reproject, and spatial_crop. To retain ALL
selected topo points inside the filled ocean side, omit water_surface_filter.

OSM convention: 1 = land, 0 = water. Find boundary-connected water and fill its
enclosed holes. Boundary-connected mainland stays intact, regardless of coast
orientation. No mainland seed, directional edge rule, or elevation fill is used.
The mask is an OSM-derived routing boundary, not a surveyed shoreline.
"""

import hashlib
import copy
import json
import logging
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from pyproj import Transformer
from scipy import ndimage

from fetchez.utils import str2bool
from globato.hooks.filters.base import GlobatoFilter

logger = logging.getLogger(__name__)
_CROSS = ndimage.generate_binary_structure(2, 1)
HOOK_VERSION = '2.3'
_OSM_KEYWORDS = frozenset(('osm', 'landmask', 'coastline'))


def _numbers(value, count, label):
    parts = value.replace(',', '/').split('/') if isinstance(value, str) else value
    try:
        result = tuple(float(x) for x in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must contain {count} numbers.') from exc
    if len(result) != count or not all(math.isfinite(x) for x in result):
        raise ValueError(f'{label} must contain {count} finite numbers.')
    return result


def _boundary(shape):
    result = np.zeros(shape, dtype=bool)
    result[:, 0] = result[:, -1] = True
    result[0, :] = result[-1, :] = True
    return result


def build_sides(image):
    """Fill holes enclosed by boundary-connected water, not mainland lakes.

    Four-neighbor water connectivity finds the initial ocean. Four-neighbor
    background connectivity defines holes in that ocean. Unknown cells are
    never selected and act as non-ocean when determining an enclosure.
    Returns remaining mainland, filled ocean side, filled land pixels, known.
    """
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0:
        raise ValueError('The source must be a nonempty 2-D binary land mask.')
    known = (image == 0) | (image == 1)
    land = image == 1
    water = image == 0
    water_seeds = _boundary(image.shape) & water
    if not np.any(water_seeds):
        raise ValueError('No boundary-connected water. Use a context mask that includes open ocean.')
    initial_ocean = ndimage.binary_propagation(water_seeds, structure=_CROSS, mask=water)
    ocean = ndimage.binary_fill_holes(initial_ocean, structure=_CROSS) & known
    removed_islands = land & ocean
    mainland = land & ~ocean
    return mainland, ocean, removed_islands, known


def _write_mask(path, values, known, profile):
    profile = profile.copy()
    profile.update(driver='GTiff', count=1, dtype='uint8', nodata=255, compress='deflate')
    # Never publish a partially written raster when several worker processes run.
    fd, temporary = tempfile.mkstemp(suffix='.tif', dir=path.parent)
    os.close(fd)
    try:
        with rasterio.open(temporary, 'w', **profile) as dst:
            dst.write(np.where(known, values, 255).astype('uint8'), 1)
            dst.update_tags(mask_meaning='1=selected, 0=not selected, 255=unknown')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class MainlandOceanMask(GlobatoFilter):
    """Keep ocean-side points including mapped rocks, or keep mainland points."""

    name = 'mainland_ocean_mask'
    meta_desc = 'Fill enclosed ocean-mask holes and select ocean or remaining mainland points.'
    meta_aliases = []

    def __init__(self, barrier='osm', side='ocean', context_region=None,
                 context_buffer=0.02, res='.333333333333s', cache_dir=None,
                 soft=False, max_pixels=50000000, log_points=True, **kwargs):
        if 'invert' in kwargs:
            raise ValueError('Use side: ocean or side: mainland instead of invert.')
        if any(key in kwargs for key in ('mainland_edges', 'mainland_seed', 'ocean_seed')):
            raise ValueError('The hole-filling version uses no edge rules or seeds. Remove those old arguments.')
        super().__init__(invert=False, **kwargs)
        if side not in ('ocean', 'mainland'):
            raise ValueError('side must be ocean or mainland.')
        self.barrier, self.side = barrier, side
        self.context_region = context_region
        self.context_buffer = float(context_buffer)
        if not math.isfinite(self.context_buffer) or self.context_buffer < 0:
            raise ValueError('context_buffer must be finite and nonnegative.')
        self.res, self.cache_dir = res, cache_dir
        self.soft = str2bool(soft)
        self.log_points = str2bool(log_points)
        self.max_pixels = int(max_pixels)
        if self.max_pixels < 1:
            raise ValueError('max_pixels must be positive.')
        self._setup_key = None
        self.mask_array = None
        self._transformer = None
        self.artifacts = {}
        self._point_stats = None
        self._stream_worker = False
        self._stream_started = False
        self._summary_logged = False

    def _context(self, mod):
        from fetchez.spatial import Region
        if self.context_region is not None:
            bounds = _numbers(self.context_region, 4, 'context_region')
        else:
            region = getattr(mod, 'wgs_region', None)
            if region is None:
                raise ValueError('OSM generation requires context_region or mod.wgs_region.')
            w, e, s, n = tuple(region)
            b = self.context_buffer
            bounds = (w - b, e + b, s - b, n + b)
        w, e, s, n = bounds
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError('context_region must be west/east/south/north in geographic degrees.')
        requested = getattr(mod, 'wgs_region', None)
        if requested is not None:
            rw, re, rs, rn = tuple(requested)
            if not (w <= rw <= re <= e and s <= rs <= rn <= n):
                raise ValueError('context_region must cover the requested DEM region.')
        return Region(w, e, s, n, srs='EPSG:4326')

    def setup(self, mod, entry):
        from globato.utils import resolve_barrier

        cache = Path(self.cache_dir or getattr(mod, '_outdir', None) or getattr(mod, 'outdir', None) or os.getcwd()) / 'mainland_ocean_masks'
        cache.mkdir(parents=True, exist_ok=True)
        # Recipe path expansion may rewrite barrier: osm to /recipe/folder/osm.
        # Recover a missing keyword path before stat(), and pass the canonical
        # keyword to resolve_barrier. Existing rasters named 'osm' stay files.
        if self.barrier is None or not str(self.barrier).strip():
            raise ValueError('barrier must be osm, landmask, coastline, or a binary raster path.')
        barrier = os.fspath(self.barrier).strip()
        candidate = Path(barrier)
        keyword = candidate.name.lower()
        magic = keyword in _OSM_KEYWORDS and not candidate.exists()
        if magic:
            barrier = keyword
        context = self._context(mod) if magic else None
        # In-memory reuse for every lidar entry of this module/region.
        region_key = tuple(context) if context is not None else None
        source_stat = None if magic else (Path(barrier).stat().st_mtime_ns, Path(barrier).stat().st_size)
        key = (barrier, region_key, source_stat, str(cache))
        if key != self._setup_key:
            if context is not None:
                from fetchez.utils import str2inc
                step = str2inc(self.res)
                if step is None or not math.isfinite(step) or step <= 0:
                    raise ValueError('res must be a positive geographic increment.')
                cells = math.ceil(context.width / step) * math.ceil(context.height / step)
                if cells > self.max_pixels:
                    raise ValueError('Context exceeds max_pixels before OSM generation; use a coarser res or smaller context_region.')
            # Match the original point_raster_mask/water_surface_filter OSM
            # overlay options. Existing raster inputs are read without resampling.
            path = (resolve_barrier(barrier, region=context, outdir=str(cache / 'source'),
                                    res=self.res, include_rivers=True, include_lakes=True,
                                    include_breakwaters=True, include_wetlands=True,
                                    include_reefs=False, include_estuaries=True,
                                    output_type='raster', target_crs='EPSG:4326')
                    if magic else barrier)
            if not path:
                raise RuntimeError('Could not build the mainland/ocean source mask.')
            with rasterio.open(path) as src:
                if src.crs is None:
                    raise ValueError('The source mask must have a CRS.')
                if src.width * src.height > self.max_pixels:
                    raise ValueError('Mask exceeds max_pixels; use a coarser res or smaller context_region.')
                image = src.read(1)
                mainland, ocean, islands, known = build_sides(image)
                self.mask_transform, self.mask_crs = src.transform, src.crs
                self.mask_array = ocean if self.side == 'ocean' else mainland
                signature = hashlib.sha256(image.tobytes() + json.dumps({
                    'version': 2, 'method': 'fill_ocean_holes', 'transform': tuple(src.transform),
                    'crs': src.crs.to_wkt(), 'shape': image.shape,
                }, sort_keys=True).encode()).hexdigest()[:16]
                for kind, values in [('mainland', mainland), ('ocean', ocean), ('offshore_islands', islands)]:
                    output = cache / f'{signature}_{kind}.tif'
                    if not output.exists():
                        _write_mask(output, values, known, src.profile)
                    self.artifacts[f'{kind}_mask'] = str(output.resolve())
                logger.info('[%s] mainland=%d ocean=%d rock/island hole pixels filled=%d; masks: %s',
                            self.name, np.count_nonzero(mainland), np.count_nonzero(ocean),
                            np.count_nonzero(islands), cache)
            self._setup_key = key
        point_crs = entry.get('src_srs')
        if not point_crs:
            raise ValueError('Point src_srs is required. Put this hook after stream_reproject.')
        self._transformer = Transformer.from_crs(point_crs, self.mask_crs, always_xy=True)
        self._point_crs = str(point_crs)
        self._entry_name = str(entry.get('dst_fn') or entry.get('url') or 'unnamed entry')
        self._point_stats = dict(input=0, finite_z=0, in_grid=0, selected=0,
                                 selected_finite_z=0, selected_above_zero=0)
        self._input_bounds = None
        self._mask_bounds = None
        self._classes = {}
        entry.setdefault('artifacts', {}).update(self.artifacts)
        return True

    def run(self, entries):
        # Streams are lazy. Freeze each entry's mask/CRS and counters rather than
        # letting a later setup() overwrite state used by an earlier generator.
        for mod, entry in entries:
            if not self.is_point_stream(entry):
                continue
            stream = entry.get('stream')
            if self.setup(mod, entry) is False:
                continue
            worker = copy.copy(self)
            worker._point_stats = self._point_stats.copy()
            worker._classes = {}
            worker._stream_worker = True
            worker._stream_started = False
            worker._summary_logged = False
            entry['stream'] = worker._process_stream(stream)
        return entries

    def _process_stream(self, stream):
        self._stream_started = True
        yield from super()._process_stream(stream)

    @staticmethod
    def _extend_bounds(previous, x, y):
        x, y = np.asarray(x), np.asarray(y)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            return previous
        bounds = (float(np.min(x[valid])), float(np.max(x[valid])),
                  float(np.min(y[valid])), float(np.max(y[valid])))
        if previous is None:
            return bounds
        return (min(previous[0], bounds[0]), max(previous[1], bounds[1]),
                min(previous[2], bounds[2]), max(previous[3], bounds[3]))

    def teardown(self):
        # Core also tears down template hooks, including ones with no setup().
        # Only a started entry worker owns meaningful point counts; log once.
        if (not self.log_points or not self._stream_worker or not self._stream_started
                or self._summary_logged or self._point_stats is None):
            return
        self._summary_logged = True
        stats = self._point_stats
        logger.info('[%s] POINTS %s: input=%d finite_z=%d in_grid=%d selected=%d '
                    'selected_finite_z=%d selected_above_zero=%d classes=%s '
                    'input_xy=%s mask_xy=%s point_crs=%s mask_crs=%s',
                    self.name, self._entry_name, stats['input'], stats['finite_z'],
                    stats['in_grid'], stats['selected'], stats['selected_finite_z'],
                    stats['selected_above_zero'], self._classes, self._input_bounds,
                    self._mask_bounds, self._point_crs, self.mask_crs)

    def filter_chunk(self, chunk):
        if self.mask_array is None or self._transformer is None:
            raise RuntimeError('mainland_ocean_mask setup has not completed.')
        stats = self._point_stats
        stats['input'] += len(chunk)
        finite_z = np.isfinite(chunk['z'])
        stats['finite_z'] += int(np.count_nonzero(finite_z))
        self._input_bounds = self._extend_bounds(self._input_bounds, chunk['x'], chunk['y'])
        if 'classification' in chunk.dtype.names:
            classes, counts = np.unique(chunk['classification'], return_counts=True)
            for cls, count in zip(classes, counts):
                key = int(cls)
                self._classes[key] = self._classes.get(key, 0) + int(count)
        keep = np.zeros(len(chunk), dtype=bool)
        finite = np.isfinite(chunk['x']) & np.isfinite(chunk['y'])
        point_ids = np.flatnonzero(finite)
        if point_ids.size:
            x, y = self._transformer.transform(chunk['x'][point_ids], chunk['y'][point_ids])
            self._mask_bounds = self._extend_bounds(self._mask_bounds, x, y)
            cols, rows = (~self.mask_transform) * (np.asarray(x), np.asarray(y))
            valid = (np.isfinite(cols) & np.isfinite(rows) & (cols >= 0) & (rows >= 0)
                     & (cols < self.mask_array.shape[1]) & (rows < self.mask_array.shape[0]))
            stats['in_grid'] += int(np.count_nonzero(valid))
            keep[point_ids[valid]] = self.mask_array[np.floor(rows[valid]).astype(int), np.floor(cols[valid]).astype(int)]
        stats['selected'] += int(np.count_nonzero(keep))
        stats['selected_finite_z'] += int(np.count_nonzero(keep & finite_z))
        stats['selected_above_zero'] += int(np.count_nonzero(keep & finite_z & (chunk['z'] > 0)))
        # GlobatoFilter expects True for rejected/classified points in soft mode;
        # invert is deliberately disabled above to avoid double inversion.
        return ~keep if self.soft else chunk[keep]
