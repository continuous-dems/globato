"""Filter a module's elevation sources with Stack QC before normal stacking.

A standard collection-stage hook for recipes that stack with ``side_stack``
and ``stack_combine``. Attach it to a module (it runs after the module's
sources have their side-stacks and before the global ``stack_combine``):

    - module: mbdb
      hooks:
        - name: stream_data
        - name: stream_reproject
          args: {dst_srs: "EPSG:4326+3855"}
        - name: stack_qc
          args: {res: 1s, qc_res: 1/9s, max_passes: 5}

or list it in ``global_hooks`` before ``stack_combine`` with ``modules: mbdb``.

The hook grids each source on the QC grid (the module's own side-stacks when
``qc_res`` is not finer than ``res``), runs the Stack QC removal and
restoration passes against working DEMs it builds itself (MultiStack plus
``ms_binary_cudem``, in this process), and writes each source's accepted
measurements as a side-stack on the DEM grid. The entry's ``side-stack``
artifact then points at the filtered side-stack, which ``stack_combine``
stacks; the original side-stack is kept and recorded as
``stack_qc_original_side_stack``.

Results are cached per module. The cache key holds only the module's sources,
weight and processing, the region, these settings and the QC code, so changes
to other datasets or to the final DEM's tiers reuse the filter results.
"""

import logging
import math

from fetchez.hooks import FetchHook
from fetchez.utils import str2bool

from globato.qc.stack.settings import Settings

logger = logging.getLogger(__name__)

# Result receipts written in this process, for stack_qc_review. focus_sink
# keeps only one entry's artifacts, so the review cannot rely on them alone.
RESULTS: list = []

# Working-DEM interpolation used when a recipe gives none: the CRM 1" ladder.
DEFAULT_INTERPOLATION = dict(
    steps=3,
    resolutions="1s/3s/9s/15s",
    weights=[0.6, 0.4, 0.2],
    algos="raster_fill/raster_fill/raster_fill/interp_gmt:tension=1",
    blend_dists="32/16/8/4",
    barrier="osm",
    previous_tier_resampling="cubic_spline",
    bathy_max_z="ocean:-0.01,river:0,lake:None,wetland:None,estuary:0",
)


def _cells(value, name):
    if isinstance(value, str):
        items = [v for v in value.replace(",", "/").split("/") if v.strip()]
    else:
        items = list(value)
    try:
        return tuple(int(v) for v in items)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stack_qc {name}: expected cell counts like 3/9/25") from exc


def _names(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.replace(",", "/").split("/") if v.strip()]
    return [str(v) for v in value]


class StackQC(FetchHook):
    """Stack QC filter for one module's sources (see the module docstring)."""

    name = "stack_qc"
    meta_stage = "collection"
    # Streams pass through unchanged; the filter acts on side-stack artifacts.
    affects_fusion_state = False
    meta_category = "filter"
    meta_desc = (
        "Remove elevation outliers and restore supported terrain before stacking."
    )
    _cli_help_text = (
        "Filter a module's sources against working DEMs built from the module's own "
        "data, restore supported ridges and valleys, then stack the accepted "
        "measurements on the DEM grid."
    )
    _cli_arg_help = {
        "res": "DEM grid the accepted measurements are stacked on (must match multi_stack res).",
        "crs": "Stack CRS (must match multi_stack crs).",
        "qc_res": "Optional finer QC grid; DEM res / qc_res must be a whole number (e.g. 1/9s for a 1s DEM).",
        "aggressiveness": "Filtering strength: 1 light, 2 standard, 3 aggressive.",
        "terrain_restoration": "0 off, 1 conservative, 2 standard, 3 stronger.",
        "max_passes": "Removal pass limit, 1-10.",
        "stop_when_stable": "Stop before max_passes once passes stop changing.",
        "consensus": "Also reject cells that disagree with other surveys.",
        "survey_exclusion": "off, near-complete or redundant.",
        "threshold_scale": "Multiply the detection thresholds (metres).",
        "median_windows": "Small/mid/large median windows in QC cells (default 3/9/25).",
        "flank_distances": "Flank distances in QC cells (default 4/6/9/12).",
        "plane_tolerance": "Metres a local plane may miss by and still protect a cell.",
        "restoration_length_scale": "Multiply the restoration distances.",
        "restoration_fit_scale": "Multiply the restoration fit tolerance.",
        "stack_strategy": "Working-DEM MultiStack strategy.",
        "stack_weight_threshold": "Working-DEM MultiStack weight tiers.",
        "interpolation": "ms_binary_cudem arguments for the working DEMs, written for the DEM grid (scaled to qc_res).",
        "modules": "As a global hook: the module names to filter.",
        "workers": "Processes for the geometry step (results do not depend on it).",
    }

    def __init__(
        self,
        res="1s",
        crs="EPSG:4326+3855",
        qc_res=None,
        aggressiveness=Settings.aggressiveness,
        terrain_restoration=Settings.terrain_restoration,
        max_passes=Settings.max_passes,
        stop_when_stable=Settings.stop_when_stable,
        consensus=Settings.consensus,
        survey_exclusion=Settings.survey_exclusion,
        stable_changed_fraction=Settings.stable_changed_fraction,
        stable_dem_change_m=Settings.stable_dem_change_m,
        stable_consecutive_passes=Settings.stable_consecutive_passes,
        threshold_scale=1.0,
        median_windows="3/9/25",
        flank_distances="4/6/9/12",
        plane_tolerance=1.75,
        restoration_length_scale=1.0,
        restoration_fit_scale=1.0,
        stack_strategy="mixed",
        stack_weight_threshold="3/2/1/.75/.5/.4/.3/.2/.1",
        interpolation=None,
        modules=None,
        cache_dir=None,
        side_stack_cache=None,
        threads=4,
        workers=4,
        **kwargs,
    ):
        super().__init__(stage="collection", **kwargs)
        self._settings = Settings(
            aggressiveness=int(aggressiveness),
            terrain_restoration=int(terrain_restoration),
            max_passes=int(max_passes),
            stop_when_stable=str2bool(stop_when_stable),
            consensus=str2bool(consensus),
            survey_exclusion=str(survey_exclusion),
            stable_changed_fraction=float(stable_changed_fraction),
            stable_dem_change_m=float(stable_dem_change_m),
            stable_consecutive_passes=int(stable_consecutive_passes),
            threshold_scale=float(threshold_scale),
            median_windows=_cells(median_windows, "median_windows"),
            flank_distances=_cells(flank_distances, "flank_distances"),
            plane_tolerance_m=float(plane_tolerance),
            restoration_length_scale=float(restoration_length_scale),
            restoration_fit_scale=float(restoration_fit_scale),
        )
        self.res = res
        self.crs = crs
        self.qc_res = qc_res
        self.stack_strategy = str(stack_strategy).lower()
        self.stack_weight_threshold = stack_weight_threshold
        self.interpolation = dict(
            DEFAULT_INTERPOLATION if interpolation is None else interpolation
        )
        self.modules = _names(modules)
        self.cache_dir = cache_dir
        self.side_stack_cache = side_stack_cache
        self.threads = int(threads)
        self._workers = max(1, int(workers))
        self._qc_increment, self._ladder = self._resolve_grids()

    # -- configuration ----------------------------------------------------

    @property
    def settings(self):
        return self._settings

    @property
    def workers(self):
        return self._workers

    @property
    def qc_increment(self):
        """QC grid increment in CRS units, or None to filter on the DEM grid."""
        return self._qc_increment

    def _resolve_grids(self):
        from globato.qc.stack.resampling import resolution

        final = resolution(self.res, crs=self.crs)
        ladder = self.interpolation.get("resolutions")
        if self.qc_res is None:
            return None, ladder
        fine = resolution(self.qc_res, crs=self.crs)
        ratio = final / fine
        if ratio < 1 - 1e-8 or not math.isclose(ratio, round(ratio), rel_tol=1e-8):
            raise ValueError(
                f"stack_qc: qc_res={self.qc_res!r} does not nest in res={self.res!r}; "
                "use res / N for a whole number N"
            )
        factor = round(ratio)
        if factor == 1:
            return None, ladder
        fine = final / factor
        # The working-DEM ladder is written for the DEM grid; scale it down so
        # its first tier is the QC grid (as the DEM's first tier is the DEM grid).
        from fetchez.utils import parse_arg_to_list
        from globato.utils import parse_increment

        if ladder:
            values = parse_arg_to_list(ladder, parse_increment)
            if not math.isclose(values[0], final, rel_tol=1e-8):
                raise ValueError(
                    "stack_qc: the first interpolation resolution must equal res"
                )
            values = [v / factor for v in values]
            values[0] = fine
            ladder = values
        return fine, ladder

    def stack_hook(self, res):
        from globato.hooks.sinks.multi_stack import MultiStackHook

        return MultiStackHook(
            res=res,
            crs=self.crs,
            strategy=self.stack_strategy,
            weight_threshold=self.stack_weight_threshold,
            output="stack_qc_working_stack.tif",
            resume=False,
        )

    def interpolation_args(self):
        args = dict(self.interpolation)
        if self._ladder is not None:
            args["resolutions"] = self._ladder
        args.pop("output", None)
        args["keep_steps"] = False
        return args

    def interpolator_hooks(self):
        """Fresh working-DEM interpolators for one build."""
        from globato.hooks.rasters.binary_cudem import BinaryCudemStepDown

        return [BinaryCudemStepDown(**self.interpolation_args())]

    def working_identity(self, stack):
        return dict(
            stack=dict(
                res=stack.res,
                crs=stack.crs,
                strategy=stack.strategy,
                weight_threshold=str(stack.weight_threshold),
            ),
            interpolation={k: str(v) for k, v in self.interpolation_args().items()},
        )

    # -- execution --------------------------------------------------------

    def run(self, entries):
        from globato.qc.stack.preparation import prepare

        groups = {}
        order = []
        for mod, entry in entries:
            if self.modules is not None and mod.name not in self.modules:
                continue
            if id(mod) not in groups:
                groups[id(mod)] = (mod, [])
                order.append(id(mod))
            groups[id(mod)][1].append((mod, entry))
        for key in order:
            mod, selected = groups[key]
            prepare(self, mod, selected, threads=self.threads)
            for _, entry in selected:
                receipt = (entry.get("artifacts") or {}).get("stack_qc")
                if receipt and receipt not in RESULTS:
                    RESULTS.append(receipt)
        return entries
