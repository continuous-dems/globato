"""Filter source contributions at Fetchez's collection barrier, before assembly.

Use ``stack_qc_grid`` at the stream stage to materialize each selected source's
native FusionState, then one global ``stack_qc`` after all module streams finish.
``modules`` selects targets; ``add_ref_modules`` adds fixed, unfiltered evidence.
The target modules automatically contribute to the shared weighted working DEM
and survey context. Each pass evaluates all targets against the same parent
state; references never receive rejection masks. Original entry owners and
source identities remain attached throughout publication and ``stack_combine``.

Both selectors accept names or normal recipe-style ``{module, args}`` mappings.
Use matching selectors, res, qc_res and crs on the two hooks. Joint contexts
require ``stack_qc_grid`` even when QC and DEM resolutions are equal. Sources
outside the context retain their normal side-stacks for final assembly.

Legacy explicit single-module recipes retain their v114 input selection and
reduction order. New joint grids materialize current streams in private paths;
QC decisions can be reused by exact grid content, source roles, weights,
settings, region, working interpolation and QC code. This does not assert a
universal upstream transform/datum fingerprint or shared immutable grid cache.

Raster review and regional recipes remain optional, separate consumers.
"""

import logging
import math
import os
from pathlib import Path
from uuid import uuid4

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
    if isinstance(value, dict):
        value = [value]
    result = []
    for selector in value:
        if isinstance(selector, dict):
            if set(selector) - {"id", "module", "args"} or not (
                selector.get("module") or selector.get("id")
            ):
                raise ValueError(
                    "QC selectors use a recipe id or module and optional args"
                )
            if not isinstance(selector.get("args", {}), dict):
                raise ValueError("QC selector args must be a mapping")
            result.append(
                dict(
                    **(
                        {"module": str(selector["module"])}
                        if selector.get("module")
                        else {}
                    ),
                    **({"id": str(selector["id"])} if selector.get("id") else {}),
                    args=dict(selector.get("args", {})),
                )
            )
        else:
            result.append(str(selector))
    return result


def _selected(module, selectors):
    """Match existing module invocations without creating aliases or new identities."""
    if selectors is None:
        return True
    for selector in selectors:
        if isinstance(selector, str):
            # A recipe ID wins over a provider-name shorthand, even if that
            # recipe entry has no files in this region. Explicit {module: ...}
            # selectors still select every matching provider invocation.
            known_ids = getattr(module, "_instance_ids", ())
            if selector in known_ids:
                if selector == getattr(module, "instance_id", None):
                    return True
            elif selector in {module.name, getattr(module, "instance_id", None)}:
                return True
        elif ("module" not in selector or selector["module"] == module.name) and (
            "id" not in selector
            or selector["id"] == getattr(module, "instance_id", None)
        ):
            args = getattr(module, "_init_kwargs", {})
            if all(
                getattr(module, key, args.get(key)) == value
                or str(getattr(module, key, args.get(key))) == str(value)
                for key, value in selector.get("args", {}).items()
            ):
                return True
    return False


def _validate_recipe_selectors(entries, selectors):
    """Catch misspelled recipe labels before streams or filtering run."""
    if not entries:
        return
    owner = entries[0][0]
    labels = getattr(owner, "_instance_ids", None)
    if labels is None:
        return
    providers = getattr(owner, "_recipe_providers", {owner.name})
    for selector in selectors or []:
        if (
            isinstance(selector, str)
            and selector not in labels
            and selector not in providers
        ):
            raise ValueError(f"Unknown QC recipe id/provider: {selector}")
        if (
            isinstance(selector, dict)
            and "id" in selector
            and selector["id"] not in labels
        ):
            raise ValueError(f"Unknown QC recipe id: {selector['id']}")


class StackQC(FetchHook):
    """Stack QC filter for collected target sources and optional fixed references."""

    name = "stack_qc"
    meta_stage = "collection"
    meta_category = "filter"
    meta_desc = (
        "Remove elevation outliers and restore supported terrain before stacking."
    )
    _cli_help_text = (
        "Filter selected sources against a shared weighted working DEM, "
        "restore supported ridges and valleys, then stack the accepted "
        "measurements on the DEM grid."
    )
    _cli_arg_help = {
        "res": "DEM grid the accepted measurements are stacked on (must match stack_combine res).",
        "crs": "Stack CRS (must match stack_combine crs).",
        "qc_res": "Optional finer QC grid; DEM res / qc_res must be a whole number (e.g. 1/9s for a 1s DEM). The filtered module's own side_stack must use it.",
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
        "modules": "As a global hook: recipe entry IDs, provider names or module/args selectors; targets also supply QC evidence.",
        "add_ref_modules": "Additional unfiltered modules in the shared QC context; selectors may use recipe-style module/args mappings.",
        "source_artifact": "Artifact holding each entry's DEM-grid side-stack (default side-stack); filtered when there is no qc_res.",
        "grid_artifact": "With qc_res: artifact holding each entry's QC-grid FusionState from stack_qc_grid (default stack-qc-grid).",
        "output_artifact": "Artifact to record the accepted side-stacks in (default qc-side-stack); other modules' entries get their source_artifact here.",
        "workers": "Processes for the geometry step (results do not depend on it).",
        "method": "geometry (upstream) or connectivity (experimental two-pass port).",
        "viz": "Automatically publish review products and a Raster Viz catalog after the final DEM.",
    }

    def __init__(
        self,
        res="1s",
        crs="EPSG:4326+3855",
        qc_res=None,
        aggressiveness=Settings.aggressiveness,
        terrain_restoration=Settings.terrain_restoration,
        max_passes=Settings.max_passes,
        method="geometry",
        viz=False,
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
        add_ref_modules=None,
        source_artifact="side-stack",
        grid_artifact="stack-qc-grid",
        output_artifact="qc-side-stack",
        cache_dir=None,
        workers=4,
        **kwargs,
    ):
        stage = str(kwargs.pop("stage", "collection")).lower()
        if stage not in {"collection", "post"}:
            raise ValueError(
                "stack_qc runs at the collection stage, on side-stacks; "
                "use side_stack + stack_combine"
            )
        super().__init__(stage="collection", **kwargs)
        self._settings = Settings(
            aggressiveness=int(aggressiveness),
            terrain_restoration=int(terrain_restoration),
            max_passes=int(max_passes),
            method=str(method).lower(),
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
        self.viz = str2bool(viz)
        self._review_receipts = []
        self.res = res
        self.crs = crs
        self.qc_res = qc_res
        self.stack_strategy = str(stack_strategy).lower()
        self.stack_weight_threshold = stack_weight_threshold
        self.interpolation = dict(
            DEFAULT_INTERPOLATION if interpolation is None else interpolation
        )
        self.modules = _names(modules)
        self.add_ref_modules = _names(add_ref_modules) or []
        self.source_artifact = str(source_artifact)
        self.grid_artifact = str(grid_artifact)
        self.output_artifact = str(output_artifact)
        self.cache_dir = cache_dir
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
        from globato.qc.stack.resampling import nested_increment, resolution

        final = resolution(self.res, crs=self.crs)
        ladder = self.interpolation.get("resolutions")
        fine = nested_increment(self.res, self.qc_res, crs=self.crs)
        if fine is None:
            return None, ladder
        factor = round(final / fine)
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

    def final_hooks(self):
        """Use the normal pipeline's final collection hooks for optional review."""
        if not self.viz:
            return ()
        from .stack_qc_review import StackQCReview

        review = StackQCReview()
        review._qc_hooks = (self,)
        return (review,)

    def run(self, entries):
        from globato.qc.stack.preparation import prepare

        self._review_receipts.clear()
        _validate_recipe_selectors(entries, self.modules)
        _validate_recipe_selectors(entries, self.add_ref_modules)
        selected, references = [], []
        for mod, entry in entries:
            target = _selected(mod, self.modules)
            reference = _selected(mod, self.add_ref_modules)
            if target and reference:
                raise ValueError(
                    f"stack_qc: {mod.name} selected for both filtering and reference"
                )
            if reference:
                references.append((mod, entry))
                continue
            if not target:
                # Not filtered: stack_combine reads its side-stack unchanged.
                artifacts = entry.get("artifacts") or {}
                if self.source_artifact in artifacts:
                    artifacts.setdefault(
                        self.output_artifact, artifacts[self.source_artifact]
                    )
                continue
            selected.append((mod, entry))
        if not selected and not references:
            return entries
        if self.add_ref_modules and any(
            self is hook for mod, _ in selected for hook in getattr(mod, "hooks", ())
        ):
            raise ValueError(
                "stack_qc with additional references requires global_hooks after all source streams"
            )
        missing_references = [
            selector
            for selector in self.add_ref_modules
            if not any(_selected(mod, [selector]) for mod, _ in references)
        ]
        if missing_references:
            logger.warning(
                "stack_qc: no available source entries for reference selectors %s",
                missing_references,
            )
        participants = [*selected, *references]
        first = (selected or references)[0][0]
        for mod, entry in participants:
            if entry.get("_stack_qc_signature"):
                raise ValueError(
                    "stack_qc sources already filtered; use one global collection hook"
                )
            if entry.get("status") in {"failed", "error"}:
                raise ValueError("stack_qc requires successful source streams")
            if (
                mod.region is None
                or mod.region.to_list() != first.region.to_list()
                or mod.region.srs != first.region.srs
            ):
                raise ValueError(
                    "stack_qc participants require the same region and CRS"
                )
        joint = (
            bool(self.add_ref_modules)
            or len({id(mod) for mod, _ in selected}) > 1
            or (self.modules is not None and len(self.modules) > 1)
            or any(isinstance(s, dict) for s in self.modules or [])
            or any(
                s == getattr(mod, "instance_id", None)
                for s in self.modules or []
                if isinstance(s, str)
                for mod, _ in selected
            )
        )
        prepare(self, first, selected, references=references, joint=joint)
        for _, entry in selected:
            receipt = (entry.get("artifacts") or {}).get("stack_qc")
            if receipt and receipt not in self._review_receipts:
                self._review_receipts.append(receipt)
            if receipt and receipt not in RESULTS:
                RESULTS.append(receipt)
        return entries


class StackQCGrid(FetchHook):
    """Grid a module's points on the Stack QC grid, for ``stack_qc``.

    A stream-stage hook, listed before ``side_stack``. For each entry of the
    named modules it bins the points on the grid ``stack_qc`` filters on (the
    ``qc_res`` grid nested in the ``res`` DEM grid), caches that FusionState like
    a side-stack, and records it as ``artifact``. The entry then carries the
    QC-grid FusionState instead of points, so ``side_stack`` (which only
    side-stacks point streams) leaves it alone, and ``stack_qc`` writes the
    accepted measurements on the DEM grid. Use the same ``modules``, ``res``,
    ``qc_res`` and ``crs`` as ``stack_qc``.
    """

    name = "stack_qc_grid"
    meta_stage = "stream"
    meta_category = "filter"
    meta_desc = "Grid a module's points on the Stack QC grid for stack_qc."
    _cli_arg_help = {
        "res": "DEM grid (as stack_qc res).",
        "qc_res": "QC grid (as stack_qc qc_res); without it, the DEM grid.",
        "crs": "Stack CRS (as stack_qc crs).",
        "modules": "Module names to grid (as stack_qc modules).",
        "add_ref_modules": "Additional QC reference modules, matching stack_qc.",
        "artifact": "Artifact recording each entry's QC-grid FusionState.",
        "cache_dir": "Cache directory (default: the module's cache directory).",
    }

    def __init__(
        self,
        res="1s",
        crs="EPSG:4326+3855",
        qc_res=None,
        modules=None,
        artifact="stack-qc-grid",
        add_ref_modules=None,
        cache_dir=None,
        compress=True,
        **kwargs,
    ):
        stage = str(kwargs.pop("stage", "stream")).lower()
        if stage != "stream":
            raise ValueError("stack_qc_grid runs at the stream stage")
        super().__init__(stage="stream", **kwargs)
        from globato.qc.stack.resampling import nested_increment

        nested_increment(res, qc_res, crs=crs)  # fail early if it cannot nest
        self.res = res
        self.crs = crs
        self.qc_res = qc_res
        self.modules = _names(modules)
        self.artifact = str(artifact)
        self.add_ref_modules = _names(add_ref_modules) or []
        self._grid_generation = uuid4().hex
        self.cache_dir = cache_dir
        self.compress = str2bool(compress)

    def run(self, entries):
        from globato.hooks.sinks.side_stack import (
            SideStackGenerate,
            read_fusion_state_stream,
            sidestack_cache_key,
            sidestack_path,
            sidestack_provenance,
            validate_sidestack,
        )
        from globato.hooks.transforms.point_pixels import Point2PixelStream
        from globato.qc.stack.resampling import qc_grid

        _validate_recipe_selectors(entries, self.modules)
        _validate_recipe_selectors(entries, self.add_ref_modules)
        grids = {}
        for mod, entry in entries:
            if not (
                _selected(mod, self.modules) or _selected(mod, self.add_ref_modules)
            ):
                continue
            region = getattr(mod, "region", None)
            if region is None:
                continue
            key = id(mod)
            if key not in grids:
                grids[key] = qc_grid(self.res, self.qc_res, self.crs, region)
            increment, qc_region, grid = grids[key]
            cache_dir = self.cache_dir or getattr(
                mod, "_outdir", getattr(mod, "outdir", None)
            )
            digest = sidestack_cache_key(entry, grid)
            joint_grid = (
                bool(self.add_ref_modules)
                or self.modules is None
                or len(self.modules) > 1
                or any(isinstance(s, dict) for s in self.modules)
                or getattr(mod, "instance_id", None) in (self.modules or [])
            )
            if joint_grid:
                # Use Fetchez contribution identity, keeping weights/invocations
                # distinct without changing the original entry's ownership.
                from fetchez.entry import set_contribution_id

                identity_entry = dict(entry)
                identity_entry["contribution_id"] = dict(
                    entry.get("contribution_id") or {}
                )
                set_contribution_id(
                    identity_entry,
                    "stack_qc_grid",
                    dict(
                        version=1,
                        weight=mod.weight,
                        module_id=mod.module_id()
                        if callable(getattr(mod, "module_id", None))
                        else mod.name,
                    ),
                )
                digest = sidestack_cache_key(identity_entry, grid)
                cache_dir = (
                    Path(cache_dir or ".")
                    / "stack_qc_grid"
                    / self._grid_generation
                    / uuid4().hex
                )
            path = sidestack_path(entry, cache_dir, digest)
            entry.setdefault("artifacts", {})[self.artifact] = path

            valid, _ = validate_sidestack(path, grid, digest)
            # The current Fetchez stream boundary has no universal fingerprint
            # of every upstream transform/remote datum dependency. New joint
            # contexts therefore materialize current streams, then cache QC by
            # their exact grid contents. Do not promote schema-only hits into a
            # new shared immutable source cache.
            if joint_grid:
                valid = False
            if valid:
                stream = read_fusion_state_stream(path)
            elif self.is_point_stream(entry):
                points = Point2PixelStream(x_inc=increment, y_inc=increment)
                writer = SideStackGenerate(
                    res=increment,
                    crs=self.crs,
                    cache_dir=cache_dir,
                    compress=self.compress,
                )
                stream = writer._cache_stream(
                    points._stream_wrapper(
                        entry["stream"], entry=entry, region=qc_region
                    ),
                    path,
                    digest=digest,
                    grid=grid,
                    entry=entry,
                    provenance=sidestack_provenance(entry),
                    replace_existing=joint_grid,
                )
            else:
                raise ValueError(
                    f"stack_qc_grid: {os.path.basename(str(entry.get('dst_fn')))} "
                    "has no cached QC grid and its points are gone (it arrived as "
                    f"{entry.get('stream_type')!r}, e.g. a cached side-stack from "
                    "side_stack_check). List stack_qc_grid before side_stack, or "
                    "remove that module's cached side-stacks and rebuild."
                )
            from globato.qc.stack.profile import timed_stream

            grid_profile = dict(
                cache_hit=valid,
                identity=digest,
                path=path,
                phase="source_read_reproject_and_qc_grid"
                if not valid
                else "qc_grid_read",
            )
            entry["_stack_qc_grid_profile"] = grid_profile
            entry["stream"] = timed_stream(stream, grid_profile)
            entry["stream_type"] = "raster-stream"
            entry["data_type"] = "fusion-state"
        return entries
