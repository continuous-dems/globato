"""Filter a module's elevation contributions before normal Globato stacking."""

from fetchez.hooks import FetchHook
from fetchez.utils import str2bool
from globato.qc.stack.settings import Settings


class StackQC(FetchHook):
    """Attach after source transforms; run with the ordinary Globato DEM recipe."""

    name = "stack_qc"
    meta_stage = "collection"
    meta_category = "filter"
    meta_desc = (
        "Remove elevation outliers and restore supported terrain before stacking."
    )
    affects_fusion_state = False
    _cli_help_text = (
        "Filter the attached module's elevation sources, then restore supported "
        "ridges and valleys. Uses native side stacks and the normal Globato DEM "
        "interpolator. Rasters use the existing stream_data reader."
    )
    _cli_arg_help = {
        "aggressiveness": "Filtering strength: 1 Light, 2 Standard, 3 Aggressive.",
        "terrain_restoration": "Terrain restoration: 0 Off, 1 Conservative, 2 Standard, 3 Stronger.",
        "max_passes": f"Pass limit, from 1 to 10. Default: {Settings.max_passes}.",
        "stop_when_stable": "true stops after small changes; false forces max_passes.",
        "qc_res": "Optional QC grid: geographic requires 1/9s or 1s; projected uses numeric CRS units. Must equal DEM increment / positive integer. Omit to follow DEM grid.",
        "review": "Optional review.html and comparison PNGs after a successful build (default false).",
    }

    def __init__(
        self,
        aggressiveness=Settings.aggressiveness,
        terrain_restoration=Settings.terrain_restoration,
        max_passes=Settings.max_passes,
        stop_when_stable=Settings.stop_when_stable,
        review=False,
        qc_res=None,
        **kwargs,
    ):
        if "level" in kwargs:
            raise TypeError(
                "Stack QC renamed level to aggressiveness; use aggressiveness: 1, 2, or 3"
            )
        if "restoration_level" in kwargs:
            raise TypeError(
                "Stack QC renamed restoration_level to terrain_restoration; use terrain_restoration: 0, 1, 2, or 3"
            )
        if kwargs:
            raise TypeError(
                "Stack QC accepts only aggressiveness, terrain_restoration, "
                "max_passes, stop_when_stable, review, and qc_res; unsupported: "
                + ", ".join(sorted(kwargs))
            )
        super().__init__(stage="collection")
        self.settings = Settings(
            aggressiveness=aggressiveness,
            terrain_restoration=terrain_restoration,
            max_passes=max_passes,
            stop_when_stable=str2bool(stop_when_stable),
        )
        # QC working files follow the module's output folder. Shared source
        # downloads remain managed by the normal Fetchez/Globato cache.
        if qc_res is not None:
            from globato.qc.stack.resampling import resolution

            # Validate syntax now; validate CRS units and nesting during preparation.
            resolution(qc_res)
        self.qc_res = qc_res
        self.cache_dir = None
        self.review = str2bool(review)
        if self.review:
            try:
                import matplotlib  # noqa: F401  -- only needed for review=true
            except ImportError as exc:
                raise RuntimeError("Stack QC review=true requires matplotlib") from exc

    def run(self, entries):
        raise RuntimeError(
            "Stack QC needs the Globato recipe preparation stage; use globato run rather than fetchez run"
        )

    def prepare_module(self, module, global_hooks, *, threads):
        from globato.qc.stack.preparation import prepare

        return prepare(self, module, global_hooks, threads=threads)
