"""Public filter controls; evidence sufficiency does not change with aggressiveness.

The tuning fields after ``stable_consecutive_passes`` default to the original
fixed constants, so default settings reproduce the original decisions exactly.
Window sizes are grid cells, so they matter when the QC grid changes.
"""

import math
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Settings:
    aggressiveness: int = 2
    terrain_restoration: int = 2
    max_passes: int = 4
    stop_when_stable: bool = True
    consensus: bool = False
    survey_exclusion: str = "off"
    stable_changed_fraction: float = 0.00001
    stable_dem_change_m: float = 0.05
    stable_consecutive_passes: int = 2
    # Detection thresholds (metres) are multiplied by threshold_scale.
    threshold_scale: float = 1.0
    # Median windows (small/mid/large) and flank distances, in QC grid cells.
    median_windows: tuple = (3, 9, 25)
    flank_distances: tuple = (4, 6, 9, 12)
    # A local plane within this many metres protects a flagged cell.
    plane_tolerance_m: float = 1.75
    # Restoration distances (metres) and fit tolerances are multiplied by these.
    restoration_length_scale: float = 1.0
    restoration_fit_scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "median_windows", tuple(self.median_windows))
        object.__setattr__(self, "flank_distances", tuple(self.flank_distances))
        if len(self.median_windows) != 3 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 or v % 2 == 0
            for v in self.median_windows
        ):
            raise ValueError("median_windows must be three odd positive cell counts")
        if not self.flank_distances or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in self.flank_distances
        ):
            raise ValueError("flank_distances must be positive cell counts")
        for key in (
            "threshold_scale",
            "plane_tolerance_m",
            "restoration_length_scale",
            "restoration_fit_scale",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a positive number")
        for key, allowed in (
            ("aggressiveness", range(1, 4)),
            ("terrain_restoration", range(4)),
            ("max_passes", range(1, 11)),
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in allowed
            ):
                raise ValueError(f"Invalid {key}: {value!r}")
        if self.survey_exclusion not in ("off", "near-complete", "redundant"):
            raise ValueError(
                "survey_exclusion must be off, near-complete, or redundant"
            )
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.stable_changed_fraction, self.stable_dem_change_m)
        ):
            raise ValueError("Convergence thresholds must be finite and nonnegative")
        if (
            not 0 <= self.stable_changed_fraction <= 1
            or isinstance(self.stable_consecutive_passes, bool)
            or not isinstance(self.stable_consecutive_passes, int)
            or self.stable_consecutive_passes < 1
        ):
            raise ValueError("Invalid convergence fraction or consecutive-pass count")

    def to_dict(self):
        return asdict(self)

    def detection_levels(self):
        from .geometry import scaled_levels

        return scaled_levels(self.threshold_scale)

    def science(self):
        """Fields that change pass decisions; part of the QC cache key."""
        return dict(
            threshold_scale=self.threshold_scale,
            median_windows=list(self.median_windows),
            flank_distances=list(self.flank_distances),
            plane_tolerance_m=self.plane_tolerance_m,
            restoration_length_scale=self.restoration_length_scale,
            restoration_fit_scale=self.restoration_fit_scale,
        )


# Shape eligibility and fit tolerances; all distances/elevation errors are metres.
RESTORATION_LEVELS = {
    1: dict(
        maximum_minor_std_m=18.0,
        minimum_trace_coherence=0.40,
        maximum_removal_fraction=0.55,
        fit_tolerance_m=1.75,
    ),
    2: dict(
        maximum_minor_std_m=30.0,
        minimum_trace_coherence=0.25,
        maximum_removal_fraction=0.80,
        fit_tolerance_m=3.5,
    ),
    3: dict(
        maximum_minor_std_m=45.0,
        minimum_trace_coherence=0.15,
        maximum_removal_fraction=0.90,
        fit_tolerance_m=5.0,
    ),
}


# Restoration POLICY entries that are distances in metres.
RESTORATION_LENGTHS = (
    "context_radius_m",
    "minimum_major_std_m",
    "maximum_minor_std_m",
    "maximum_flank_distance_m",
    "flank_sample_spacing_m",
    "maximum_growth_m",
    "along_distance_m",
)


def restoration_policy(level, length_scale=1.0, fit_scale=1.0):
    """Explicit strengths; recovery always checks original elevations within 8 m."""
    if level not in range(4):
        raise ValueError("terrain_restoration must be 0 through 3")
    from .restoration import POLICY

    policy = dict(POLICY)
    strength = dict(RESTORATION_LEVELS.get(level, {}))
    if length_scale != 1.0:
        for key in RESTORATION_LENGTHS:
            policy[key] *= length_scale
        if "maximum_minor_std_m" in strength:
            strength["maximum_minor_std_m"] *= length_scale
    if fit_scale != 1.0:
        policy["fit_tolerance_m"] *= fit_scale
        if "fit_tolerance_m" in strength:
            strength["fit_tolerance_m"] *= fit_scale
    return dict(policy, source_tolerance_m=8.0, **strength)
