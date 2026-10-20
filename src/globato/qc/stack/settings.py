"""Small public controls; evidence sufficiency does not change with aggressiveness."""

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

    def __post_init__(self):
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
        import math

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


def restoration_policy(level):
    """Explicit strengths; recovery always checks original elevations within 8 m."""
    if level not in range(4):
        raise ValueError("terrain_restoration must be 0 through 3")
    from .restoration import POLICY

    return dict(POLICY, source_tolerance_m=8.0, **RESTORATION_LEVELS.get(level, {}))
