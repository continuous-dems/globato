"""Hook-owned, single-original-raster scientific entry point.

Retains a separate scientific subprocess to release large raster arrays before
mask publication. The unchanged regional science owner receives direct inputs
rather than a fabricated study catalogue or tile-runner recipe.
"""

import argparse
from pathlib import Path
from .raster_science import run, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("area", "evidence", "output", "cache"):
        parser.add_argument("--" + item, required=True)
    for item in ("water-mask", "overwater-structures", "context-region"):
        parser.add_argument("--" + item, required=True)
    for item in ("ocean-anchor", "mapped-water-areas", "mask-layer"):
        parser.add_argument("--" + item)
    parser.add_argument(
        "--anchor-scope",
        choices=("external", "declared"),
        default="external",
        help="Internal normalized-ocean evidence scope",
    )
    a = parser.parse_args()
    a.remove = "ocean"
    a.overwater_structures_layer = "mask"
    a.shore_level_m = 0.0
    a.threshold_m = 0.01
    a.inland_m = 100.0
    a.output = Path(a.output).expanduser().resolve()
    a.cache = Path(a.cache).expanduser().resolve()
    if a.output.exists():
        parser.error("Science output must be new")
    a.output.mkdir(parents=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    try:
        run(a, a.output)
    except Exception as exc:
        record(a.output / "status.json", {"status": "ERROR", "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
