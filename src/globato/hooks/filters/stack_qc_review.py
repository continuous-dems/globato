"""HTML before/after review of Stack QC results, written after the final DEM.

Optional companion to ``stack_qc``. Put it in ``global_hooks`` after the final
interpolation (and its ``focus_sink``), so the page can show the finished DEM:

    - name: stack_qc_review
      args: {dem: "%name%_%batch_name%.tif"}

For every Stack QC result used in the build it writes ``review.html`` with
hillshades of the module's working DEM before filtering, after each pass,
after terrain restoration, and of the final combined DEM. It reads only the
cached QC results and never changes the DEM. A failure is logged and written
to ``review_ERROR.txt``; the build continues.
"""

import json
import logging
import os
import traceback
from pathlib import Path

from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)


class StackQCReview(FetchHook):
    """Write Stack QC review pages for the filtered modules."""

    name = "stack_qc_review"
    meta_stage = "collection"
    meta_category = "metadata"
    meta_desc = "Before/after hillshade review pages for Stack QC results."
    affects_fusion_state = False
    _cli_arg_help = {
        "dem": "Final DEM to show (default: the entry the build is focused on).",
        "output": "Folder for the review pages (default: <DEM folder>/stack_qc_review).",
    }

    def __init__(self, dem=None, output=None, **kwargs):
        super().__init__(stage="collection", **kwargs)
        self.dem = dem
        self.output = output
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("stack_qc_review requires matplotlib") from exc

    def _final_dem(self, entries):
        if self.dem:
            return Path(self.dem).resolve()
        for _, entry in reversed(entries):
            path = entry.get("dst_fn")
            if path and str(path).endswith(".tif") and os.path.isfile(path):
                return Path(path).resolve()
        return None

    def run(self, entries):
        from fetchez.spatial import Region
        from globato.hooks.sinks.multi_stack import MultiStackHook
        from globato.qc.stack.review.visualize import completed

        receipts = []
        for _, entry in entries:
            path = (entry.get("artifacts") or {}).get("stack_qc")
            if path and path not in receipts:
                receipts.append(path)
        if not receipts:
            logger.info("stack_qc_review: no Stack QC results in this build")
            return entries
        final = self._final_dem(entries)
        if final is None or not final.is_file():
            logger.warning("stack_qc_review: final DEM not found; no review written")
            return entries
        base = Path(self.output) if self.output else final.parent / "stack_qc_review"
        for path in receipts:
            receipt = json.loads(Path(path).read_text())
            cache = Path(receipt["cache"])
            run = base / f"{receipt['module']}_{cache.name[:12]}"
            run.mkdir(parents=True, exist_ok=True)
            (run / "review.html").unlink(missing_ok=True)
            try:
                stack = MultiStackHook(**receipt["stack"], output="review_stack.tif")
                west, east, south, north = receipt["region"][:4]
                region = Region(west, east, south, north, srs=receipt.get("region_srs"))
                result = dict(
                    receipt["result"],
                    qc_res=receipt["stack"]["res"],
                    dem_resolution=receipt["dem_res"],
                )
                images = completed(
                    run,
                    cache=cache,
                    result=result,
                    stack=stack,
                    region=region,
                    final=final,
                )
                (run / "review_ERROR.txt").unlink(missing_ok=True)
                logger.info(
                    "stack_qc_review: %d figures; %s", len(images), run / "review.html"
                )
            except Exception:
                (run / "review_ERROR.txt").write_text(
                    "Stack QC review failed; the DEM is unchanged.\n\n"
                    + traceback.format_exc()
                )
                logger.exception(
                    "stack_qc_review failed; see %s", run / "review_ERROR.txt"
                )
        return entries
