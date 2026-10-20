"""Publication only. Invoked after native Globato interpolation has completed."""

import logging
import traceback
from pathlib import Path
from fetchez.hooks import FetchHook

logger = logging.getLogger(__name__)


class ReviewPublisher(FetchHook):
    name = "stack_qc_review_publisher"
    meta_stage = "collection"

    def __init__(self, *, cache, result, stack, region, interpolator, module_name):
        super().__init__()
        self.cache = Path(cache)
        self.result = result
        self.stack = stack
        self.region = region
        self.module_name = module_name
        self.output = None
        output = getattr(interpolator, "output", None)
        if not output:
            raise ValueError(
                "Stack QC review=true needs an explicit final DEM output path"
            )
        self.final = Path(output).resolve()

    def run(self, entries):
        from .visualize import completed

        if not self.final.is_file():
            raise RuntimeError(
                f"Stack QC review=true: normal Globato final DEM is missing: {self.final}"
            )
        # Single-module builds keep review.html at top level. Multiple review
        # attachments each get their own comparison folder.
        output = self.output or self.final.parent
        output.mkdir(parents=True, exist_ok=True)
        # Remove an old page before generating a new one in the same folder.
        # Old individual PNGs are harmless without a current review index.
        (output / "review.html").unlink(missing_ok=True)
        try:
            images = completed(
                output,
                cache=self.cache,
                result=self.result,
                stack=self.stack,
                region=self.region,
                final=self.final,
            )
            (output / "review_ERROR.txt").unlink(missing_ok=True)
            logger.info(
                "Stack QC review: %d figures; %s", len(images), output / "review.html"
            )
        except Exception:
            error = output / "review_ERROR.txt"
            error.write_text(
                "Stack QC review failed; the production DEM is unchanged.\n\n"
                + traceback.format_exc()
            )
            logger.exception(
                "Stack QC review failed; final DEM is preserved: %s", error
            )
        return entries
