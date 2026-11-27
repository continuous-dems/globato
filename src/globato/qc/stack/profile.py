"""Small reporting helpers; these values never enter scientific decisions."""

from time import perf_counter

from . import io


SCHEMA = "stack_qc_profile/1"


def timed_stream(stream, receipt):
    """Time producer next() calls, excluding time spent in downstream consumers.

    On a miss the producer includes upstream reading/reprojection and QC-grid
    materialization. Those cannot be separated at this lazy-stream boundary.
    """
    receipt.update(seconds=0.0, status="pending")
    iterator = iter(stream)
    try:
        while True:
            started = perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                receipt["seconds"] += perf_counter() - started
                receipt["status"] = "complete"
                return
            except BaseException:
                receipt["seconds"] += perf_counter() - started
                receipt["status"] = "failed"
                raise
            receipt["seconds"] += perf_counter() - started
            yield item
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def write_stage(path, stage, seconds, inputs, outputs):
    """Publish a stage receipt without reopening or hashing large rasters."""
    io.write(
        path,
        dict(
            schema=SCHEMA,
            stage=stage,
            seconds=seconds,
            clock="perf_counter",
            inputs=inputs,
            outputs=outputs,
        ),
    )
