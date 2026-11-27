"""Worker pools for Stack QC's block-parallel steps.

Workers are spawned (no inherited locks or GDAL state) and stop when the build
that started them dies, so killing a build leaves no orphaned workers.
"""

import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor


def _stop_with_parent(parent):
    """Ask Linux to terminate this worker when its parent exits."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass  # Not Linux: workers still exit when the pool shuts down.
    if os.getppid() != parent:
        os._exit(1)  # The parent died before the request took effect.


def _start(parent, initializer, initargs):
    _stop_with_parent(parent)
    if initializer is not None:
        initializer(*initargs)


def process_pool(workers, initializer=None, initargs=()):
    """A spawn-context ProcessPoolExecutor whose workers die with this process."""
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start,
        initargs=(os.getpid(), initializer, initargs),
    )
