"""Optional compiled path loop with a NumPy fallback and explicit build receipt."""

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

_library = None


def prepare(folder, backend="auto"):
    global _library
    _library = None
    os.environ.pop("GLOBATO_STACK_QC_PATH_LIBRARY", None)
    start = time.perf_counter()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if backend == "numpy":
        return {"backend": "numpy", "setup_seconds": time.perf_counter() - start}
    compiler = shutil.which("cc") or shutil.which("gcc")
    source = Path(__file__).with_name("path_kernel.c")
    path = folder / "path_kernel.so"
    if not compiler:
        if backend == "native":
            raise RuntimeError("A C compiler is required for --backend native")
        return {
            "backend": "numpy",
            "reason": "No C compiler found",
            "setup_seconds": time.perf_counter() - start,
        }
    flags = [
        "-O3",
        "-std=c99",
        "-shared",
        "-fPIC",
        "-fno-fast-math",
        "-ffp-contract=off",
    ]
    proc = subprocess.run(
        [compiler, *flags, str(source), "-o", str(path), "-lm"],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        if backend == "native":
            raise RuntimeError("Native path compilation failed: " + proc.stderr)
        return {
            "backend": "numpy",
            "reason": proc.stderr,
            "setup_seconds": time.perf_counter() - start,
        }
    os.environ["GLOBATO_STACK_QC_PATH_LIBRARY"] = str(path.resolve())
    _library = load()
    return {
        "backend": "native",
        "compiler": compiler,
        "flags": flags,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "setup_seconds": time.perf_counter() - start,
        "library": str(path.resolve()),
    }


def load():
    global _library
    if _library is not None:
        return _library
    path = os.environ.get("GLOBATO_STACK_QC_PATH_LIBRARY")
    if not path:
        return None
    lib = ctypes.CDLL(path)
    array = np.ctypeslib.ndpointer
    i32 = array(dtype=np.int32, flags="C_CONTIGUOUS")
    f64 = array(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.path_masses.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        i32,
        f64,
        f64,
        f64,
        i32,
        ctypes.c_double,
        f64,
        f64,
        f64,
        f64,
    ]
    lib.path_masses.restype = ctypes.c_int
    _library = lib
    return lib


def measure(
    g, seed, centers, steps, radius, tolerance, rejected, accepted, support, squared
):
    lib = load()
    if lib is None:
        return False
    n = len(seed)
    if n >= 2**31:
        raise ValueError("Working graph exceeds int32 node range")
    loss = (g["raw_cost"] / tolerance**2 - np.log(np.maximum(g["spatial"], 1e-30))) * g[
        "lengths"
    ]
    args = [
        np.ascontiguousarray(v, dtype=d)
        for v, d in [
            (g["links"], np.int32),
            (g["lengths"], np.float64),
            (loss, np.float64),
            (seed, np.float64),
            (centers, np.int32),
        ]
    ]
    lib.path_masses(
        n, len(centers), steps, *args, radius, rejected, accepted, support, squared
    )
    return True
