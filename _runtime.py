import os

for _thread_variable in (
    "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

_cache_root = (
    os.environ.get("LOCALAPPDATA")
    or os.environ.get("TMPDIR")
    or os.environ.get("TEMP")
)
if _cache_root:
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", os.path.join(_cache_root, "SLIC_numba_cache")
    )
