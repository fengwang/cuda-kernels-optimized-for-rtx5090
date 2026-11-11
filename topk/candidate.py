"""Top-k candidate wrapper: loads its kernel build, allocates the outputs and launches the kernel.

values, indices = the k = 50 largest entries of each row of x[m, n] in descending order and their positions. The one
`launch` entry of libkernel.so serves both dtypes: fp32 (0) and bf16 (1), row-major x; values keep x's dtype and
indices are int64. `plan` picks the columns per block from the kernel's occupancy so the grid is one resident wave;
rows split into several chunks merge through an int64 workspace and a per-row int32 counter (zeroed once, reset by the
kernel). The plan and workspace of the last shape are cached, so a repeated call allocates only its outputs.
params.json (optional) may set "log2_threads" (7..9), "log2_depth" (1..3, tiles of loads in flight per block) and
"min_chunk".
"""

import ctypes
import json
import os

import torch

K = 50
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS_PATH = os.path.join(_HERE, "params.json")
_params = {}
if os.path.exists(_PARAMS_PATH):
    with open(_PARAMS_PATH) as _handle:
        _params = json.load(_handle)
_THREADS = 1 << int(_params.get("log2_threads", 8))
_DEPTH = 1 << int(_params.get("log2_depth", 2))
_MIN_CHUNK = int(_params.get("min_chunk", 2048))
_DTYPES = {torch.float32: 0, torch.bfloat16: 1}

_lib = ctypes.CDLL(os.path.join(os.environ["CKOPT_BUILD_DIR"], "libkernel.so"))
_lib.plan.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.plan.restype = ctypes.c_longlong
_lib.launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_longlong, ctypes.c_void_p]
_lib.launch.restype = ctypes.c_int

_counters = {}
_last = None  # (key, chunk, work pointer, counters pointer, work, counters) of the last shape


def _row_counters(device, rows):
    counters = _counters.get(device)
    if counters is None or counters.numel() < rows:
        counters = torch.zeros(rows if rows > 4096 else 4096, dtype=torch.int32, device=device)
        _counters[device] = counters
    return counters


def _plan(x, rows, cols, dtype):
    # Launches of one shape on the stream are serialized and every chunk block rewrites its workspace slots before the
    # row's last block reads them, so the workspace is reused; a new shape frees the old one before allocating.
    global _last
    key = (x.get_device(), rows, cols, dtype)
    if _last is not None and _last[0] == key:
        return _last
    _last = None
    chunk = _lib.plan(rows, cols, dtype, _THREADS, _DEPTH, _MIN_CHUNK)
    nchunks = -(-cols // chunk)
    work = torch.empty(rows * nchunks * K if nchunks > 1 else 1, dtype=torch.int64, device=x.device)
    counters = _row_counters(x.device, rows)
    _last = (key, chunk, work.data_ptr(), counters.data_ptr(), work, counters)
    return _last


def run(inputs):
    x = inputs["x"]
    dtype = _DTYPES.get(x.dtype)
    if dtype is None or x.dim() != 2 or x.shape[1] < K:
        raise ValueError(f"topk candidate takes a 2-D fp32 or bf16 tensor with at least {K} columns, "
                         f"got {x.dtype} with shape {tuple(x.shape)}")
    x = x.contiguous()
    rows, cols = x.shape
    _, chunk, work_ptr, counters_ptr, _, _ = _plan(x, rows, cols, dtype)
    values = torch.empty((rows, K), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, K), dtype=torch.int64, device=x.device)
    err = _lib.launch(x.data_ptr(), values.data_ptr(), indices.data_ptr(), work_ptr, counters_ptr,
                      rows, cols, K, dtype, _THREADS, _DEPTH, chunk, torch.cuda.current_stream().cuda_stream)
    if err:
        raise RuntimeError(f"kernel launch failed: CUDA error {err}")
    return {"values": values, "indices": indices}
