"""RMSNorm candidate wrapper: loads its kernel build, allocates y and launches the kernel.

y = x times rsqrt(the mean of x^2 over the last dim + 1e-6) for x[rows, cols] in bf16 (1) or fp16 (2), with that
rsqrt rounded to x's dtype before the product (the task's reference); the one `launch` entry of libkernel.so serves
both dtypes, and x is passed contiguous. params.json (optional) may set the threads per block as "threads" (a multiple
of 32 from 32 to 1024, default 256).
"""

import ctypes
import json
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARAMS_PATH = os.path.join(_HERE, "params.json")
_params = {}
if os.path.exists(_PARAMS_PATH):
    with open(_PARAMS_PATH) as _handle:
        _params = json.load(_handle)
_THREADS = int(_params.get("threads", 256))
_EPS = 1e-6
_DTYPES = {torch.bfloat16: 1, torch.float16: 2}

_lib = ctypes.CDLL(os.path.join(os.environ["CKOPT_BUILD_DIR"], "libkernel.so"))
_lib.launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
                        ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
_lib.launch.restype = ctypes.c_int


def _problem(x):
    """Why x is not an input this kernel serves, or None."""
    if x.dtype not in _DTYPES:
        return f"dtype {x.dtype}"
    if x.dim() != 2 or x.shape[1] < 1:
        return f"shape {tuple(x.shape)}"
    return None


def run(inputs):
    x = inputs["x"]
    problem = _problem(x)
    if problem:
        raise ValueError(f"rmsnorm candidate: {problem}")
    x = x.contiguous()
    y = torch.empty_like(x)
    rows, cols = x.shape
    err = _lib.launch(x.data_ptr(), y.data_ptr(), rows, cols, _DTYPES[x.dtype], _EPS, _THREADS,
                      torch.cuda.current_stream().cuda_stream)
    if err:
        raise RuntimeError(f"kernel launch failed: CUDA error {err}")
    return {"y": y}
