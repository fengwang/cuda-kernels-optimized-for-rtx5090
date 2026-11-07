"""RoPE candidate wrapper: loads its kernel build, allocates q_out and k_out and launches the kernel.

q_out, k_out = q, k rotated by the cos/sin cache rows that position_ids pick (half-split pairing, full rotary, fp32
math, one rounding to the output dtype), out-of-place. The one `launch` entry of libkernel.so serves q/k in bf16 (1)
or fp16 (2) and a cache in fp32 (0), bf16 (1) or fp16 (2); tensors are passed contiguous. params.json (optional) may
set the heads a block rotates at once as "rows" (a block is head_dim/16 x rows threads) and the blocks that split a
token's heads as "splits".
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
_ROWS = int(_params.get("rows", 40))
_SPLITS = int(_params.get("splits", 1))
_QK = {torch.bfloat16: 1, torch.float16: 2}
_CACHE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}

_lib = ctypes.CDLL(os.path.join(os.environ["CKOPT_BUILD_DIR"], "libkernel.so"))
_lib.launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_longlong] + [ctypes.c_int] * 7 + [ctypes.c_void_p]
_lib.launch.restype = ctypes.c_int


def _problem(q, k, cos, sin, ids):
    """Why these inputs are not a rope call this kernel serves, or None."""
    if q.dtype not in _QK or k.dtype != q.dtype or cos.dtype not in _CACHE or sin.dtype != cos.dtype:
        return f"dtypes q {q.dtype}, k {k.dtype}, cos {cos.dtype}, sin {sin.dtype}"
    if ids.dtype != torch.int64 or q.dim() != 4 or k.dim() != 4 or ids.dim() != 2 or cos.dim() != 2:
        return f"position_ids {ids.dtype} {tuple(ids.shape)}, q {tuple(q.shape)}, k {tuple(k.shape)}"
    if tuple(q.shape[:2]) != tuple(ids.shape) or tuple(k.shape[:2]) != tuple(ids.shape) or \
            tuple(sin.shape) != tuple(cos.shape) or not q.shape[3] == k.shape[3] == cos.shape[1] or q.shape[3] % 2:
        return f"shapes q {tuple(q.shape)}, k {tuple(k.shape)}, cos {tuple(cos.shape)}, position_ids {tuple(ids.shape)}"
    return None


def run(inputs):
    q, k, cos, sin, ids = (inputs[n] for n in ("q", "k", "cos", "sin", "position_ids"))
    problem = _problem(q, k, cos, sin, ids)
    if problem:
        raise ValueError(f"rope candidate: {problem}")
    q, k, cos, sin, ids = (t.contiguous() for t in (q, k, cos, sin, ids))
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    batch, seq, q_heads, head_dim = q.shape
    err = _lib.launch(q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(), ids.data_ptr(), q_out.data_ptr(),
                      k_out.data_ptr(), batch * seq, q_heads, k.shape[2], head_dim, _QK[q.dtype], _CACHE[cos.dtype],
                      _ROWS, _SPLITS, torch.cuda.current_stream().cuda_stream)
    if err:
        raise RuntimeError(f"kernel launch failed: CUDA error {err}")
    return {"q_out": q_out, "k_out": k_out}
