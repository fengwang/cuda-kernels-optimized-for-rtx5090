"""Reference for the W4 softmax task: y = softmax(x) over the last dim of x[m, n] (harness side; candidates may not
call it).

Adapted from KernelBench level 1, problem 23_Softmax (MIT; see ../NOTICE): the Model's torch.softmax(x, dim=1)
becomes dim=-1 (the same axis for the task's 2-D inputs), and run(inputs) is the harness entry point. The harness
calls it at the shape's dtype (fp32, or bf16 with torch's fp32 accumulation) and in fp64 (the oracle).
"""

import torch


def run(inputs):
    return {"y": torch.softmax(inputs["x"], dim=-1)}
