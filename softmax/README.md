# Softmax Optimization Report

Speedup = reference time / incumbent time; below 1.0 is a slowdown.

| workload | correctness | timing | candidate ms | baseline ms | seed ms | vs baseline | vs seed | dtype |
|---|---|---|---|---|---|---|---|---|
| s4096x1024_fp32 | PASS | PASS | 0.02605 | 0.0263 | 0.0281 | 1.01 | 1.079 | fp32 |
| s4096x8192_fp32 | PASS | PASS | 0.1876 | 0.1838 | 0.2043 | 0.9795 (slowdown) | 1.089 | fp32 |
| s256x262144_fp32 | PASS | PASS | 0.3577 | 0.6937 | 0.8503 | 1.94 | 2.377 | fp32 |
| s4096x1024_bf16 | PASS | PASS | 0.01786 | 0.0217 | 0.02365 | 1.215 | 1.324 | bf16 |
| s4096x8192_bf16 | PASS | PASS | 0.09757 | 0.114 | 0.1042 | 1.168 | 1.068 | bf16 |
| s256x262144_bf16 | PASS | PASS | 0.1875 | 0.3109 | 0.6633 | 1.658 | 3.538 | bf16 |

