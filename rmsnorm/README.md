# RMSNorm

Speedup = reference time / incumbent time; below 1.0 is a slowdown.

| workload | correctness | timing | candidate ms | baseline ms | seed ms | vs baseline | vs seed | dtype |
|---|---|---|---|---|---|---|---|---|
| s1_bf16 | PASS | PASS | 0.01178 | 0.01907 | 0.016 | 1.62 | 1.359 | bf16 |
| s1024_bf16 | PASS | PASS | 0.01712 | 0.0568 | 0.03059 | 3.318 | 1.787 | bf16 |
| s32768_bf16 | PASS | PASS | 0.3608 | 2.793 | 0.3849 | 7.741 | 1.067 | bf16 |
| s1_fp16 | PASS | PASS | 0.01181 | 0.01891 | 0.01603 | 1.602 | 1.358 | fp16 |
| s1024_fp16 | PASS | PASS | 0.01718 | 0.0567 | 0.03059 | 3.3 | 1.78 | fp16 |
| s32768_fp16 | PASS | PASS | 0.3608 | 2.792 | 0.3849 | 7.738 | 1.067 | fp16 |

