# RoPE kernel optimization

Speedup = reference time / incumbent time; below 1.0 is a slowdown.

| workload | correctness | timing | candidate ms | baseline ms | seed ms | vs baseline | vs seed | dtype |
|---|---|---|---|---|---|---|---|---|
| s1x1_bf16 | PASS | PASS | 0.01203 | 0.07142 | 0.01238 | 5.936 | 1.029 | bf16 |
| s16x1_bf16 | PASS | PASS | 0.01235 | 0.07254 | 0.01402 | 5.873 | 1.135 | bf16 |
| s64x1_bf16 | PASS | PASS | 0.01222 | 0.07869 | 0.01411 | 6.437 | 1.154 | bf16 |
| s1x512_bf16 | PASS | PASS | 0.01629 | 0.07968 | 0.02032 | 4.892 | 1.248 | bf16 |
| s1x2048_bf16 | PASS | PASS | 0.03949 | 0.2117 | 0.04301 | 5.362 | 1.089 | bf16 |
| s1x8192_bf16 | PASS | PASS | 0.1256 | 1.337 | 0.1295 | 10.65 | 1.031 | bf16 |
| s1x1_fp16 | PASS | PASS | 0.01194 | 0.07718 | 0.01395 | 6.466 | 1.169 | fp16 |
| s16x1_fp16 | PASS | PASS | 0.01219 | 0.08026 | 0.01408 | 6.583 | 1.155 | fp16 |
| s64x1_fp16 | PASS | PASS | 0.01222 | 0.0825 | 0.01414 | 6.749 | 1.157 | fp16 |
| s1x512_fp16 | PASS | PASS | 0.01638 | 0.08624 | 0.02058 | 5.264 | 1.256 | fp16 |
| s1x2048_fp16 | PASS | PASS | 0.03952 | 0.2134 | 0.04339 | 5.401 | 1.098 | fp16 |
| s1x8192_fp16 | PASS | PASS | 0.1249 | 1.338 | 0.1276 | 10.71 | 1.022 | fp16 |

