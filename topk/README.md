# topk

Speedup = reference time / incumbent time; below 1.0 is a slowdown.

| workload | correctness | timing | candidate ms | baseline ms | seed ms | vs baseline | vs seed | dtype |
|---|---|---|---|---|---|---|---|---|
| s1x128256_fp32 | PASS | PASS | 0.02019 | 0.05414 | 1.161 | 2.681 | 57.51 | fp32 |
| s64x151936_fp32 | PASS | PASS | 0.04304 | 0.1415 | 1.429 | 3.288 | 33.2 | fp32 |
| s512x32000_fp32 | PASS | PASS | 0.05549 | 0.2399 | 1.08 | 4.323 | 19.46 | fp32 |
| s1x128256_bf16 | PASS | PASS | 0.02048 | 0.04003 | 1.188 | 1.955 | 58 | bf16 |
| s64x151936_bf16 | PASS | PASS | 0.0369 | 0.1039 | 1.453 | 2.817 | 39.37 | bf16 |
| s512x32000_bf16 | PASS | PASS | 0.03891 | 0.1701 | 0.6538 | 4.372 | 16.8 | bf16 |

