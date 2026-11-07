# RoPE kernel optimization

Speedup = reference time / incumbent time; below 1.0 is a slowdown.

|  | bf16 | fp16 |
| -- | -- | -- |
| Decode, batch 1 / 16 / 64      | 5.94× / 5.87× / 6.44×  | 6.47× / 6.58× / 6.75×  |
| Prefill, seq 512 / 2048 / 8192 | 4.89× / 5.36× / 10.65× | 5.26× / 5.40× / 10.71× |

