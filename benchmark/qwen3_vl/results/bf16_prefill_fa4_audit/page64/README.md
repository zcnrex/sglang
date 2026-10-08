# Direct paged page64/tile64: rejected at prefill gate

This external diagnostic calls internal `_flash_attn_fwd` with
`tile_mn=(128,64)` and explicit CLC. The public varlen API has no tile override;
no production API or implementation was changed. Matching page64 to tileN64
retains TMA and CLC, but doubles the number of KV tiles versus tileN128.
The exact source manifest is verified before CUDA; native HND BF16 caches,
Q32/KV8/D128 and strided fresh QKV are retained.

Only the measured prefix/query tuple [7968,0,0]/[232,8200,7872] was tested.
Both variants include the maintained cache writer. Four alternating rounds
compare current page32 packing plus FA4tile128 against direct page64 FA4tile64.

| GPU | Cold page32 us | Cold page64 us | Regression |
| --- | ---: | ---: | ---: |
| 2 | 819.23 | 947.22 | 15.62% |
| 3 | 775.07 | 861.24 | 11.12% |

Warm medians also regress21.6–29.3%; writer costs are nearly equal. Raw rounds
show clock drift but provide no winning gate. KV conversion is bitwise exact;
attention outputs are not bitwise, but passed the same TRT tolerance. A FP32
reference at12 query positions per request gives comparable relative errors
for both paths; maximum TRT disagreement is0.0078125 for tile64.

No decode, model, or serving run followed this negative prefill result.
Scripts are exact `.py.txt` archives. GPUs2/3 completed and were released.

`original-artifacts.tar.gz` preserves raw files before repository formatting; recorded hashes refer to extracted originals.
