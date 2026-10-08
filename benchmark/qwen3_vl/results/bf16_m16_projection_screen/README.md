# M16 QKV and down projection screen

Exact shapes M16/N6144/K2560 and M16/N2560/K9728 were absent from the retained small-batch split-K results (M1/2/4/8). The high-M TGV archive covers M64/128, not these cases. Production selectors were invoked and asserted to choose F.linear for both M16 shapes. No production edits or new kernel implementation.

Actual pinned Qwen weights from layers 0–15 rotate inside each CUDA graph: QKV concatenates checkpoint Q/K/V exactly in projection order; down uses the original tensor. Total rotating weights are approximately 503 MB / 797 MB, exceeding L2. Inputs and outputs stay BF16. Three selected existing split-K configurations and public startup-tuned cuBLASLt were compared with actual F.linear. Six alternating graph timing rounds normalize 30 replays of 16 GEMMs. All graph/input/output owners are retained; changed-input replay checked before timing. Tuning/compilation is excluded.

| Shape/config | Production / candidate median (us) | Median paired gain |
|---|---|---|
| QKV public Lt | 7.402 / 7.388 | +0.24% |
| QKV split-K (128,8,2,6) | 7.409 / 10.620 | −30.23% |
| QKV split-K (128,16,2,6) | 7.403 / 6.793 | +8.96% |
| QKV split-K (64,16,4,9) | 7.404 / 10.615 | −30.24% |
| Down public Lt | 11.673 / 11.654 | +0.15% |
| Down split-K (128,8,4,6) | 11.683 / 14.982 | −22.02% |
| Down split-K (64,16,4,9) | 11.665 / 12.657 | −7.79% |
| Down corrected (128,16,4,5) | 11.674 / 10.255 | +13.84% |

Tuple fields are mma_m, mma_n, split_k, ab_stages. Down (128,16,4,6) was rejected before launch because it requires 245896 B shared memory versus 232448 B available. The compiler reported maximum 5 stages, so exactly one corrected stage 5 test was added; this is a resource admission failure, not a CUDA fault. Original error remains in down/report.json.

Independent second-GPU confirmation of only the winners produced QKV +8.918% and down +13.740%, all six paired rounds positive each. Public Lt was effectively flat and was not advanced. Correctness used NRMS<0.005; best QKV observed NRMS 0.00009265 (confirmation 0.00012015), max abs 0.015625, and down NRMS 0.00013120, max abs 0.03125. They are not bitwise equal to F.linear. All tested winner argmax rows matched; projection argmax is a diagnostic only, not a model token guarantee. Changed-input graph NRMS also passed and is recorded. Subsequent model gates must check accumulated effects.

Remote workers qkv463192, down463193, corrected down463925, QKV confirmation464297 and down confirmation464579 all terminated. Source `/root/qvl/sglang-prefix-production`; pinned model revision ebb281ec70b05090aa6165b016eac8ec08e71b17. Separate isolated model checks are recorded elsewhere; no serving claim follows from this table.

## Exact script hashes

- `qvl-m16-projection.py`: `e339ee4c9deda2ef0b2ace87f72768295ec99356cc5e0dcbbe71d1e23e35ea98`
- `qvl-m16-down-stage5.py`: `d8ea325e8fc66829b24c1e7353897e5f8c4f652eeb437c3ed5ad459282652b85`
- `qvl-m16-qkv-confirm.py`: `07b9cb85770137cb04bdfb934ac29bf8e630a07026629446f42be587d2d73451`
