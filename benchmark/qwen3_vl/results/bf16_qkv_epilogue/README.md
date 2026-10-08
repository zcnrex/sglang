# External B8 QKV epilogue prototype

Standalone correctness and performance gates passed. No production integration or serving claim is made. The actual-model diagnostic also passes: two-seed bitwise logits/tokens and a 1.105% fixed-state replay reduction; see `model/README.md`.

| Gate | Reference us/layer | Fused us/layer | Time reduction | Saving across 36 layers |
| --- | ---: | ---: | ---: | ---: |
| QKV + preparation, GPU2 | 9.9235 | 9.0413 | 8.89% | 31.759 us |
| QKV + preparation + TRT attention, GPU2 | 52.6313 | 51.6867 | 1.795% | 34.005 us |
| QKV + preparation + TRT attention, GPU3 | 52.5950 | 51.8204 | 1.473% | 27.888 us |

Every one of eight opposite-order pairs was positive in each gate. These are retained-graph whole-sequence intervals, not sums of profiler kernel durations. Inputs/caches remain fixed between timed replays, so these are synthetic steady-state diagnostics, not growing-context serving rates.

## Prototype contract

The external clone of installed FlashInfer `dense_bf16_gemm_sm100_splitk.py` preserves B8 tactic (128,8,2,6), MMA/DMA, FP32 DSMEM reduction and the GEMM BF16 output round. Its owner CTA redistributes the rounded 128x8 tile through an additional 2 KiB shared buffer. Four epilogue warps process eight token-head vectors in two iterations. Q/K normalization preserves sequential FP32 square accumulation, XOR reduction order and BF16 rounding before explicit BF16 rotary multiply/FMA. Q and K outputs are retained; rotated K and unchanged V also write the existing HND page 32 cache. Negative slots skip cache writes.

This prototype is intentionally restricted to B8, Q32/KV8/D128, BF16 inputs/weights/query/KV/output, epsilon 1e-6, no bias, full-width NeoX rotation, ordinary TP1 Qwen projection. It is not a registered public operator. Only external artifacts changed. The kernel organization skill was read; a future production implementation would require its own placement/registration and integration review.

The original early PDL trigger is preserved. The paired baseline is current GEMM plus current fused norm/rope/cache preparation. The dependent sequence adds the same actual TRT attention consumer and persistent counter buffer to both arms. Separate PDL source/SASS audit is maintained by the scheduler agent; these runtime checks complement that audit rather than claiming universal safety for every consumer variant.

## Correctness and cache discipline

Initial actual layer 0 correctness passes bitwise for full QKV, key cache and value cache, including a negative slot and disjoint valid slots. The 36-layer gates use all 36 distinct actual model QKV/norm weights, approximately 1.13 GB of weights, and disjoint full 8K HND caches (approximately 9.74 GB per arm). Sequence caches are seeded finite BF16 values and identical between arms. All 36 layers match bitwise for QKV, complete K/V caches and attention outputs initially and after hidden-input/position changes through retained graphs.

The separate `distinct_axes` fixture uses different values for all three position axes, verifies its interleaved [24,20,20] axis map against the actual model configuration and `MRotaryEmbedding._build_axis_map`, and passes bitwise initially and after changed-input graph replay. This is bounded kernel coverage, not image-model accuracy validation. Earlier timing fixtures use equal text-position rows, where contiguous versus interleaved axis maps are equivalent.

## Frozen evidence

`standalone-original.tar.gz` preserves raw source/scripts/logs/results. `standalone/` contains readable `.py.txt` copies and result/launch JSONs for `attempt1`, `pair_gpu2`, `sequence_gpu2`, `sequence_gpu3`, and `distinct_axes`. No failed compile attempts occurred. Source hashes are recorded in results, with the original installed GEMM source retained in `attempt1`. Remote root `/root/qvl/experiments/bf16-qkv-epilogue`.
