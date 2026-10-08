# Low-batch BF16 decode attention: rejected

The existing public FlashInfer FA2 tensor-core paged decode wrapper is slower than production TRT at both previously unmeasured low-batch shapes. No model or serving follow-up is warranted, and no production code changed.

| Batch / GPU | TRT median us/layer | FA2 median us/layer | FA2 slowdown |
| --- | ---: | ---: | ---: |
| 8 / GPU2 | 43.2585 | 50.8571 | 17.5656% |
| 16 / GPU3 | 80.4195 | 89.8558 | 11.7338% |

All eight opposite-order pairs were negative for each shape. These are standalone measurements, not serving rates. Both owned processes (584718/584719) completed and released GPUs2/3.

## Admission and prior exclusions

The retained real-model B8 baseline graph trace at `/root/qvl/experiments/lmhead-m8-diagnostic/m8/baseline-graph-trace.json` contains 36 TRT attention calls totaling 1591.43us, mean 44.206us/layer. The separate unprofiled model graph is 3531.907us; profiling overhead prevents directly interpreting their ratio as an exact production share. This identifies a sizable cost, unlike the already rejected marginal M32 vocabulary extension. A hypothetical 5us/layer improvement would save 180us per 36-layer decode, but the screen found a regression instead.

Prior raw Triton attention sweeps cover B128/L8192; the public FA2 same-cache comparison in `../mixed_decode_screen` covers B39 and was 27.03% slower. The earlier followup script archive contains norm, context-bound and CPU-profile scripts, not this low-batch comparison. No new tile, page-size, layout, maximum-length or PDL sweep was run.

## Exact screen

BF16 Q/K/V/output, 32 query heads, 8 KV heads, dimension 128, HND page 32. Lengths are 8192+i across each batch. Query rows retain actual QKV-split stride 6144. Each of 36 independent layer states owns separate random finite BF16 K/V and Q storage; physical pages are shuffled and shared identically by the two backends. Total rotating physical cache is 9,696,706,560bytes for B8 and 19,398,131,712bytes for B16. Each full 36-layer graph traversal therefore exceeds L2 by orders of magnitude; this is not repeated timing of one cache-hot layer. Values are seeded synthetic inputs, not saved model KV.

Candidate uses `BatchDecodeWithPagedKVCacheWrapper(..., backend="fa2", use_tensor_cores=True)` with planning outside timing. TRT uses production max-sequence bound 262144 and a persistent zero-initialized multi-CTA counter buffer sized by the installed utility for 8192 requests. Both use 512MiB workspaces, identical softmax scale 1/sqrt(128), and explicit BF16 outputs.

The retained CUDA graphs each contain 36 layer calls. Both graphs are replayed three times after capture before timing, avoiding first-instantiation contamination. Eight paired rounds alternate order; each interval times 12 full graph replays and divides by 432 calls. No tuning, planning or reference calculation occurs in timed intervals. Cache traffic figures describe allocated/input geometry; no new NCU bandwidth claim is made.

## Correctness and limits

All 36 independent states pass cross-backend normalized RMS error below 0.005. Every request in state 0 is separately compared against FP32 matmul/softmax reference with TF32 disabled; maximum normalized RMS errors are 0.002851 (B8) and 0.002904 (B16). Changed-query eager outputs and a second changed-input graph replay pass atol 0.002 / rtol 0.02. Both Q/KV/output remain BF16; differing accumulation order need not give bitwise output equality. This limited correctness gate does not claim full model accuracy equivalence.

Raw scripts, launch records, graph dumps, logs and JSON are preserved in `raw_evidence.tar.gz`; readable script copies are `.py.txt`. The exact script SHA256 is recorded in both launch records. Remote evidence root is `/root/qvl/experiments/low-batch-attention-screen`. No GPU jobs remain and no follow-up is queued.
