# Down-only M16 serving and subset sanity

Both variants use `/root/qvl/sglang-lmhead-production`, exact committed 5e1601731b equivalent: all 4229 expected source hashes verified. The existing M4 LM-head improvement is shared. Candidate changes only the external down projection M16 dispatch to split-K(128,16,4,5), startup compiled on an actual model weight. No QKV combination or production edits. BF16 weights/activations/KV/output, HND page 32, TRT decode, mixed chunk 16384, normal public startup autotuning.

Two same-GPU crossover phases at C16, N80, warmup 64, input 8192 / output 1024 with flush before measurement. All four runs completed 80 requests, 655360 input tokens and 81920 output tokens.

| GPU | Control / candidate output tok/s | Gain | TTFT (ms) control / candidate | TPOT (ms) control / candidate |
|---|---|---|---|---|
| 0 | 2561.571 / 2574.217 | +0.4937% | 601.341 / 591.030 | 5.66999 / 5.62740 |
| 1 | 2554.412 / 2562.373 | +0.3116% | 622.693 / 629.775 | 5.68160 / 5.65385 |

Geometric mean gain is +0.4026%. TTFT improved on one GPU and regressed on the other. The observer emits flush epochs and logs each actual ModelRunner.forward with 128 input rows; none occurred in any worker, including measured epoch 1. This is observed row-count coverage, not inference solely from client concurrency. No prefill graphs or explicit larger decode-bucket override are used. Normal M128 startup tactics are preserved in summary.json and raw startup caches. Candidate READY/capture and real decode 16 forward markers are retained.

The paired 64-question GSM sanity (five-shot rows 0–4, evaluated rows 5–68, C16, greedy max 2048) scored control 61/64 and candidate 60/64. Row 20 changed from correct to wrong; rows 12/62/67 were wrong in both. This small adverse difference does not quantify population accuracy, but prevents claiming numerical equivalence or an accuracy pass. Prior teacher-forced 255/256 agreement and synthetic 24/256 free-running changes remain relevant. No full 1314 evaluation was launched for this candidate.

The first subset attempt completed requests but failed before persisting results: its copied evaluator at `/tmp/qvl-m16-eval-gsm8k.py` indexed parents[2], unavailable at that shallow path. Original failure logs are retained under raw/m16-down-accuracy. Exactly one retry placed the unchanged evaluator at `/root/qvl/experiments/m16-down-eval/eval_gsm8k.py`; this fixed metadata path resolution without modifying scoring. Corrected results are under raw/m16-down-accuracy-fixed. Source revision there is null because the experiment directory is not a Git checkout; source_audit.json supplies provenance.

All jobs exited. Serving parent 466135; phase A servers 466144/466145; phase B servers 472078/472079. Corrected accuracy drivers 475301/475302 and servers 475303/475304. Raw-original.tar.gz preserves unnormalized logs/results/cache files. No promotion decision is implied by the throughput gate.

## Exact harness hashes

- `qvl-m16-down-serving-hook.py`: `a3d7e7c8d720da4f3be9f5a0a7efe373fe7ca4f9f4a38cf2ce6a6778ffde0c4e`
- `qvl-m16-down-serving.py`: `c1351655e37f4255c69e570603073e2ab47101bf3a062a4fc1ef579e479a1e6e`
- `qvl-m16-down-launch.py`: `1f3513af9d912283beccd0ec18f0bf27161e9045940c2ad03a3dccc5fdcd072e`
- `qvl-m16-down-gsm.py`: `d1cc4802307b4e4a07ed587a5b60c0c62a95d96a98e20cf9a2b27c5093768b2f`
- `qvl-m16-down-gsm-fixed.py`: `84a8e589a3937ebcad500e08423f4e8a73f255299b351e420cf7aaac0328b89e`
- `qvl-m16-down-accuracy-launch.py`: `51e6fe1661ea88aa24589ce2e46b93348a2101a50f31e91437b2bf3e6f11dc67`
