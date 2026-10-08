# Production M16 down validation

Control is committed `5babd15f4f` (M4/M8 LM-head paths included); candidate is `06d1a12172`. The 4,229-file source audit finds exactly one changed file, `unquant.py`, adding `(16, 2560, 9728): (128, 16, 4, 5)` to the existing map. No algorithm hook was used in these production runs; observer wrappers forward unchanged.

The actual production dispatcher selected split-K, with `_prefer_direct` false. Eager and changed-input CUDA graph output match the external winning kernel bitwise; against Torch the standalone normalized RMS error is 0.00009862. The corrected real-text B16 model probe observed 36 unique captured down weights and four finite decode steps. This proves the branch/capture, not numerical equivalence of complete models. The earlier initializer-alias observer failed its assertion after model initialization; its log is retained separately, with no claimed capture proof.

C16 serving used N80, warmup64, flushed caches, fixed 8192 input and 1024 output tokens, BF16 weights/queries/KV, TRT HND page32, mixed chunk16384, and normal public startup tuning. Two phases swap variants on the same GPUs. All four runs completed exactly 80 requests, 655360 input tokens and 81920 output tokens.

| GPU | Control output tok/s | Candidate output tok/s | Gain | TTFT control/candidate ms | TPOT control/candidate ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 2562.444 | 2575.846 | 0.523% | 599.139 / 601.580 | 5.65738 / 5.62964 |
| 1 | 2555.865 | 2565.170 | 0.364% | 601.040 / 607.427 | 5.70315 / 5.64546 |

Geometric mean throughput gain is 0.4433%. TTFT increased by 0.41% and 1.06%; this small two-pair experiment does not establish a latency improvement. Both candidate servers observed actual split-K graph capture, and all workers observed measured B16 decode forwards after the flush epoch. No runtime M128 forward was observed in any epoch; public startup tactic records remain archived rather than assumed equal. Decode tails may use the shared M4/M8 LM-head paths.

The production kernel is bitwise identical to the earlier external kernel whose matched-startup full GSM comparison was control 1221/candidate 1220 of 1314; that finite diagnostic is not proof of distributional equivalence. No new full production accuracy run was launched here.

Raw tar archives preserve original bytes; extracted files are normalized only by repository checks. `summarize.py.txt` reproduces the compact summary after copying it to `summarize.py` in this directory. Scripts retain exact source and observer logic; telemetry and server logs preserve timing/clock context. Parent 514694 and all four owned server processes terminated normally after phases A/B. The failed model observer PID 507744 and corrected model PID 512348 are both terminal.
