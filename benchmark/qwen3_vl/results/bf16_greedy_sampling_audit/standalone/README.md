# Greedy cast/argmax standalone cost

Bounded GPU 0 screen; no production changes or serving. PIDs 534574 (superseded) and 535466 (corrected) terminated normally. GPU 0 was empty before launch; unrelated work occupied GPUs 6/7 and was untouched.

Inputs are BF16 logits produced with the pinned checkpoint's actual tied embedding weight `[151936,2560]` and seeded synthetic hidden states. These are not saved model activations. Vocabulary GEMM is outside timed regions. Both implementations retain the full FP32 logits output and int64 token IDs. The candidate is a startup-compiled `torch.compile(fullgraph=True,dynamic=False)` callable, not a custom kernel or reduced-precision output.

The corrected baseline performs `out.copy_(bf16_logits)` followed by `torch.argmax(out,dim=-1,out=ids)`. The initial run unnecessarily copied the token-ID result, artificially adding baseline work; its positive 1.52 microsecond B128 difference is superseded and must not be used. Both scripts and reports are preserved.

| Batch | Cast alone, us | Argmax alone, us | Baseline pair, us | Compiled pair, us |
| --- | ---: | ---: | ---: | ---: |
| 32 | 12.310 | 16.417 | 29.072 | 46.013 |
| 64 | 18.479 | 20.535 | 40.411 | 77.941 |
| 128 | 53.253 | 30.510 | 84.034 | 84.168 |

These are medians of ten counterbalanced rounds, each timing 100 retained CUDA graph replays after five warmups. Inputs, outputs, graph owners and callables remain live. Separate component timings need not sum exactly to the pair because cache/order effects differ. The compiler's effective GPU code is measured as generated; this report does not assert that it emitted one fused kernel.

Initial and changed-input logits match bitwise as FP32 words, and token IDs match exactly. Additional rows verify the first index among equal maxima, first of multiple NaNs, all negative infinity, and first of multiple positive infinities; expected first IDs are 7/17/0/9 in both implementations. These finite tests are not a proof for all exceptional bit patterns or model accuracy.

No measured saving remains after correcting the baseline. Reject the compiled callable without a model or serving gate. The B128 baseline cost is about 0.084 ms, roughly 0.34% of a 25 ms decode step; this is measured standalone cost, not end-to-end recoverable headroom. This screen preserves the FP32 output and does not evaluate the separate idea of removing that output from model interfaces.

Compilation and tuning are excluded from timing. Clocks were not locked or sampled during the measurement; warmup and alternating order reduce but do not eliminate clock/cache sensitivity. Post-run clock observations are not presented as active measurement telemetry.
