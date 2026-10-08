# Immutable production cast model gate

Candidate `/root/qvl/sglang-cast-production` is the frozen M16 source plus only the three-file committed cast patch `f9a704d70a`. It excludes later M32/M64 vocabulary selections. Exact patch and recursive source manifests are retained. Baseline comparison uses the original M16 `_copy_logits_to_buffer` method extracted unchanged into the same loaded model; candidate invokes the unmodified production method. A wrapper observes the public cast kernel without replacing its computation. Other model operations and startup tactics are shared within each pair.

Production Triton kernels warm before candidate graph capture. B64 observes one target capture and two eager warm calls; B128 observes two captures and four eager calls because both 64/128 buckets qualify. Every call returns the exact supplied FP32 buffer. No torch.compile is used by the production optimization.

Input length 128, output length 16, real model forward at batch 64/128. Prefill, first decode and last decode full logits match bitwise; all generated tokens agree. A changed seed also matches final full logits and tokens. Both retained graphs share input/KV-position buffers; all snapshots, including full FP32 logits, remain unchanged through fixed-state replay.

| Batch | Baseline graph ms | Production cast graph ms |
| --- | ---: | ---: |
| 64 | 2.817608 | 2.804440 |
| 128 | 3.279959 | 3.248562 |

Six counterbalanced pairs, 256 replays per graph after common warmup. These are short-context fixed-state GPU-event timings, not successive-token serving throughput. Reports additionally retain actual 15-step timings and separate untimed warmed traces. No clock locking; post-run clock snapshots are not continuous telemetry. Full model logits tensors remain remote, with SHA256 in summary; exact compact comparisons are archived.

PIDs 582094/582158 on GPUs 0/1 terminated normally. This gate passed before the separately archived production serving crossover. It does not establish long-context/full-task accuracy by itself.
