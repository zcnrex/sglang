# C64 vocabulary short serving screen

Two short same-GPU pairs show a positive signal, not full N320 validation or a precise effect estimate. No new production change or accuracy evaluation follows from this screen alone.

| GPU | Control / candidate tok/s | Throughput change | Median TTFT change | Median TPOT change |
|---|---|---|---|---|
|4|3489.283590 / 3500.979514|+0.33520%|−8.20661%|+0.24613%|
|5|3522.238352 / 3534.621561|+0.35157%|+29.28320%|−0.89977%|

C64/N128/warm64, 8192 input/1024 output tokens, flush after warmup. Every run completed 128 requests, 1048576 input and 131072 output tokens. Both arms use unchanged audited lmhead-production including M4, BF16 weights/KV/query, TRT HND page 32/mixed 16384. External candidate changes only exact M64 vocabulary GEMM; normal public autotuning selects tactic 7 in both candidate runs. Actual capture and target graph replay are recorded. No private startup policy is used.

Both arms explicitly cap decode graphs at 64 and pin 1600000 KV tokens, checked through server_info before requests. Twelve graph buckets preserve exact 64 and smaller decode paths. Normal M128 projection down tactics differ across phases (A4control4/A5candidate4/B4candidate6/B5control1); gate/up 1 and M4 vocabulary 2 are common. A complete post-flush measured prefill audit has 68/67/66/67 records, and every conservative [new-token,new-token+running-req] interval excludes 128. TP1, complete single-rank prefill logging, no prefill graphs or distributed MLP row padding, and decode cap 64 establish no measured M128 projection use. Probe/warmup records are excluded. The raw intervals and all selected records are preserved.

Driver 545319 completed both phases; servers 545452/545453 then 558725/558726 cleaned up. GPUs 4/5 released. Remote root /root/qvl/experiments/lmhead-c64-serving; raw compact archive /tmp/lmhead-c64-serving.tar.gz retained locally/remotely. Exact source manifests, commands, telemetry, per-run absolute metrics and hook accompany this report. Matched C128 diagnostic ran independently on 6/7 during part of this screen; no shared GPU.
