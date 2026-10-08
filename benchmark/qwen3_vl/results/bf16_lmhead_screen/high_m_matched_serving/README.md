# Matched-startup C128 vocabulary diagnostic

The isolated short serving screen is negative; no M128 promotion, full 640 or GSM run is justified by it. Two pairs do not establish a precise general regression.

| GPU | Control / candidate tok/s | Throughput change | Median TTFT change | Median TPOT change |
|---|---|---|---|---|
|6|3795.296854 / 3763.662474|−0.83352%|+6.35662%|−0.06872%|
|7|3805.729976 / 3804.780978|−0.02494%|−0.28216%|−0.00008%|

Both arms use identical frozen lmhead-production, BF16 weights/query/KV, HND page 32/mixed 16384, C128/N128/warm128, 8192 input/1024 output and benchmark flush. Every run completed 128 requests, 1048576 input and 131072 output tokens. Exact commands, manifests, metrics, seed hashes and selected public cache records accompany results.

This is an external startup-policy diagnostic, not normal automatic selection or a production patch. Separate caches are seeded from the same recorded control. Scoped search-cache wrapper temporarily disables tuning mode under the tuner lock only for existing projection shapes and M4 vocabulary. It verifies cache hit, returned runner index, expected tactic and current registry membership. It restores tuner mode in finally and restores original search function after init_cuda_graphs before requests. All four runs prove projection gate/up 1/down 4 and M4 vocabulary 2, each in valid tactics 0–7. Candidate M128 vocabulary remains normal public tuning and independently selects 6 both runs. Actual optimized target capture/replay and graph128 replay prove active paths, not fallback.

Both arms cap decode graph size 128 (20 smaller/exact buckets) and explicitly pin 1600000 KV tokens, asserted through server_info after startup. These differ from the preceding normal screen's default 52 buckets and approximately 1607904 tokens, so only within-diagnostic pairs are interpreted. Production sources are unchanged. All four capacity assertions and startup-policy checks passed before measurement.

Driver 531917 completed; phase A servers 531922/531923, phase B servers 537140/537139 cleaned up. GPUs 6–7 released. C64 standalone serving continued independently on 4–5; other jobs were untouched. Remote root /root/qvl/experiments/lmhead-highm-matched; original compact archive /tmp/lmhead-highm-matched.tar.gz retained locally/remotely.
