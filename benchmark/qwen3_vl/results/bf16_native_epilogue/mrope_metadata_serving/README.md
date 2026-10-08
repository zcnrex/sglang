# Text-only device MRoPE: final bounded serving screen

This experiment isolates integer position-metadata construction; no MLP fusion or production source edit is present. Both variants use current audited packed-prefix production, BF16 weights/query/KV, TRT HND page32, mixed chunk16384, and an explicitly equal 1,400,000-token KV capacity. Each GPU runs both variants in opposite phase order, concurrency 128, 128 measured requests, 128 warmups, 8192 input/1024 output, flush after warmup.

The candidate repeats the already-computed CUDA int64 positions into three contiguous rows. It accepts only EXTEND/MIXED, no speculative or dLLM metadata, no custom extend_position_info, no deterministic override, every multimodal entry None, and matching contiguous position shape/host lengths. Any multimodal object or nonzero delta retains the original branch. Guards were tightened relative to the short model; the exact semantic limits are preserved in the hook.

Initial `ineligible_v1` used EXTEND equality and accidentally excluded real MIXED serving batches. Its counters caught only one fast call per measured run; these timings are explicitly ineligible for the intended comparison. That run completed unchanged and all GPUs/ports were free before v2. The corrected hook permits EXTEND or MIXED. Before v2, a real ForwardBatch(MIXED) using production compute_position(trtllm_mha), exact B42/M16331 prefixes and 39 trailing decode rows matched original MRoPE integers bitwise. A deliberately shifted custom-position signal correctly fell back.

Both variants use fresh isolated tuner caches seeded identically and the previously verified scoped startup-only private tuner policy, not normal independent autotuning. Both optimized Lt shapes are observed during CUDA-graph capture; startup policy is restored before requests; startup and terminal configurations must match the seed. Source manifests and effective KV size are asserted. No algorithm state changes during measurement.

Coverage is aggregated in memory, written at cache flush and terminal cleanup, with one first-MIXED early signal per epoch. No per-call logging or GPU scalar reads. Epoch 0 includes readiness/probe/warmup; epoch 1 is measured post-flush. V2 asserts every measured candidate metadata call used the fast path and more than one MIXED call occurred. The early snapshot incurs a single shared instrumentation write per epoch; these remain diagnostic serving runs.

Raw pre-format archives are retained as original-v1.tar.gz and original-v2.tar.gz; exact commands, script hashes, four-run summaries, coverage, cache configs, source manifests, telemetry and cleanup evidence accompany final analysis. The v2 results are recorded separately and must not be pooled with v1.

The corrected result does not support promotion: same-GPU throughput changes were −0.185412% and +0.156663%, mean −0.014374% (two pairs, not a precise zero-effect estimate). Median TTFT changed +0.335511%/−0.204226%, TPOT +0.074668%/−0.189151%. All four runs completed exactly 128 requests, 1,048,576 input tokens and 131,072 output tokens. Full absolute metrics are in analysis.json.

Measured coverage was 65/65 and 66/66 candidate calls fast, zero measured fallback, including 64 and 65 MIXED calls. Each candidate warmup executed five fast calls. The only warmup fallback was the readiness path. Thus this neutral/mixed result is not explained by missing candidate dispatch.

Corrected driver 433717 and servers 433722/433723 (phase A), 438486/438485 (phase B) completed. Terminal nvidia-smi reported no active compute processes. Raw archive /tmp/mrope-serving-v2-results.tar.gz is retained locally and remotely. No full GSM or further serving repeat is warranted from this bounded gate; production remains unchanged.
