# Read-only runtime audit

Audit source: local HEAD `60971e79d1186942df7ce7964fc50fe9c6c1b28e`, with shared working-tree changes potentially present. The sampler and NaN-sanitization branches were also read directly from `/root/qvl/sglang-perf/python/sglang/srt/` on the experiment host; this was a targeted source comparison, not a full source-hash equivalence check. No code changes or new GPU measurements were performed for this audit.

## Findings

Paths below are repository-relative; line numbers refer to the inspected local source.

- `python/sglang/srt/models/qwen3.py:469–490`: the decoder layer delegates to the residual/attention boundary. No repeated residual clone was found in this inspected forward path. This is a source observation, not a complete allocation trace.
- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:1396–1421`: ordinary decode pads batch size to a captured bucket. Batch 128 has an exact bucket in the tested configuration, so a full c128 step does not duplicate requests through batch padding. Smaller tail batches can still pad; their aggregate cost was not measured here.
- `python/sglang/srt/layers/logits_processor.py:897–914`: explicit pre/post LM-head synchronization is gated by `SGLANG_TRACE_LOGITS_E2E_SYNC`, default false at `python/sglang/srt/environ.py:477`.
- `python/sglang/srt/utils/async_probe.py:66–76`: NaN sanitization returns early when its flag is off. `SGLANG_SANITIZE_NAN_LOGITS` defaults false at `python/sglang/srt/environ.py:1352`; this is not an unconditional full-logits reduction in the normal path.
- `python/sglang/srt/layers/logits_processor.py:970–1033,1154–1182`: default same-dtype LM-head output is subsequently copied/cast to FP32 logits. `python/sglang/srt/layers/sampler.py:186–194` then uses argmax for greedy sampling.

## Untested minor opportunity

For a tightly guarded greedy path with no requested logprobs, custom processors, scaling/softcap transformations or special nonfinite handling, argmax could consume the BF16 LM-head result directly. Exact BF16-to-FP32 conversion preserves finite values and their ordering, but tie-breaking and exceptional-value behavior would still require tests. This is a proposed source-path optimization, not a tested implementation or recommendation to alter precision.

At batch 128 and vocabulary 151936, the FP32 buffer contains 19,447,808 values (77,791,232 bytes). Avoiding its write and subsequent read removes approximately 155.6 MB of traffic, before accounting for cache reuse and replacement BF16 reads. At an assumed 5–7 TB/s effective bandwidth, that traffic corresponds to roughly 22–31 microseconds. Compared with an approximately 25 ms decode step, this is about 0.09–0.12%; an informal allowance for launch overhead still suggests less than roughly 0.2%.

This is an optimistic scale estimate, **not a measured speedup or a rigorous upper bound**: cache residency, kernel utilization and graph boundaries affect the actual result. The supplied profile attributed 87.9% of pure-decode GPU time to attention. Nothing in this audit establishes enough remaining runtime headroom to close the current throughput gap, so no implementation or new serving benchmark was started.
