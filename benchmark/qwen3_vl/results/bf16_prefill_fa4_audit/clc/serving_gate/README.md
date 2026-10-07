# Fresh-prefix-only FA4 serving gate

Both variants use verified5f60fb8b67-equivalent source, excluding M1. Candidate alone loads the external FA4 CLC hook, SHA2564fbc394b593d5a449fdce817569484962fed757eb26b82691597c76cf037aced. TRT decode, HND, page32 and BF16 weights/KV/queries remain unchanged. Four GPU pairs use c128/N128/warm128,8192/1024, ordinary serving radix caching. All eight runs complete128 requests,1048576 input tokens and131072 output tokens.

Paired throughput geometric mean is **-0.2036%**; individual changes are-0.4380%,+0.0044%,-0.3453%,-0.0349%. Descriptive t95% interval[-0.5550%,+0.1489%]. This fresh-prefix-only candidate provides no measured serving gain and is not promoted.

Serving coverage snapshots prove actual CLC=True FA4 calls:252–324 layer calls and2.50–3.39million layer-tokens per candidate run. Cached-prefix fallback snapshots each report2049 calls. Counters are sparse snapshots, not exact total counts. The first recorded context is a78-token startup/warmup context with prefix0, not a measured prefix histogram.

The128-question c128 GSM screen disables radix caching on BOTH variants solely to improve fresh-context coverage. Control scores121/128 and candidate120/128; only row20 changes correctness. However, the candidate final coverage JSON contains zeros. Multiple server processes register writes to the same output path, so shutdown overwrite is suspected. This accuracy score has **unverified candidate coverage** and must not be used as a validated accuracy gate. No additional accuracy run was made after the negative serving result.

All raw commands, metrics, coverage snapshots, logs, source hashes and telemetry are preserved. Source patches are compressed before formatting. No production files changed. Follow-up baseline-only prefix observation is separate and does not replace this experiment.
