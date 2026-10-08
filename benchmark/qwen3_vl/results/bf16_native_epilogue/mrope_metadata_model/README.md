# Device MRoPE metadata: bounded external model gate

The text-only metadata replacement passed exact integer and model-output checks, but this short model gate did not establish a useful forward gain. No production change was made. A separate bounded serving screen was subsequently selected to resolve whether the CPU saving affects the request loop.

The external hook reuses the existing one-dimensional CUDA int64 positions and repeats them into three contiguous rows, replacing per-request CPU arange/repeat/cat and HtoD construction. It requires ordinary EXTEND, no speculative metadata, every multimodal entry None, and matching row count/layout. Every multimodal object, including nonzero MRoPE delta, takes the original branch. This is a narrowly scoped diagnostic; arbitrary custom-position modes were not validated for production dispatch.

Six standalone integer cases passed: fresh 8192, exact B42/M16331 prefixes/chunks/39 decode tails, mixed zero-length rows, all-zero length, one-token rows, and high prefix offset. Explicit nonzero multimodal-delta fallback passed. Eight alternating metadata-only timing blocks (100 calls each, CUDA synchronized at boundaries) measured median 803.0925 us original versus 24.2464 us candidate. CPU profiler instrumentation was disabled. This isolated loop measures repeated metadata construction, not whole-model latency.

Both actual model runs used the same audited 4229-file packed-prefix source and exact recorded B42/M16331 geometry, seed 42 synthetic token inputs. The prior harness rebuilt identical per-variant cache state, warmed both variants, then measured eight forwards per GPU in ABBA/BAAB order. Each candidate invoked the fast path once, retained 36 actual FA4 calls, and performed no compilation/weight transforms. Prefill and three subsequent decode logits were bitwise equal on both GPUs; this is not a full accuracy-equivalence claim.

| GPU | Control mean ms | Candidate mean ms | Latency reduction |
|---|---:|---:|---:|
|4|137.827298|137.670936|0.113447%|
|5|134.885080|134.820677|0.047747%|

Only 4/8 adjacent pairs improved. The profiler's 2.19–2.45 ms metadata gap and isolated 803 us loop must not be treated as end-to-end savings: profiler overhead, real execution overlap and timing variation remain possible contributors. No contemporaneous clock telemetry was collected for this bounded gate, so the small differences cannot be attributed reliably.

Scripts, commands, source audit count, exact geometry, counters, checks and raw logs are preserved. The raw pre-format archive is retained as `original-artifacts.tar.gz`, with a remote copy at `/tmp/mrope-metadata-model-results.tar.gz`. Workers 424198/424199 completed; terminal nvidia-smi reported no compute processes. No further GPU jobs were part of this model gate.

Timing scope is inclusive: the script synchronizes, starts perf_counter, calls runner.extend(reqs), synchronizes, then stops the timer. `_TorchBenchRunner.extend` delegates to `one_batch.extend`, which performs ScheduleBatch preparation, ForwardBatch.init_new (including MRoPE preparation), ModelRunner.forward and sampling. The fast-call counter is read directly before and after that timed call. Metadata was not excluded from the timer. Prior cache construction is outside the interval and identical between variants.
