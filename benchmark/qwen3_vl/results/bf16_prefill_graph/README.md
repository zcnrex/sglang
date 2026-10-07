# Text-only prefill graph audit

The Qwen3-VL handoff explicitly noted that its multimodal architecture disabled prefill graphs. Recent archived benchmarks also explicitly disabled them. No earlier graph-enabled BF16 serving result was located in the retained evidence; this does not prove that no old-host prototype was run.

The current multimodal breakable/piecewise allowlists omit Qwen3VL. This external prototype appends only that architecture to the breakable allowlist and rejects batches containing multimodal inputs in `can_run_graph`, preserving their eager path. It is not a production change. Adding this allowlist entry in production would alter default graph resolution and requires an explicit behavior decision.

The existing graph runner supports MRoPE buffers and maps MIXED to EXTEND. The optimized TRT attention route recognizes one-token suffixes in EXTEND metadata. Therefore text-only replay is a concrete compatibility hypothesis, rather than an inference from another model.

## Existing trace bound

`trace_gap_audit.json` uses the archived late-mixed profiler trace. Its 2,702 kernels sum to 705.672 ms, with interval union 702.237 ms over a 704.984 ms first-to-last kernel span. Uncovered intervals total 2.747 ms (0.390%). These are profiler-sample figures, exclude forward-boundary overhead, and do not identify the cause of every gap. Whole-forward CUDA event time versus wall time alone cannot rule out host submission gaps within forward. The kernel union provides a much narrower bound for this particular late-mixed sample.

## Correctness screens

The authoritative remote `/root/qvl/sglang-public-lt-clean/python` was compared against all 4,229 committed Python files at `5f60fb8b67`: no mismatches or extras. An initial old-source eligibility smoke was discarded from validation; its timings are not included here.

- B1 input8192/output2: graph and eager logits were bitwise equal, greedy outputs identical, with an exact8192 capture bucket. Eight alternating measurements after common warmups suggest a small B1 improvement, but are preliminary and include an eager outlier.
- Heterogeneous input lengths8200/8007 plus124 fresh single-token requests: total16331, bucket16384,53 padded rows (0.325%). Last-token logits and greedy outputs were bitwise equal; instrumentation recorded four actual graph executions. Two measured graph/eager pairs show no gain. These lengths are a synthetic representative geometry, not recovered lengths from the production trace. Fresh suffix requests do not validate long cached decode tails.

`check.py` captures correctness only during identical-input warmup, outside measured iterations. The heterogeneous report retains the benchmark's nominal input-length throughput fields, which are invalid for the modified request lengths; use latency only.

The external serving harness adds an actual image request to test eager fallback, captures replay eligibility/shape counts, and counterbalances graph/control on each GPU. Both variants use the same audited source. The initial graph buckets8192/8704/16384 follow the existing512-token step near8k;8704 is not a demonstrated hard alignment constraint. Padding distribution must accompany interpretation of the serving results.

## Serving result: reject for high concurrency

Two-GPU crossover, N256/c128, nominal8192 input and1024 output tokens,128 warmup requests:

| GPU | Graph output tokens/s | Eager output tokens/s | Change |
| --- | ---: | ---: | ---: |
| 3 | 3820.099 | 3823.109 | -0.079% |
| 4 | 3814.713 | 3821.664 | -0.182% |

Geometric change is -0.130%. All four runs completed with exactly2,097,152 input and262,144 output tokens. Recorded phaseA padding adds0.841% token rows, dominated by batches near16330; narrower8k buckets affect few recorded batches. No full sweep or bucket tuning is justified by this result.

The actual image API request succeeded and the candidate hook counted multimodal eager fallbacks. Greedy text agrees for all128 paired probes on each GPU. Output logprobs are bitwise on GPU3; GPU4 differs by up to0.0426643. The probe's4096-token prefill pads to8192 in the graph variant, unlike the exact8192 and near16384 standalone checks; this experiment does not establish broad logprob equivalence. Separate per-process autotuning is another arithmetic variable. Raw probes are retained without attributing the difference to a proven cause.

All owned serving processes exited. Compiled-cache paths were mistakenly fresh for this crossover, adding startup compilation only; future experiments reuse warmed compiled caches and isolate tuning JSON alone. No production change or default enablement is proposed.
