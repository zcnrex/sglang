# Offline trace-comparison contract (awaiting captures)

No measurements are populated yet. Keep decode and prefill reports separate. Compare actual B8/B16/B128, five pure-decode graph-on forwards with observed per-request contexts near8704. Record all five context vectors; do not infer batch/context from launch shape or requested concurrency. Prefill requires its own exact prompt lengths, cached-prefix lengths, and scheduling mode. Fixed-state replays and growing-context forwards are different protocols and must be labeled.

Capture manifest JSON has `runs` entries containing framework, immutable source_revision, versions (Torch/CUDA/driver/FlashInfer/Triton as applicable), trace path, trace_sha256 (compressed bytes if gzip), device (trace args.device), stage, batch_size, graph_on, input_contract_evidence (saved observer path/hash), windows, and decode sequence_lengths (five arrays of B lengths). Each window has start_us/end_us and a step identifier. Use GPU execution boundaries established by correlation/flow/observer evidence; CPU forward scopes alone need not enclose asynchronous kernels. Never select arbitrary first-five kernels/CPU ranges. Include graph capture/replay proof, model/precision/KV layout/backend and relevant server arguments, warmup protocol, rank/device, and active unrelated GPU processes in metadata. Cross-framework version/config differences are report inputs, not hidden confounders.

`python measure_trace.py manifest.json --out metrics.json` reads JSON/gzip Chrome traces, validates hashes, selects exact `cat=kernel`, `ph=X` GPU events on one explicitly chosen device, rejects boundaries cutting a kernel, and measures each nonoverlapping step. Copies/memsets/CPU operations are excluded from kernel metrics and require separate accounting if material. Chrome timestamps/durations are microseconds; displayTimeUnit does not rescale them. If a capture uses different category/schema, inspect and explicitly adapt the parser, never silently treat CPU ops as kernels.

For each step report kernel count, sum of kernel durations, interval union over streams, first-kernel-start to last-kernel-end span, gaps within that span, and sum-minus-union observed concurrency. Span is not a dependency-DAG critical path; it excludes host/request boundaries. Union cannot be obtained by adding per-kernel-name unions. Neither observed overlap nor summed kernel time implies removable latency. Preserve per-step metrics and raw launch shapes/streams; aggregate only after proving comparable inputs.

Run existing skill triage unchanged on each trace:

```bash
python .agents/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py --framework FRAMEWORK --input TRACE > triage.txt
```

Archive its kernel, overlap-opportunity, and fusion-pattern tables, command, script hash, and referenced catalog revisions. Default1% cutoff hides small kernels; corrected metrics retain every kernel. Read `references/fuse-overlap-catalog.md`, `overlap-catalog.md`, `vllm-torch-compile-fusions.md`, and `heuristics.md` before interpreting suggestions. Catalog entries are leads, not proof the pinned runtime contains/enables them. Do not mutate raw triage to conceal classification errors.

Corrected classification uses exact-name `kernel_annotations` entries, each with `classification`, `dtype`, immutable `source_revision`, `source` path/line, and `reason` linking pinned source/CPU correlation/shape evidence. Unannotated names remain unverified. Do not classify nvjet as FP8 from its name: verify BF16 input/output and dispatch. Do not classify fmha attention as RoPE because names share family/context. Distinguish QKV+QKnorm+MRoPE+cache-write fusion from attention computation and generic cast kernels. Graph-on traces may lack CPU mappings; use source/capture proof or a separately labeled mapping trace, without substituting its timings.

## Report skeleton

- Capture/protocol table: framework, commit, versions, B, contexts, graph proof, precision, attention backend, KV layout, trace hash, stage.
- Per-step timing table: kernel count/sum/union/span/gaps/concurrent excess; five rows per decode capture.
- Raw triage: unchanged three tables, separately for decode and prefill.
- Corrected kernel table: exact name, count, summedus, source-verified operation/dtype/launch shape; explain discrepancies from triage.
- Overlap table: observed stream intersections, dependency evidence, resource contention, whether opportunity is established or unknown. No speculative speedup arithmetic.
- Fusion table: producer-consumer contract, pinned source entry points, current eligibility, observed kernel presence, existing catalog pattern, evidence limit.
- Framework comparison: matched measured inputs first, separate CPU/capture/launch differences, then supported bottleneck conclusions. State profiler perturbation and five-step sampling limits.

No recommendation or performance claim until traces and source-backed annotations are supplied.

## Correlation-derived model windows

`python extract_step_windows.py TRACE --device 0 --out windows.json` requires exactly five complete record_function scopes named `QVL_decode_step0_B8` through step4 (suffix permitted). It joins same-pid/tid enclosed CUDA runtime/driver launch events to device kernels via args.correlation, rejects missing/reused IDs, requires cudaGraphLaunch plus graph-id kernel evidence for every step, and preserves exact event-index membership. Copy its windows into the metrics manifest; measure_trace.py honors membership rather than including unrelated kernels that happen to overlap the bounds. Ambiguous runtime/driver duplicate correlation IDs are rejected for explicit inspection rather than silently double-counted.

Unassigned GPU kernels are retained with timing/name/args and overlaps with model intervals. Keep these separate: a five-model-forward capture may include sampling between the first four forwards but omit final sampling. CUDA event elapsed times are corroborating model-stream scopes, not absolute trace timestamps; no conversion from hostmonotonic to GPU timestamps is assumed. Multi-thread launch dispatch not nested on the scope thread needs separately evidenced association and is intentionally rejected by this helper.

Offline parser verification used a synthetic five-scope fixture populated with the existing B4 trace's real cudaGraphLaunch correlation18/graph209 schema and293 kernel events per launch. It establishes join handling only, not new performance measurements. Real capture scopes and input contracts remain required.

## v2 decoder and vocabulary projection scopes

`python extract_step_windows.py TRACE --device 0 --layout vllm-v2 --out windows.json` requires five outer `QVL_decode_stepN_BB_TB` and five `QVL_decode_sample_stepN_BB` scopes. Each execute scope must contain exactly one `QVL_decoder_model_stepN_BB`; each sample scope must contain exactly one `QVL_compute_logits_stepN_BB`. Primary windows select the exact decoder-plus-projection kernel membership. Separate components retain worker bookkeeping and sample work outside projection. Before/after-projection classification is temporal until source proves operation identity. The combined first-to-last span includes intervening excluded work and gaps; report it alongside decoder and projection spans, never as pure model latency. Capture mode comes from actual dispatch metadata, not startup capture messages.
