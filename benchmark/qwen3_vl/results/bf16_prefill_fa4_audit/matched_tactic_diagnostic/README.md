# Matched M128 tactic diagnostic

Full1314-question GSM8K c128, ordinary radix, five shots, greedy/max2048, unchanged production source and BF16 precision: control1217, candidate1216. Completion tokens240207/241776. Ten control-only rows and nine candidate-only rows remain; this is numerical variation, not equivalence. No further full accuracy repeat was launched. Control GPU7, candidate GPU6 reverse the original accuracy assignment.

Both startup and terminal JSON checks prove identical gateup1/down1 tactics. Both READY sets contain the two optimized shapes; both actual optimized dispatches were observed during CUDA graph capture. Exact candidate completed packing count is36 (one36-layer context); control has no packing calls. Coverage is limited and does not validate many cached-prefix cases. Prior model and standalone tests provide separate cached-prefix correctness evidence; this score must not be described as exhaustive packing coverage.

This is explicitly a PRIVATE-STATE STARTUP-POLICY DIAGNOSTIC, not normal autotuning or a production fix. The external wrapper intercepts AutoTuner.search_cache only for bf16_gemm exactM128 shapes gateup/down, holds its lock, temporarily sets is_tuning_mode=False for the lookup, and restores it in finally. It asserts loaded-cache hit/tactic1. After observing both captured production dispatches, it restores the original search method. Serving math/operators remain unchanged. The pack observer wraps the original function without changing arguments/results and writes an exact completed-call counter perPID plus sparse samples. Source files were not modified;4229-file manifests for each tree rechecked before launch. Exact wrapper and SHA are retained.

The preceding public autotune(False) startup-only attempt failed safely before evaluation: FlashInfer reference-counted tuning mode remains enabled inside an enclosing True context. That failed attempt and logs are retained in fa4-policy-startup. The subsequent private-state startup-only proof passed, then the full pair was authorized and launched. This does not bypass any production accuracy assertion; each full worker still required identical saved tactics, READY and captured dispatch before evaluation, and identical tactics again afterward.

Drivers390511/390512, children390513/390514 terminal; GPUs6/7 empty after cleanup. Exact sparse records/counters, seed hash/configs, startup/terminal configs, full per-question reports, commands/logs, and standalone proof are preserved. Compiled caches omitted; HTML compressed without altering bytes. Current model source paths: control /root/qvl/sglang-m1-production, candidate /root/qvl/sglang-prefix-production.

Control-only original zero-based dataset rows:20,87,326,562,663,724,955,1006,1088,1127. Candidate-only:159,241,323,422,675,858,901,1146,1199.

`original-artifacts.tar.gz` preserves this archive before further repository formatting. Hashes refer to the extracted originals.
