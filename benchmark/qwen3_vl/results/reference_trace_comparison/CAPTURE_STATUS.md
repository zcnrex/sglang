# Completed captures — 2026-10-08

All requested batch8, batch16 and batch128 captures completed for both frameworks. Each decode capture contains five correlation-verified full-batch CUDA-graph forwards at mean actual context≥8704, following warmup. Separate warmed prefill traces retain their actual chunk geometry; SGLang mixed-chunk traces also contain explicitly accounted provisional decode tails.

Only GPUs4–7 on `chunan-b300-8` were used. vLLM used4/5 and SGLang6/7; the unrelated TP4 workload on0–3 was left untouched. Owned profiling servers shut down after capture. See [report.md](report.md) for measurements, scope differences and interpretation.

## Provenance

- SGLang exact source: `c76cfaba1a1154a4a24b8694184cb8d3318fc358`.
- vLLM official wheel: `0.30.1rc1.dev648+g92044241a`, SHA256 `357e6af0aa6a80e4cae8f5a825795bf9d873f599d3e3720b4ebeb23d5d83710b`, installed separately in `/root/qvl/venv-vllm-profile`. Torch2.13.0+cu130, FlashInfer0.7.0.post1, Transformers5.17.0. This is a reconstructed environment; the original full dependency lock was unavailable. The resolved lock and wheel provenance are archived under `vllm_preparation`.
- Both frameworks use BF16 weights/query/KV, the same model snapshot and shared native8192-token inputs. These controlled synthetic inputs are not the historical retokenized chat benchmark. Traced durations are not unprofiled serving throughput or TTFT measurements.
- Exact shared fixture is archived in three parts under `shared/`; concatenate in filename order as described in `fixture_archive.json`. Both compressed and uncompressed hashes were verified against `fixture_metadata.json`.
- Actual decode graph modes were verified per step. Startup PIECEWISE graph messages do not imply PIECEWISE decode.
- LBHNC and HND labels alone do not prove different per-layer axis order. Kernel page sizes differ; see the report for separately supported layout/packing observations.

## Recovered setup and client failures

Preserved failures are diagnostic history, not performance results:

1. Initial SGLang startup lacked the pinned Rust1.92 host target component. Installing that component repaired startup without a source/backend change.
2. The SGLang client initially expected JSON from a successful plain-text cache-flush response. Only client decoding was repaired; the same servers were retained.
3. SGLang mixed-chunk max-output1 prefill includes provisional decode tails. Prompt rows and tail rows were audited separately, then decode capture resumed on the same servers.
4. The first vLLM observer omitted vocabulary projection because the default v2 runner computes it in its later sampling call. Those startup attempts were stopped before capture. Corrected observers retain separate decoder and logits scopes and stop after five complete sampling calls.
5. The first vLLM sanity client lacked the SGLang evaluator module in the isolated environment. It was rerun using the existing evaluator environment; runtime servers were unchanged. The evaluator's source_revision field is not the vLLM model-runtime revision.
6. An expired rx token interrupted monitoring. The user restored authentication; existing server/client handles were inspected and resumed without duplicate launches.

Historical handoff state is preserved as such. Current results and mappings supersede it. No production code or upstream PR contents changed during this profiling task.

## Repository packaging

Repository hooks require normalized text whitespace. Where an imported text artifact needed this normalization, its exact original bytes are preserved beside it as `.original.gz`; `raw_text_normalization.json` records both hashes. Trace gzip files remain byte-for-byte unchanged. See `TRACE_ARCHIVES.md` for lossless reconstruction of the two larger prefill traces from their parts.
