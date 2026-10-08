# Pinned reference audit: no supported new low-batch lead

This read-only audit inspected saved vLLM source requested at `92044241a`,
the reference version recorded in the handoff, for a BF16 C8/C16 optimization
absent from the promoted SGLang M16 configuration. It did not launch GPU work,
change production files, or remeasure the reference.

## Provenance

`source_manifest.json` hashes the exact saved files inspected in
`/tmp/qvl-vllm-source`. `provenance.txt` records the local revision-metadata
query and file hashes. The earlier report records full revision
`92044241a02f05de51420d654daf669de20b4691`; this audit did **not** independently
resolve that full SHA. The earlier retrieval audit explicitly pins the
abbreviated revision and hashes four raw files:
[reference attention audit](../bf16_decode_lt/production_validation/reference_attention_audit.md).
There is no retained local Git checkout whose `git rev-parse HEAD` could
independently establish provenance. Hashes identify inspected content, not
proof of the historical server's executed kernels.

## Exact source findings

| Source at the pinned revision | Finding and consequence |
| --- | --- |
| [cuda.py:155–165](https://github.com/vllm-project/vllm/blob/92044241a/vllm/platforms/cuda.py#L155), [flashinfer.py:2614–2641](https://github.com/vllm-project/vllm/blob/92044241a/vllm/v1/attention/backends/flashinfer.py#L2614) | Causal SM10x selects FlashInfer and dispatches `trtllm_batch_decode_with_kv_cache`. Current SGLang already uses this decode API; this is not evidence of an additional faster reference kernel. |
| [flashinfer.py:1850–1857](https://github.com/vllm-project/vllm/blob/92044241a/vllm/v1/attention/backends/flashinfer.py#L1850) | `use_cascade_attention` unconditionally returns `False`. Cascade reuse is therefore not a supported explanation for the reference's advantage. |
| [layers/utils.py:616–645](https://github.com/vllm-project/vllm/blob/92044241a/vllm/model_executor/layers/utils.py#L616) | Automatic unquantized CUDA dispatch uses the default GEMM; FlashInfer BF16 alternatives require an explicit backend selection. Saved baseline configuration does not establish such a selection. Existing GEMM screens cover the relevant alternatives. |
| [qwen3.py:155–174](https://github.com/vllm-project/vllm/blob/92044241a/vllm/model_executor/models/qwen3.py#L155) | Source applies Q/K normalization followed by rotary and attention. SGLang already has fused normalization/rotary/cache preparation. Python source alone cannot exclude additional compiler fusion in the reference. |
| [flashinfer.py:1340–1361](https://github.com/vllm-project/vllm/blob/92044241a/vllm/v1/attention/backends/flashinfer.py#L1340) | Mixed decode/prefill slicing and head-major cache handling overlap existing SGLang routing/layout work. Page size, bounds, PDL, attention alternatives, chunk scheduling and GEMM/cast paths have already been screened; this audit does not justify repeating them. |

No source is copied wholesale; the links, identifiers and hashes preserve
the inspected locations. The source-level selections above are inferences
from the saved configuration, not newly observed reference execution.

## Decision and limits

No recovered C8/C16 reference kernel trace establishes a new hotspot cost or
an untested mechanism with a defensible whole-run upside. Current SGLang
profiles cannot establish which historical vLLM operation explains the gap.
Consequently there is no supported falsifiable standalone candidate to launch
from this audit. This does not prove that all possible optimizations are
exhausted, or that reference and candidate execution are identical. In
particular, compiled fusion, CPU scheduling and historical cache-hit behavior
remain unmeasured rather than ruled out.

The performance goal remains unmet; no improvement or accuracy-equivalence
claim follows from this report.
