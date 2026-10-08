# BF16 gate/up producer epilogue feasibility

Read-only audit on 2026-10-07. No GPU experiments or production edits.

For Qwen gate/up N=19456, K=2560, the actual optimized producer is not the open TGV kernel. With the validated tuned route ready, M128 uses FlashInfer cuBLASLt; M8192 falls through to `F.linear`/cuBLAS. These closed library mainloops do not expose an accumulator callback through the inspected interface.

## Selector evidence

`python/sglang/srt/layers/quantization/unquant.py:426–464` implements the tuned cuBLASLt route; `:468–505` orders dispatch before the `F.linear` fallback. An isolated AST extraction of the unchanged `use_cutedsl_bf16_gemm` function (`python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py:1357`) returned `False` for both `(128,19456,2560)` and `(8192,19456,2560)`. Its K<4096 branch requires K>=3072. This selector check needs no CUDA execution; the small-M shape set cannot affect these inputs.

Installed FlashInfer `gemm/gemm_base.py:665–735` exposes bias but no paired activation callback through `mm_bf16`. Its `CuteDSLCublasltFallbackBf16Runner` at line1826 delegates M>32 to cuBLASLt. Choosing the CuTe backend therefore does not expose a large-M accumulator consumer.

## Open replacement interface and mapping

Installed `gemm/kernels/tgv_gemm_cute_ext.py`, class `TgvGemmCuteExtKernel` (line176), loads TMEM into FP32 registers at line973, converts to BF16 at989, then stores to global memory. This is a possible source-level fusion point retaining the **TGV** mainloop, not the winning production cuBLAS/cuBLASLt mainloop. There is no generic epilogue callback in the inspected wrapper.

TGV swaps operands (`_to_cute_swap`, line1215): internal M represents output channels and internal N represents token rows. The corresponding maintained SGLang implementation documents the CTA64x8 mapping at lines850–859: thread0 gets `(0,0),(0,1),(8,0),(8,1)`. Gate/up halves separated by9728 channels are not adjacent paired registers. Fusion requires a weight-row permutation and matching output addressing, or additional communication. Preserve the intermediate BF16 rounding before SiLU and multiplication; directly applying activation to FP32 accumulators changes semantics. This differs from a conventional token-M/channel-N persistent GEMM epilogue.

## Source identity

Local paths are relative to the repository; installed paths are relative to `/root/qvl/venv-sgl/lib/python3.12/site-packages/flashinfer` on `chunan-b300-8`. SHA256:

| Source | SHA256 |
| --- | --- |
| `python/sglang/srt/layers/quantization/unquant.py` | `0fa01b10b34bec46f4cca14b6dfb96fa67a07d94e68c1d08035b8c851740f431` |
| `python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py` | `65c7abdff98005040170c77ab67677551fcfef211ca487b1660af91f9733b34e` |
| installed `gemm/gemm_base.py` | `6ab2f4f8496b8fff54319a299565dbb41858012edf54f86c721578504a34108c` |
| installed `gemm/kernels/tgv_gemm_cute_ext.py` | `badae5c2638ac423dc5e0aac9fec29cfefe3cf6ee980b323dd3a9e541a3576cc` |

This audit does not predict the performance of the separately tested native persistent epilogue prototype. It only establishes that attaching its paired epilogue to the actual production producer is not available through the inspected open interfaces.
