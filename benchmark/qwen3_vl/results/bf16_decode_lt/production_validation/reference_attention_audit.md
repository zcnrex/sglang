# Exact-version reference attention audit

Retrieved from official vllm-project/vllm raw GitHub URLs at revision
`92044241a` on2026-10-07. Revision is the supplied abbreviated commit ID;
no independently resolved full SHA is claimed. Full sources remain outside
the repository at `/tmp/qvl-vllm-source/`.

For the handoff's default causal BF16 TP1 text-only configuration on SM103,
source selection prioritizes FlashInfer; automatic BF16 KV prefill selects TRT
context, and SM100-family decode selects TRTLLM-GEN rather than XQA. This is
a source-derived selection, not a newly observed baseline server startup.
Qwen3-VL uses its language-model path; the benchmark has no vision inputs.

- [CUDA backend priority](https://github.com/vllm-project/vllm/blob/92044241a/vllm/platforms/cuda.py#L158)
- [Automatic TRT prefill selection](https://github.com/vllm-project/vllm/blob/92044241a/vllm/utils/flashinfer.py#L739)
- [SM100-family decode selection](https://github.com/vllm-project/vllm/blob/92044241a/vllm/v1/attention/backends/flashinfer.py#L1087)
- [TRT context call](https://github.com/vllm-project/vllm/blob/92044241a/vllm/v1/attention/backends/flashinfer.py#L2397)
- [TRT decode call](https://github.com/vllm-project/vllm/blob/92044241a/vllm/v1/attention/backends/flashinfer.py#L2614)

Mixed prefill and decode are sliced into separate calls, matching the candidate's
already implemented split routing. The reference passes actual metadata
max-sequence length to context; the candidate passes the model limit. That
specific bound was already isolated experimentally and showed no material
benefit. Page/layout and Q-compaction alternatives were also previously
screened. A retained standalone FA4 direct-varlen comparison covers contiguous
unpaged BF16 prefill only; it does not establish end-to-end performance of
FA4 prefill with unchanged TRT page32 decode. No authoritative hybrid-serving
command or result was located. See `../../bf16_prefill_fa4_audit/README.md` for
the exact old harness, results and provenance limitations. This source audit
alone does not rule out that hybrid configuration. No kernel, server or
production change resulted.

## Retrieved content hashes

SHA256 hashes identify the exact raw files inspected:

- `vllm/platforms/cuda.py`: `27bd20e6fe42dd70b72fe58ce12cb615a2385776f0ac705349646c67ca68bb26`
- `vllm/v1/attention/backends/flashinfer.py`: `251a8d365d5c327d60ed0339f32bd7c8b8879c2e72a5fa48d64a146b9b3cdbf1`
- `vllm/model_executor/models/qwen3_vl.py`: `56dfe9ccab48b81d86874dfbe16ea61c67354c011603022f4cec82474a2c3fb9`
- `vllm/utils/flashinfer.py`: `b360f8211c3b22bc7bc9a5c58b9e8b7a549bc44ef6a8c5ba7e97092013490f16`
