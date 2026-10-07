# Retained FA4 prefill screen: scope audit

These are exact copies of the old local `/tmp/qvl_bf16_prefill.py` and `/tmp/qvl-perf/bf16-attention/prefill.log`. The harness is saved as `harness.py.txt` to preserve its original bytes through formatting hooks; Python can execute that filename. No GPU experiment was rerun for this audit.

The original source revision, package versions, GPU identifier, seed and invocation were not recorded alongside these files. Treat this as an old diagnostic comparison with unverified provenance, not a current-source performance claim.

SHA256:

- Harness: `cd1d809037abc20a72b87a057cdd188ee16508c1506a0996a65d3a72c0c3071d`
- Log: `c8894578930d72fb71e62daa1cb7589feb085218c3faf51d616fa48456b53836`

## Exact scope

BF16 causal attention, batch1/2/4,8192 tokens per request,32 query heads,8 KV heads, dimension128. Q/K/V are contiguous random BF16 tensors. FA4 calls `flash_attn_varlen_func` with unpaged contiguous K/V. TRT calls `trtllm_batch_context_with_kv_cache` with page32 K/V views backed by NHD storage and permuted to HND, a contiguous page table and512MiB workspace. TRT is tested with max-KV bounds8192 and262144; max-Q remains8192.

Five calls are captured per CUDA graph, followed by two warmup replays and ten timed replays. This is kernel-only timing: no cache writer, Q/K/V producer, compaction, serving, mixed decode tails or cached prefixes. Order is fixed TRT8192, TRT262144, FA4. The harness reports maximum output difference but does not assert an accuracy tolerance or repeat alternating timing rounds.

| Batch | FA4 us | TRT bound8192 us | TRT bound262144 us |
| --- | ---: | ---: | ---: |
| 1 | 324.935 | 378.982 | 378.410 |
| 2 | 786.439 | 731.690 | 814.449 |
| 4 | 1575.978 | 1523.956 | 1618.165 |

FA4's maximum reported absolute output difference is0.00390625 in each case. B1 improves about14.3% against actual-bound TRT; B2/B4 are7.48%/3.41% slower. Against the model-limit TRT comparison, B2/B4 improve modestly. Thus the retained experiment does not justify saying FA4 loses under every comparator. Dominant current mixed batches contain approximately two prefill requests, making the B1 advantage insufficient evidence of a high-concurrency gain.

## Defaults and evidence gap

The old harness supplies no FA4 tuning parameters. In the currently inspected source, the public signature defaults `num_splits=1`, `pack_gqa=None`; internal resolution enables GQA packing when query/KV head ratio exceeds1. The public call selects the normal architecture-specific BF16 tensor-core implementation; it is not a non-tensor-core reference. Internal `_flash_attn_fwd` configuration includes tile/overlap/thread controls, but the harness does not override them. Current source behavior cannot retroactively establish the exact old revision's defaults or generated kernel.

A separate retained page128 FA4 decode screen is a different experiment. No authoritative retained invocation or serving result was located for **FA4 prefill-only with unchanged TRT page32 decode**. Conversational references to a page128 hybrid are insufficient to rule out that configuration. The reference backend audit was corrected accordingly. No new implementation or sweep is proposed by this evidence-only audit.
