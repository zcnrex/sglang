# Mixed decode audit

Read-only inspection at local HEAD `07e9970bc24622db44dcd8a1cf5f4cf0db916a0f`. No kernel, server, or GPU experiment was run for this audit.

## Finding

No source evidence indicates redundant context KV reads in the mixed decode suffix. In `python/sglang/srt/layers/attention/trtllm_mha_backend.py:2024`, the suffix passes sliced query, page table, sequence lengths and output to `_run_fixed_q_len_decode`. The ordinary BF16 decode route calls the same helper at line 1825. Its direct group call at line 1471 uses the same persistent-counter TRT decode API, cache, model maximum length and attention scales. With the established one-split configuration, there is no sorting or gather. Mixed suffixes are eager slices, not padded decode graph buckets. The full cache pointer does not imply reading every request: the sliced block table and lengths define accessible rows.

The strongest memory evidence remains the standalone B128/L8192 NCU result in `../bf16_lossless_screens/ncu_decode/README.md`: 4,296,317,440 read bytes versus 4,294,967,296 mandatory K/V bytes, only 0.0314% excess, at 94.15% sustained DRAM throughput. This cannot establish B39 bandwidth or traffic.

## Existing timing evidence and its limit

`../current_c128_profile/trace-summary.json` contains a historical late mixed trace with 180 TRT decode launches averaging 648.231 us, versus 637.670 us for 180 launches in its adjacent B128 pure-decode trace. Mixed total batch sizes were 124–125, so these are large tails, not the current B39 case. Different lengths, thermal state, preceding compute and profiler effects prevent interpreting the 1.66% time difference as a bandwidth loss. Another pure-decode trace averages 594.529 us, illustrating the comparability limit. Kernel elapsed time alone does not measure excess DRAM traffic.

The retained `../bf16_native_epilogue/subtile_model/native_model_geometry.json` has B42 with three context requests and 39 one-token decode requests. Their sequence lengths sum to 319,542, ranging from 7,990 to 8,221. Mandatory BF16 K+V traffic is 319,542 × 8 KV heads × 128 dimensions × 2 bytes × 2 = 1,308,844,032 bytes per layer. Scaling the unprofiled B128 reference 586.609 us solely by bytes gives about 178.8 us. That is a bandwidth-only reference, not a prediction or measured lower-tail result. No retained current B39 hardware-counter measurement was found.

## One bounded remaining comparison

If attention work resumes, compare existing TRT decode against the public FlashInfer paged decode wrapper at exactly this B39 geometry, HND page 32, BF16 queries/KV/output, Q stride 6144, real shuffled block tables and the recorded heterogeneous lengths. Plan the alternative once outside timing; preserve identical cache, output rows and scaling. Check BF16 numerical tolerance and changed input before alternating retained CUDA-graph timings. Test both standalone cold rotating caches and the same preceding context computation to distinguish intrinsic small-batch utilization from scheduling interference. This is a benchmark gap, not evidence that the alternative is faster; no new implementation is warranted until it wins.

Prior raw Triton attention's bounded 12-configuration sweep covers B128/L8192 and loses to TRT. Page-128, cache-layout, Q-compaction, PDL, maximum-length and stream/green-context experiments already exist and should not be repeated. The lossless XQA producer experiment is not a same-input vanilla BF16 paged-decode backend comparison. No B39 same-cache public paged-decode comparison was located in the inspected evidence. A result at B39 would still need measured suffix-frequency weighting before any serving claim.

## Exact inspected hashes

- `python/sglang/srt/layers/attention/trtllm_mha_backend.py`: `4896aa04b862ef0d926b9f0fcb489fcce095aeccb20a2a579be6fa8f85d8476f`
- `benchmark/qwen3_vl/results/current_c128_profile/trace-summary.json`: `fc6592975add71a919341853e51ff8de048394f5159d67f480dd1f7d445ba9fa`
- `benchmark/qwen3_vl/results/bf16_native_epilogue/subtile_model/native_model_geometry.json`: `b0db26203f0c4b3e1e590ae1a62e5a468d760e9e7827b50dda170da0f5cc0eed`
