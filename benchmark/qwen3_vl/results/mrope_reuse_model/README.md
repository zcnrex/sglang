# Shared normalization-kernel model gate

The final fixed-prefix comparison passes all 18 saved logit checkpoints bitwise: two seeds at batches 1/16/128, each with prefill/decode0/decode14. Prompt and teacher-forced token hashes match across variants. This is bounded model-computation evidence, not a full accuracy evaluation or serving TPS measurement.

Context is 8192 tokens for B1/B16 and 1024 tokens for B128. Both variants use BF16 weights/query/KV, HND page 32, mixed chunk 16384, graph cap 128 with the existing 20 buckets and token capacity 1600000. B1 is a regression gate for the small QKV epilogue path; B16/B128 exercise the reused normalization kernel.

Candidate diagnostic traces contain exactly 36 calls each at B16 and B128 to `fused_qknorm_warp<128l, true, __nv_bfloat16, QKNormMRoPEEpilogue<true>, false>`. `dispatch-verification.json` records exact names/counts and compressed trace hashes. The common normalization load/reduction/store is reused; rotation and cache writes run through its epilogue hooks.

An external startup-only diagnostic matched validated public GEMM tactics before graph capture. Both `matched-policy-restored.json.gz` records confirm the search method was restored before evaluation. This controls a known startup confound; it is not a normal-independent-startup experiment or a production tuner change.

Median fixed-state raw CUDA graph replay milliseconds, five rounds of 50 after 10 warm replays:

| Batch | Context | Control | Candidate |
|---:|---:|---:|---:|
|1|8192|2.195152|2.187882|
|16|8192|4.922007|4.932593|
|128|1024|5.613600|5.614059|

These small differences do not establish serving improvement or noninferiority. Replays do not advance contexts and exclude request scheduling. Input-buffer hashes and per-round times remain in each report. No serving TPS is inferred.

## Preserved evidence

- `model/comparison-1-16-128.json.gz` and its final log preserve the completed comparison; the initial B1 gate is also retained.
- `model/control` and `model/candidate` contain compressed launches, admission/source manifests, startup tactics, policy validation/restoration, logs, reports and traces. Large `outputs.pt` tensors remain remote.
- `model-tools` preserves harness source as `.py.txt.gz`; root provenance includes the candidate preparation script, namespace-test log and exact source manifests.
- `candidate-source-diff.json.gz` proves five changed production files, two deleted standalone files and no additions. Full admission remains separate from these focused differences.
- `cache-preservation.json.gz` and `baseline-generated-cache` retain four generated test-cache files removed before source admission. This cleanup did not change production source. The retained cache README is named `retained-readme.raw.gz` to avoid formatting compressed data as Markdown.
- `remote-inventory.json` inventories62 original files, including SHA256 and byte size for six omitted logits tensors. `local-verification.json` maps56 copied payloads to compressed filenames and verifies their original decompressed bytes. All raw payloads are gzip-compressed to preserve bytes through formatting hooks; use `gzip -dc FILE.gz` to inspect them.

Collection began only after both model processes were terminal with `MODEL_BATCH_COMPLETE 128`. It launched no experiments or comparison jobs and made no production edits.
