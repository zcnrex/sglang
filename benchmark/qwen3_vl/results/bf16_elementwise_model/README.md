# BF16 elementwise fixed-prefix model evidence

These completed experiments preserve failures and a startup-tactic confound. They do not establish an accepted normalization change or a serving throughput gain. The source trees, launch arguments, startup tactics, device admission, logs, comparisons, per-batch reports and diagnostic traces are retained in the three experiment directories.

| Experiment | Context | Result |
|---|---|---|
| `bf16-normalization-model` |8192 at B1/8/16; attempted8192 at B128 | B1 exact; B8/B16 decode logits differ under the strict gate. Both arms ran out of memory at B128. |
| `bf16-normalization-model-b128-short` |1024 at B128 | Completed; strict numerical comparison failed. |
| `bf16-activation-model` |8192 at B1/8/16;1024 at B128 | B1/8/16 exact across both seeds/checkpoints. B128 strict comparison failed, with a differing startup GEMM tactic. |

The fixed-prefix protocol uses two seeds,8192-token inputs except the explicitly shortened B128 reruns, and identical teacher-forced decode tokens. It saves prefill, decode0 and decode14 logits. A strict failure means nonidentical logits, not a change of BF16 storage precision. The comparisons retain max-absolute difference, normalized RMS, KL and top1 agreement. No thresholds have been loosened in this collection.

Normalization B8/B16 maximum absolute logit difference is0.1875; minimum checkpoint top1 agreement is0.75. Short-context B128 reaches0.25 maximum absolute difference and0.9453125 minimum top1 agreement. B1 is outside the new normalization dispatch and its exact result does not validate the changed path. Normalization startup tactic files are identical across the original pair.

Activation B1/8/16 is exact at all six saved checkpoints per batch. At B128, the maximum absolute difference is0.25 and minimum top1 agreement is0.9453125. Startup tactics differ specifically for BF16 GEMM `(128,9728) × (9728,2560)`: control selects tactic2 and candidate tactic1. This makes B128 unsuitable for attributing the numerical difference to the activation change alone. No outcome from later matched-tactic attempts is included here.

## Fixed-state timing

Median milliseconds per raw graph replay from five rounds of50 replays after10 warm replays:

| Experiment | B | Control | Candidate |
|---|---:|---:|---:|
| Normalization,8192 |1|2.2199|2.2286|
| Normalization,8192 |8|3.4766|3.4107|
| Normalization,8192 |16|4.9408|4.8738|
| Normalization,1024 |128|5.6699|5.5922|
| Activation,8192 |1|2.2223|2.1877|
| Activation,8192 |8|3.4786|3.4450|
| Activation,8192 |16|4.9407|4.9148|
| Activation,1024 |128|5.6842|5.5997|

These timings are diagnostic fixed-state model graphs, not request latency or end-to-end throughput. Numerical failures and the B128 activation tactic mismatch prevent treating all rows as accepted improvements. The8192 and1024 context rows are different workloads. Each compressed report preserves graph keys, integrated decode counts, replay buffer hashes and per-round times.

## Preservation and verification

- `remote-inventory.json` records original remote paths, byte counts and SHA256 hashes for125 files. Sixteen large `outputs.pt` files remain remote; their bytes were hashed without downloading them.
- `local-verification.json` verifies all109 copied files against original bytes after lossless decompression where applicable. Raw stored hashes describe compressed bytes; original hashes describe decompressed bytes.
- All copied raw artifacts use gzip to preserve original bytes through formatting hooks; `gzip -dc FILE.gz` reconstructs their original bytes. Their original hashes are in both inventories.
- `bf16-elementwise-model-tools` preserves the collected harness variants as `.py.txt.gz`, including full/short versions and the first launcher. Launch records retain the invoked script path. The directory was collected after execution; absent a per-launch script hash, its current files are evidence of retained versions rather than proof that an overwritten path had identical bytes at every earlier launch.
- The original B128 memory failures are retained in both normalization `run.log.gz` files. No failed run was silently replaced.

Collection performed no GPU experiments, production changes or commits. Files are ready for review and archival.

Raw payload names append `.gz` to the original filename (harness Python files use `.py.txt.gz`). For example, inspect a comparison with `gzip -dc bf16-activation-model/comparison-1-8-16-128.json.gz`. Inventories and this README are generated archival metadata and remain plain text.

The retained harness README is named `bf16-elementwise-model-tools/harness-readme.raw.gz` to avoid text hooks treating a compressed README as Markdown.
