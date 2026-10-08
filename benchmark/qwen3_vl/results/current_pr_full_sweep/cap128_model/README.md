# Cap128 actual-model stride-fix gate

PASS: seeds123/124, B8 input8192/output16. Complete [8,151936] prefill, first-decode and fifteenth-decode logits match bitwise, as do all128 generated tokens per seed. Both arms execute30 graph8 decode calls and preserve all12 shared input buffers through additional fixed-state replay.

Control is exact PR cd99227527cb08f84274d282a87a53c8771a0822 at /root/qvl/sglang-current-pr-full-sweep. Candidate /root/qvl/sglang-current-pr-cap128-fix differs in exactly one file, qkv_norm_mrope.py SHA256 4c9d946f0b826b8607760c3d4a86c498efaf720812214938666fcab2ceb584ec. All10,245 tracked source files are verified. The later formatted file fbf528dfcd39f1b0e31a79017fd70a40f1c88628a43d9453a41f38dd16d9e52c is reported AST-identical by the implementation owner; this live gate preserved the original tested bytes.

Actual eager graph warmup and capture both use positions[3,8] with stride[128,1]. Original control returns None and follows its expected fallback. Fixed public operator returns output in both phases and all36 model layers capture fused dispatch. Observers do not replace computation. Both use identical explicit20-bucket cap128 graph list, pinned1,600,000 KV tokens, BF16 weights/query/KV/output, TRT/HND32/mixed chunk16384 and normal public tuning. No serving sweep or accuracy test is included in this gate.

GPU0 control PID655667 and GPU2 candidate PID655732 are terminal. Raw archive retains source manifests, scripts, logs, layout/capture/replay proofs and complete numerical comparison. Large outputs.pt tensors remain remotely with hashes in numerics_hashes.json; cache directories are omitted. Readable Python copies end .py.txt. This short text-model gate establishes this actual strided graph view, not broad task accuracy equivalence.

Large readable source manifests are losslessly gzip-compressed. Concatenate `original.tar.gz.part*` in lexical order to recover the raw archive; `archive-parts.json` records full and part hashes.
