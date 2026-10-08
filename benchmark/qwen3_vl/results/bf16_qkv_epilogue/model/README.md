# External actual-model B8 gate: passed

Eight opposite-order retained-graph pairs give baseline 3.519405 ms versus candidate 3.480517 ms: 38.888 us saving, 1.105% time reduction. All eight pairs are positive. This is fixed-state GPU-event replay after a real decode, not growing-context throughput or a serving claim.

Both arms use the same frozen `/root/qvl/sglang-m16-down-production` source; full 4231-file remote manifests are retained. Two distinct random input seeds 123/124, B8 input 8192 and output 16, pass bitwise comparison of full prefill, first decode and fifteenth decode logits, plus all 128 generated tokens per seed. All 36 layers dispatched the fused epilogue during candidate graph capture. All 12 shared runner tensor-buffer snapshots remain unchanged after timing; the successful script asserts this and the result audit repeats the assertion.

## Hook scope

Only `Qwen3Attention.forward_prepare_native` at active exact B8 decode is replaced. The hook validates BF16, TP1, ordinary decoder/TRT pool, no speculation, no bias/LoRA, epsilon 1e-6, matching cache write arguments and the model's actual rotary buffers. It runs fused QKV projection plus Q/K normalization, MRoPE and K/V writes, then returns `(q,k,v,False)` so downstream attention does not duplicate cache writes. Prefill/other shapes invoke the original function; attention, output projection, MLP, logits and sampling remain original. Candidate kernels compile in eager graph warmup, never during active capture. The baseline and candidate retained graphs share the final real-decode input buffers and overwrite fixed KV slots during timing.

## Preserved interface failure

`model_gpu2` PID 619529 failed before candidate capture because actual norm parameters require gradients and DLPack refuses their direct export. No numerical/performance result came from it. `model_gpu2_retry1` PID 619795 uses zero-copy `.detach()` views for QKV/norm parameters; kernel source and inference arithmetic are unchanged. This successful retry also adds the explicit unchanged-buffer assertion. No live job was modified or restarted in place.

Both PIDs are terminal and GPU 2 released. No serving or production integration was launched by this gate. Raw archives preserve scripts, logs, source manifests, launch environment/cache provenance and compressed diagnostic traces. `numerics.pt` stays at the corresponding remote directory under `/root/qvl/experiments/bf16-qkv-epilogue`; tensor values are summarized in the bitwise result checks.
