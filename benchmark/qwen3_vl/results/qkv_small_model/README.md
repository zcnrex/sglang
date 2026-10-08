# External B1/B2 QKV fusion actual-model gate

Both batches pass the bounded same-loaded-model gate. Exact PR3016 source is shared between baseline and candidate; only the external exact-B1 or exact-B2 projection/preparation hook differs. Existing B8 and every other dispatch remain unchanged. No production edits or serving runs occurred.

| Batch / GPU | Baseline graph ms | Candidate graph ms | Time reduction |
| --- | ---: | ---: | ---: |
| B1 / GPU0 | 2.334410 | 2.215662 | 5.0868% |
| B2 / GPU1 | 2.533477 | 2.423066 | 4.3581% |

All eight alternating fixed-state CUDA-event pairs favor fusion. Each measurement repeats its retained graph 256 times after common warmup; B2 reverses the initial order. These are fixed-state decode timings, not serving throughput. Input-buffer snapshots remain unchanged.

Seeds 123 and 124 each use 8192 input tokens per request and 16 generated steps. B1 compares all 16 output tokens per seed; B2 compares all 32 output tokens per seed (saved tensor shape [16,2]). Full logits at prefill, first decode and fifteenth decode are bitwise identical, with zero maximum and normalized RMS error. B2 logits are [2,151936], confirming the batch dimension. Each candidate captures all 36 fused layers with positions stride (128,1); input/norm/weight detachment preserves storage pointers. Fresh prefill and cleared state precede each numerical arm.

Workers 680468 and 680532 are terminal. Remote roots: /root/qvl/experiments/qvl-qkv-small/b1/model_gate/gpu0 and b2/model_gate/gpu1. Scripts, launches, normal public tuner cache, source manifests, logs, numeric shapes and reports are retained. Full numerical tensors remain remotely at recorded paths with SHA256 hashes. original.tar.gz preserves unnormalized scripts/logs/reports; readable copies may receive whitespace-only hook normalization. No generated-token distribution or GSM accuracy claim follows from these two synthetic seeds.
