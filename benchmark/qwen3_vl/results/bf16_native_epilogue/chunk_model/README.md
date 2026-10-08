# Chunked native epilogue mixed-model gate

The standalone kernel gain did not produce a robust model improvement, so this candidate does not advance to serving.

| GPU | Control mean, ms | Candidate mean, ms | Latency reduction |
| --- | ---: | ---: | ---: |
| 4 | 136.869 | 138.970 | −1.535% |
| 5 | 135.895 | 135.828 | +0.050% |

Each loaded model runs four warmup variants followed by eight measurements (four control, four candidate), ABBA order on GPU4 and BAAB on GPU5. The same model and identical committed source include packed FA4. Only the external Qwen2MLP forward hook changes: candidate interleaves each gate/up weight once, uses the exact chunked native producer, then calls the normal down projection. Setup, interleaving and compilation occur in warmup. Full extend-forward wall time includes Python dispatch, DLPack conversion and removal of separate activation. This prototype incurs descriptor conversion per layer; timings do not isolate that overhead from kernel/model behavior.

Both variants use BF16 weights, inputs and KV cache, TRT decode, HND page32, mixed16k and ordinary decode graphs. No speculative decoding or precision change. Source audit verifies all4,229 Python files against the frozen packed-FA4 manifest. Exact commands, environment, hook SHA and source verification count are retained. GPU-specific startup autotuning may differ across devices; each within-device comparison uses the same loaded model/tactics.

The exact measured geometry is B42/M16331: context prefixes[7104,0,0], query lengths[1096,8188,7008], plus39 real one-token decode tails with the original individual cache lengths. The observer record with time_ns1791415348741804163 is included. Synthetic token IDs use NumPy/Torch seed42; this reproduces measured shapes/cache geometry, not the original prompt texts. Each variant starts from cleared request/KV allocation state, builds tail caches with the normal model, and uses the actual generated token for each tail.

All candidate measurements prove36 fused MLP calls and36 packed-FA4 calls; controls prove0 fused calls and36 packed-FA4 calls. During warmup, next-token logits for the target prefill and three decode steps compare bitwise for all42 requests, with greedy tokens equal. This bounded synthetic result is not a full accuracy/equivalence claim.

The hook retains36 additional interleaved weights (3,586,129,920 bytes) plus635,471,872-byte full-width scratch while writing a compact activated result into its first half. Setup timings and exact scratch byte count are in report.json. Original weights remain available for control. No production files were edited. Model workers412762/412763 completed and released GPUs4/5. Remote root /root/qvl/experiments/native-epilogue-chunk-model; original-artifacts.tar.gz preserves raw bytes before evidence formatting. The separate smaller-subtile experiment is not included in this result.
