# Bounded GPU-event diagnostic

This resolves the initial host-timer ambiguity; it is not a serving benchmark. No production edits. Same external LM-head hook and unchanged frozen source as the initial model gate, with actual tied BF16 weight and startup-only public tuning. Both worker PIDs 431532/431533 exited.

Each variant first ran 15 actual decode steps after a fresh matching 8K prefill. Separately, six counterbalanced blocks replayed each retained decode graph 256 times, preceded by 40 common baseline/candidate warmup pairs. These synthetic replays use the runner's shared DecodeInputBuffers populated by the final real decode: identical input tokens, position and sequence-length tensors, no host updates between variants. Replays overwrite the same KV positions; they do not simulate 256 growing-context tokens. The buffer tensor names were recorded, but snapshots were not serialized or compared after replay. Actual independent-prefill numerical checks cover first/fifteenth logits and all generated tokens, all bitwise equal. The synthetic diagnostic is therefore a fixed-state graph timing probe, not a fresh-input correctness extension.

| Batch | Baseline / candidate graph median (ms) | Median paired speedup | Actual LM-head GPU time in warmed trace (us) |
|---|---|---|---|
| 4 | 2.822079 / 2.809562 | 0.4529% | 126.081 / 115.200 |
| 16 | 4.960881 / 4.948850 | 0.2403% | 128.096 / 116.128 |

All six pairs at each batch improved. The roughly 12 us total-graph improvement matches the warmed LM-head kernel saving. B16 candidate uses the same kernel name for 36 short projection launches plus the final 116.128 us LM-head launch; summing all name matches would misattribute work. Separate standalone LM-head traces identify its kernel family and the graph's final long launch identifies its instance.

Single-sequence-order actual 15-step event medians were 3.996416 / 3.949728 ms at B4 and 5.658656 / 5.633152 ms at B16. They include host launch gaps between event recording and completion, and are weaker evidence than amortized graph pairs. Profiler traces add overhead and are used for attribution, not headline timing. Clock/temperature/power snapshots are in report.json; clocks were not locked, no system settings changed. The fixed-state graph gain does not establish closed-loop throughput or TTFT benefit. No serving or GSM run was launched.

## Exact harness hashes

- `qvl-lmhead-diagnostic.py`: `11880d00017bbc0bc46ffdf598fccf242fac703ca8df8f83e3184ad07e261d31`
- `qvl-lmhead-diagnostic-launch.py`: `5f5a60997835385ed6cc6bc37ab39f4b7cdf0da9b75c6795d063bebde59ebe45`
