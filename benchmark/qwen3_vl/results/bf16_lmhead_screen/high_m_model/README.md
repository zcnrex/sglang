# High-M vocabulary short model gate

All six fixed-graph pairs per shape improve. These are short-context 128 diagnostics, not 8192-token serving results; relative gains are inflated by lower attention cost.

| M / GPU | Baseline / candidate median ms | Median paired improvement |
|---|---|---|
|32 / 4|2.461211 / 2.450796|0.42311%|
|64 / 5|2.805690 / 2.793865|0.42334%|
|128 / 6|3.291647 / 3.283902|0.23599%|

Both arms use current frozen lmhead-production including M4. All three 4229-file manifests match the archived 5e1601731b candidate manifest exactly. External dispatch changes only the exact tested vocabulary M, with the actual tied BF16 weight and normal public startup autotuning before graph capture. Existing projection tactics are shared within the same loaded model. No production edits.

Matching real prefills use input length 128 and seed 123, followed by 15 decode calls (16 greedy tokens per sequence). Prefill, first and fifteenth decode full logits match bitwise; every token agrees. A second real sequence with seed 124 has bitwise final logits and identical final tokens. One optimized target-M graph capture is recorded in each case. These bounded checks do not establish full accuracy equivalence.

Retained baseline/candidate graphs share final real-decode inputs and fixed KV positions. Forty common warmup pairs precede six opposite-order timing pairs, each 256 replays under CUDA events. All shared tensor snapshots are unchanged after timing. This is repeated fixed-state replay, not growing-context generation. Actual 15-step event times include host gaps and are reported separately. Profiler captures occur after timing and are diagnostic only. Large numerical tensors and raw traces remain at /root/qvl/experiments/lmhead-highm-model/m{32,64,128}; compact original archive /tmp/lmhead-highm-model.tar.gz retained locally/remotely.

Workers 494891/494955/495020 completed and GPUs 4/5/6 were released. No serving job was launched.
