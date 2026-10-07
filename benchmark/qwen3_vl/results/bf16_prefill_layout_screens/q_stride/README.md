# Q-stride and producer compaction screen

Production large-M Q remains a view into QKV with token stride6144 BF16
values. In-place normalization and MRoPE preserve that stride. Earlier
prefill backend screens used contiguous Q; this isolates a different question.

GPU1, pinned devbox environment, seed20261007. Observed total M16331/B126
is retained, but individual lengths were not captured. Synthetic compatible
prefill lengths [8200,8007] plus124 decode rows are used. Context tests only
the two-prefill prefix, with KV lengths [8200,8007] or [8200,8200]. Decode
work is excluded. All output comparisons are bitwise exact.

| KV lengths | Strided Q us | Compact Q us | Copy plus context us |
|---|---:|---:|---:|
|8200,8007|933.417|908.830|1036.656|
|8200,8200|961.823|948.766|1074.903|

`compact_rope.py` is an isolated copy of the existing Triton MRoPE producer,
modified only to direct Q stores into a separate contiguous output. K remains
in place. This avoids an extra Q-copy launch and preserves BF16 rounding.
`chain.py` compares existing QK normalization, MRoPE and context against the
compact-output variant. A common input-reset copy prevents successive replays
from repeatedly transforming their inputs. Both use identical static paged KV;
the common KV write is omitted. Q, K and attention outputs match bitwise.

| KV lengths | Normal chain us | Compact chain us |
|---|---:|---:|
|8200,8007|1110.774|1112.618|
|8200,8200|1132.217|1135.166|

Eight alternating CUDA-graph measurements are retained. Timings show warmup
drift; no reliable chain gain appeared. Producer/layout costs erase the
isolated attention benefit. No production patch or serving run was promoted.
