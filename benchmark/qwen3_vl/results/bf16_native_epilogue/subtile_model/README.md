# Smaller-subtile mixed-model gate

The separate smaller-subtile candidate improves this short mixed-forward test on both GPUs, with seven of eight adjacent pairs positive. This modest result is not a serving or accuracy-equivalence claim.

| GPU | Control mean, ms | Candidate mean, ms | Latency reduction | Positive pairs |
| --- | ---: | ---: | ---: | ---: |
| 4 | 136.576 | 135.538 | 0.760% | 3/4 |
| 5 | 134.982 | 132.668 | 1.714% | 4/4 |

The exact same model/source/control and B42/M16331 geometry as ../chunk_model are reused. Only the frozen native producer/helper import changes to the separately validated smaller-subtile implementation. Hashes are asserted before loading: producer1240d08cb558c0918fd07527a3540f881803c88e4c03bc8a59cf299fa5a60655 and helper68b7746ce1e6a785124cf2fc86d8de732e754419874c88a2c26b2d221a794d6f. Both variants include current committed packed FA4, M1 decode optimization, BF16 weights/queries/KV, HND page32 and ordinary decode graphs. Source audit checks all 4,229 Python files against the frozen current manifest; no production edits or speculative decoding.

Four warmup variants precede eight measurements per GPU (four control and four candidate). ABBA order on GPU4 and BAAB on GPU5 counterbalances adjacent pairs. Full extend-forward wall time includes Python dispatch and descriptor creation. Every candidate measurement records 36 fused MLP calls, 36 packed-FA4 calls, zero compilation and zero weight-descriptor setup; the 36 input DLPack conversions remain included in timing. There is exactly one native compilation and 36 weight transformations per loaded model, all during warmup. No dispatch cost is subtracted.

The recorded geometry has context prefixes [7104,0,0], queries [1096,8188,7008] and 39 one-token tails with the actual recorded cache lengths, giving 16,331 input rows. Seed42 synthetic IDs reproduce the measured geometry, not original request texts. Each variant rebuilds request/KV state using the normal model. Next-token logits for the mixed prefill and three decode steps are bitwise equal for all 42 requests in the warmup comparison, and greedy outputs match. This bounded observation does not establish general numerical equivalence or GSM8K accuracy.

Only mixed-prefill work is affected, and this gate tests one measured shape. Earlier whole-workload profiling attributed roughly 27.6% to mixed forwards, so even applying the observed model gain to every mixed step would suggest only around 0.2–0.5% total wall-time reduction. That is an optimistic scope extrapolation, not an observed serving gain. Serving is held pending the independent wrapper-overhead diagnostic; no repeat of this unchanged candidate is implied.

Remote root /root/qvl/experiments/native-epilogue-subtile-model. Workers414421/414422 completed and released GPUs4/5. Commands, environment, observer/algorithm-hook hash, setup storage and per-forward counts are preserved. Original-artifacts.tar.gz retains exact raw files before formatting. External weight storage and full-width scratch costs are unchanged from chunk_model.
