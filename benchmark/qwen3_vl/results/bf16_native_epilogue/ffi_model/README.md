# Direct-Torch FFI subtile model gate

The direct-Torch FFI wrapper shows small gains against production in this short mixed-model test, but no established additional model benefit over the earlier subtile wrapper. The independent runs are not a direct paired comparison between wrappers. No serving run or production change follows this evidence.

| GPU | Control mean, ms | FFI candidate mean, ms | Latency reduction | Positive pairs |
| --- | ---: | ---: | ---: | ---: |
| 4 | 137.680 | 136.793 | 0.644% | 3/4 |
| 5 | 134.814 | 134.281 | 0.395% | 3/4 |

The same current source, synthetic requests, exact recorded B42/M16331 mixed geometry and ABBA/BAAB eight-measurement protocol as ../subtile_model are used. Only the external fused MLP callable changes to the independently validated direct-Torch FFI wrapper. It takes Torch input/weight/output tensors directly and obtains the current CUDA stream from the TVM-FFI environment. Its GPU producer and BF16 operation order remain unchanged.

Producer SHA 0a0897443d7a4f41361ef24a030ea6356dd6e406d510d5c3f419542bfaf19b30 and helper SHA 68b7746ce1e6a785124cf2fc86d8de732e754419874c88a2c26b2d221a794d6f are asserted before loading. All 4,229 current Python source files are verified against the frozen packed-FA4 manifest. Baseline and candidate retain identical BF16 weights, queries, KV cache, ordinary decode graphs and attention configuration within each loaded model.

Every measured candidate forward records 36 fused MLP calls and 36 packed-FA4 calls, with zero compilation, zero input/weight DLPack descriptor conversions and zero weight transforms. Each model performs one compilation, one input descriptor conversion, one weight descriptor conversion and 36 weight transforms in warmup. Output descriptor construction is also confined to that single setup branch. Full-forward wall time includes the direct-Torch call and tensor-view construction; no host overhead is subtracted.

Prefill and three decode-step logits compare bitwise for all 42 synthetic requests, with greedy tokens equal. This is a bounded numerical observation, not full accuracy equivalence. Six of eight adjacent timing pairs are positive; the small effect and independent-run variability do not justify claiming the standalone ~29-us submission saving adds directly to model speed. Any CPU/GPU overlap explanation remains an inference, not measured attribution. Only one mixed shape is covered, and possible whole-serving gain would be smaller still.

Workers 415336/415337 completed and GPUs4/5 were released. Remote root /root/qvl/experiments/native-epilogue-ffi-model. Exact commands, environment, hook hash, source/geometry metadata and per-forward counters are included. Original-artifacts.tar.gz preserves raw bytes before formatting. No old candidate files or scheduler-owned wrapper modules were changed.
