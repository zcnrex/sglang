# Cast-only compiled callable

Bounded follow-up on verified-empty GPU 0, PID 541230, terminal normally. The earlier rejected combined cast/argmax experiment is unchanged. No production changes, model gate or serving run.

Startup compilation covers only `lambda x: x.float()`, with `fullgraph=True,dynamic=False`. The pair then uses unchanged `torch.argmax(...,out=ids)`. Both paths preserve the full FP32 logits output and int64 IDs. Input scope is actual pinned tied checkpoint weights multiplied by seeded synthetic hidden states, not captured model activations; GEMM is outside timing. Graphs, tensors, outputs, callables and compiled output owners remain live.

| Batch | Eager cast, us | Compiled cast, us | Eager argmax, us | Eager pair, us | Compiled-cast pair, us | Pair saving, us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 12.311 | 8.228 | 16.426 | 29.017 | 24.608 | 4.409 |
| 64 | 18.474 | 8.233 | 20.531 | 42.652 | 29.081 | 13.571 |
| 128 | 53.271 | 19.619 | 30.486 | 84.036 | 49.220 | 34.816 |

Ten counterbalanced rounds each use 100 retained graph replays after five warmups; startup compilation is excluded. Separate component costs need not sum exactly to pair costs because memory residency/order differs. Clocks are not locked or measured during timing. This demonstrates a standalone cast implementation gain, not a model or serving speedup. B128 saving is roughly 0.14% of a 25 ms decode step.

All initial/changed-input checks compare FP32 storage words exactly and require exact token IDs. A row containing all 65,536 BF16 bit patterns also matches, covering signed zeros, finite values, infinities, signaling/quiet NaN encodings and signs for this conversion. Separate argmax rows verify first equal maximum, first of multiple NaNs, all negative infinity and first positive infinity. These bounded checks do not establish arbitrary model or sampler integration correctness.

The baseline uses a preallocated FP32 output buffer; compiled `.float()` returns its own retained output. Integration must preserve caller buffer identity/lifetime where required, avoid runtime compilation, and measure any output-buffer adaptation cost. No claim is made that directly replacing `_copy_logits_to_buffer` with this callable is safe or equally fast. No follow-on experiment was launched.
