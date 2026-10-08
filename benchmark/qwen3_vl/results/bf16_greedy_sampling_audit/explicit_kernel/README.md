# Explicit cast kernel gate

The generated-equivalent external Triton kernel reproduces the proven configurations: B64 BLOCK 512/8 warps/1 stage; B128 BLOCK 1024/4 warps/1 stage. It performs one BF16 load, exact FP32 conversion and caller-owned output store. No torch.compile dependency is introduced by the production operator.

| Batch | Torch copy us | Winning compiled copy us | Explicit copy us |
| --- | ---: | ---: | ---: |
| 64 | 18.474 | 12.3288 | 12.3283 |
| 128 | 53.270 | 18.9016 | 18.8314 |

Same buffers, actual checkpoint tied-weight logits from seeded synthetic hidden states, ten counterbalanced rounds with 100 retained graph replays. Kernel outputs and argmax IDs match on changed input, ties, NaNs, infinities and all 65,536 BF16 bit patterns. Returned output identity is preserved. Historical report key `compiled` in this specific screen denotes the explicit-copy-plus-unchanged-argmax pair; `compiled_cast` denotes the winning compiled copy alone. No model gain is inferred from this screen.

Production integration is committed separately as `f9a704d70a`. It has a lazy elementwise registry entry and only changes matching existing logits-buffer copies: shapes 64/128×151936, BF16, contiguous CUDA tensors on the same device, SM103, no-grad and no outer compilation. Softcap, truncation, mismatched buffers and unsupported inputs retain the existing implementation. The public operator preserves full FP32 output and caller ownership.

External checks execute an AST-isolated real production method and actual production kernel: 18 checks pass, including eligible shapes, CPU, dtype, input/output layout, truncated vocabulary, missing/mismatched buffers, grad mode, unsupported architecture, outer-compile fallback, softcap-helper fallback, nondefault stream and changed-input graph. Actual outer Torch compilation preserves the fallback and exact output identity. These are isolated checks, not full model integration proof.

Existing unmodified namespace test: 67 passed. The source-only remote copy initially lacked the test file; that no-tests-ran outcome is retained, then the unchanged repository test was supplied and passed. The first expanded external probe constructed SimpleNamespace inside a fullgraph test wrapper, unsupported by Dynamo; the corrected wrapper constructs metadata outside compilation. This was a harness error, not a production failure; both logs remain.

Prototype PID 572459 terminated normally on GPU 0. Production source hashes, exact prototype/probe scripts and logs are archived. The later production model/serving gates are separate. No new PR test files or benchmark files were added by the production implementation.
