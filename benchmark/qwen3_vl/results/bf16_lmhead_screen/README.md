# BF16 LM-head standalone screen

Actual tied embedding weight from pinned Qwen revision ebb281ec70b05090aa6165b016eac8ec08e71b17, shape [151936, 2560], approximately 778 MB (larger than L2). Random BF16 hidden states; BF16 outputs retained. Public FlashInfer cublasLt autotuning occurs before graph capture. Ten alternating timing rounds, 40 graph replays per measurement. Changed-input graphs retain all tensors. No production changes.

| M | GPU 0 Torch / FI (us) | GPU 1 Torch / FI (us) | Paired speedup GPU 0 / 1 |
|---|---|---|---|
| 4 | 128.460 / 115.240 | 127.423 / 115.290 | 11.45% / 10.41% |
| 8 | 129.302 / 116.735 | 128.139 / 116.617 | 10.76% / 9.83% |
| 16 | 130.505 / 116.955 | 129.059 / 117.004 | 11.57% / 10.31% |

Initial logits were bitwise equal in all six cases, with identical argmax. Changed inputs passed tolerance and argmax checks. Script tolerance is rtol 0.02, atol 1; the recorded initial zero errors are stronger than that loose assertion. Public tuner records select tactics 2/2/3 at M4/8/16 on both GPUs. Those are evidence, not hardcoded dispatch. Logs preserve software versions, selected records and worker PIDs 424904/425064. Both jobs terminal. This saves roughly 12–14 us once per model decode step; no model or serving gain is established by this screen.
