# High-M BF16 vocabulary public Lt screen

All three standalone cases improve, with bitwise initial and changed-input outputs. No model or serving claim follows. The saving is only about 9–13 microseconds once per decode step.

| M / GPU | Torch median µs | Public Lt median µs | Median paired speedup | Selected tactic |
|---|---:|---:|---:|---:|
|32 / 4|132.301|118.847|11.349%|2|
|64 / 5|131.341|120.946|8.581%|7|
|128 / 6|138.651|129.188|7.377%|6|

Actual pinned Qwen3-VL-4B tied BF16 embedding weight [151936,2560] is approximately 778 MB, exceeding L2. Revision ebb281ec70b05090aa6165b016eac8ec08e71b17; weight key/file and environment versions are retained in reports and public tuner exports. Random BF16 input seed is 700+M. Default Torch BF16-output GEMM is compared with public FlashInfer mm_bf16 cublaslt. Each worker uses normal public autotune for only its exact bucket before capture; no forced tactic or private tuning policy. Selected indices are evidence, not a proposed hardcoded interface.

Ten paired graph measurements alternate order, with three warmup replays and forty timed replays per measurement. Captured graphs and input/output/weight owners remain live. Tuning and compilation are excluded. Changed input replay precedes timing; both outputs match bitwise and argmax agrees. FP32 reference uses TF32 disabled; both methods have identical normalized RMS about .001660–.001661. These bounded random-hidden-state checks do not establish model accuracy.

Workers 493182/493183/493184 completed; GPUs 4/5/6 released. Remote root /root/qvl/experiments/lmhead-highm, executed script /tmp/qvl-lmhead-highm.py. No production files changed and no model job launched. Raw script SHA256: acf52f4f11d72f65e08e0d0ec027392c1879faaf0c3e6b0b8a8706633101601c.
