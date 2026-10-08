# Eight-output register epilogue

The only kernel change is to compute/store eight BF16 activated outputs per128-bit store instead of retaining the entire activated subtile. Producer, tile256×256, cluster2×1, two-CTA mode, gate/up interleaving, BF16 intermediate cast and activation order remain unchanged. Source patches apply to the ragged_shapes prototypes; no production files were edited.

| GPU | M | Mean paired latency reduction | Positive pairs |
| --- | ---: | ---: | ---: |
| 4 | 16331 | 2.356% | 8/8 |
| 5 | 16331 | 2.314% | 8/8 |
| 4 | 16384 | 0.602% | 4/8 |
| 5 | 16384 | 1.624% | 6/8 |

Same-process four-path confirmation uses retained graph state, forty common warmups, opposite forward/reversed order across GPUs, eight blocks and twenty replays per interval. Each graph rotates four weights beyond L2. Actual production dispatch is F.linear plus existing activation. All four weights compare bitwise to same-native materialized BF16 output plus activation;256-row and unused-half canaries pass. Graph DOT files prove candidate kernels were captured. Clock/power telemetry is retained. This is standalone evidence, not a serving claim or proof of equivalent model accumulation.

Targeted NCU reports1,633,772 local-memory spill requests, down53.6% from3,521,536 in ncu_original. Registers remain255/thread and shared memory229.63KB/block. The correction reduces but does not eliminate spilling. This single profiled kernel is not a paired timing benchmark. NCU command used --profile-from-start off --launch-count1 --section MemoryWorkloadAnalysis --section LaunchStats --clock-control none; candidate selected by PROFILE_KIND=candidate, GPU4.

Remote root /root/qvl/experiments/native-epilogue-chunk. Timing PIDs411853/411854 completed; telemetry411852 stopped. Profiler412227/child412251 completed. Python /root/qvl/venv-sgl/bin/python; PYTHONPATH=/root/qvl/sglang-prefix-production/python; MAX_JOBS=8; CUDA_VISIBLE_DEVICES=4/5; PAIR_OFFSET=0/1. Model validation is separate and does not modify this frozen screen. Original-artifacts archive preserves exact pre-format bytes.
