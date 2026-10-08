# M16331 native epilogue bottleneck profile

One representative production GEMM and fused kernel were profiled on GPU4, unchanged 256×256/two-CTA native tile. This is a diagnostic, not a paired performance measurement. SM clocks differed (1.63 GHz production, 1.49 GHz candidate), and production alone excludes the separate activation.

| Metric | Production cuBLAS GEMM | Fused native candidate |
| --- | ---: | ---: |
| Tensor pipeline utilization | 98.1% | 87.9% |
| DRAM throughput | 18.93% | 39.54% |
| Registers per thread | 255 | 255 |
| Local-memory spill requests | 0 | 3,521,536 |
| Dynamic shared memory per block | 213.28 KB | 229.63 KB |
| Achieved occupancy | 9.38% | 8.57% |

The candidate's spilling is a concrete potential target. Its epilogue retains all rounded BF16 accumulators and an entire half-width activated result tensor before a vector store. Processing eight output values per store could reduce live ranges while preserving producer and numerical order; this report does not establish that correction works. Occupancy alone is not a target: production has similarly low occupancy while reaching near-saturated tensor utilization. NCU's generic compression and estimated-speedup suggestions are not adopted. The report also flags excess global sectors; their source is not localized here. Detailed preset did not collect separate warp-stall sample sections.

Driver 411232 completed both captures successfully (children 411257 and 411463); initial driver 411148 failed with a quoting SyntaxError before GPU work. Its failed log is preserved. Commands are in per-variant command JSON; scripts and raw reports/CSV are included. Profile selects one kernel after common warmup with --profile-from-start off, --launch-count 1, --set detailed and --clock-control none. The source is /root/qvl/sglang-prefix-production/python; retained inputs, canaries and actual F.linear dispatch are checked before profiling. No production edits or serving runs.

Raw `.ncu-rep` files are gzip-compressed without modifying their contents; decompress before importing into Nsight Compute.
