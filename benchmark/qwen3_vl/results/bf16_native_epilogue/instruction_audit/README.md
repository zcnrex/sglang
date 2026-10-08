# Unchanged-kernel instruction and stall attribution

One focused NCU run profiled the frozen, spill-free FFI/subtile kernel at M = 16331, N = 19456, K = 2560. GPU 0, Python PID 416405; one candidate launch after 100 warmup calls and a bitwise/canary check. Four sections were collected: InstructionStats, SchedulerStats, WarpStateStats, ComputeWorkloadAnalysis. The profiler required 19 replay passes. No kernel, tile, wrapper or production changes were made.

The new evidence does **not** support an output-store or epilogue instruction-issue bottleneck. It points primarily to high tensor execution activity, while leaving source-level wait attribution unresolved:

| Counter | Measured value |
|---|---:|
| Tensor FP pipeline active, elapsed cycles | 95.587% |
| TC pipeline active, elapsed cycles | 95.628% |
| LSU instruction issue relative to peak | 0.320% |
| TMEM instruction issue relative to peak | 0.0187% |
| TMA instruction issue relative to peak | 0.374% |
| TMA pipeline active, elapsed cycles | 0.747% |
| XU instruction issue relative to peak | 11.936% |
| FMA instruction issue relative to peak | 4.943% |
| Register-spill instructions | 0 |
| Long scoreboard stall cycles per issued instruction | 3.588 |
| Wait stall cycles per issued instruction | 2.492 |
| Short scoreboard stall cycles per issued instruction | 0.987 |
| MIO throttle cycles per issued instruction | 0.0392 |
| LG throttle cycles per issued instruction | 0 |

The long-scoreboard category accounts for about 40.6% of the 8.84 average warp cycles per issued instruction. This is a warp-local wait breakdown, not 40.6% recoverable kernel time: tensor work can proceed while producer or epilogue warps wait. Likewise 84.42% scheduler cycles with no eligible warp is not evidence that tensor hardware is idle in a warp-specialized persistent GEMM. NCU's generic suggested speedups must not be treated as additive or realistic performance predictions.

The previous memory-pipe activity of about 95% cannot be assigned to epilogue stores from these aggregates. The full compute report also lists TMEM cycle activity near tensor activity; that does not isolate the epilogue's explicit TMEM loads. Low LSU/TMEM instruction issue pressure and negligible LG/MIO throttling argue against a store-width change as the next useful experiment. TMA issue pressure is low, and tensor execution is already about 95.6% active, so reducing operand pipeline depth is also unsupported. Compute activity is not a precise FLOP-efficiency ceiling and does not prove an absolute upper bound on optimization.

No PC-sampling locations were available: NCU warns that optional smsp__pcsamp_sample_count was not collected. Therefore this capture cannot definitively locate the long-scoreboard dependency in operand handling versus epilogue handling. Raw opcode histograms are retained in the binary report, while CLI tables omit their instance names. No second profiling run or kernel variant was launched to fill that gap.

**Decision:** no bounded kernel change is justified by this evidence. Preserve the existing exact BF16 arithmetic; the generic FMA-fusion suggestion would change rounding and is outside this task. The measured model benefit remains the appropriate promotion criterion.

Clocks were not locked (--clock-control none); counter collection replay and warm fixed inputs differ from the rotating-weight timing protocol. Keep these caveats when relating the profile to end-to-end inference. The compressed raw instruction.ncu-rep.gz, full/compact text exports, raw CSV, selected metrics, exact harness and source hashes are adjacent. Run root: /root/qvl/experiments/native-epilogue-instruction-audit. The worker finished and no GPU work remains.
