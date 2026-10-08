# One bounded epilogue subtile refinement

The external native producer retains the same2CTA MMA256×256 tile and CTA128×256, but its non-TMA epilogue partition changes from fullCTA to128×64. The existing helper consequently loads four N subtiles,64 FP32 accumulator values per thread instead of256. Activation retains BF16 rounding before FP32 SiLU/multiply and BF16 output, interleaved gate/up pairing and128-bit stores. Accumulator release occurs only after the final TMEM subload with an async-TMEM-load fence, including the conservatively handled overlap branch. The exercised native kernel calls the helper with default non-overlapping accumulation; the overlap branch is not separately validated.

Isolated root /root/qvl/experiments/native-epilogue-subtile. No production or other agent files were modified. GPU6 worker413063, GPU7 launch-shell413573 (Python worker finished before its PID was captured), focused NCU worker413896. Both timing workers finished; no serving/model run was performed.

Retained graph/callable/tensor state and all four rotating ~99.6MB weights per shape avoid the prior harness ownership bug and exceed L2. Shapes M16331/16384, N19456,K2560. Candidate is bitwise identical to native unactivated BF16 GEMM followed by the existing SiLU kernel for all four inputs at each shape;256-row tail and unused output-half NaN canaries pass before timing. This does not claim bitwise equality with F.linear, whose reduction implementation differs. Existing production selector was invoked and verified to choose F.linear for these shapes; production timing includes its SiLU. Inputs use seeded synthetic BF16 values, not model activation captures.

|GPU|M|Subtile us|Unchanged chunk us|Production us|Latency reduction vs production|
|---|---:|---:|---:|---:|---:|
|6|16331|1273.268|1326.208|1351.967|5.82%|
|6|16384|1277.945|1316.999|1328.481|3.80%|
|7|16331|1287.130|1341.856|1361.753|5.48%|
|7|16384|1274.067|1322.374|1343.615|5.18%|

Eight alternating-order rounds per GPU, opposite initial order across GPUs, captured four-GEMM CUDA graphs. Candidate wins every paired round against both comparators. Clocks were not locked and order effects remain visible; report medians and full rounds rather than claim these percentages as model gains. One-time weight interleave occurs before steady-state timing and creates an extra full-size weight copy; it is not a per-call operation. No layout conversion is hidden in production baseline.

Focused NCU M16331 reports **0 local-memory spilling requests**,90 registers/thread,229.63KB dynamic shared memory/block; the prior chunk implementation reported1,633,772 spill requests. This supports the register-pressure hypothesis. NCU uses MemoryWorkloadAnalysis and LaunchStats, --profile-from-start off --launch-count1 --clock-control none; profiler starts only around the candidate after common warmup. Binary report remains remotely at subtile16331.ncu-rep; exported text/log are archived here.

Exact scripts are preserved as .py.txt. Original modules/logs/graphs/results before formatting are in raw_evidence.tar.gz, with source hashes. Rename scripts to their original .py names together and use the pinned /root/qvl/venv-sgl Python, PYTHONPATH=/root/qvl/sglang-prefix-production/python, CUDA_VISIBLE_DEVICES=6 or7 and PAIR_OFFSET=0 or1. No broader tile sweep or model integration is justified without the separate next gate.
