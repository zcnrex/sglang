# BF16 QK normalization and rotary standalone protocol

Compare the existing CUDA implementation from immutable source d7b4556d6b with the adapted operator in the existing Triton module. Use GPU 7 only; production files are not modified by this harness.

The cases cover batches 1, 2, 4, 8, 16, 32, 64 and 128, with 36 disjoint layer states and actual checkpoint Q/K normalization weights. Q/K/V use BF16 with packed row stride 6144. Both contiguous and interleaved multimodal axis maps use distinct position rows and row stride 128. Cache variants use physical HND page size 32, disjoint buffers for each arm and layer, and logical cache views. Initial writes use valid slots. Changed-input graph validation changes inputs and positions and marks the final slot negative. Valid slots are restored for timing.

Bitwise equality is required for Q/K/V and the complete K/V cache buffers before timing. Layers 34 and 35 add wide dynamic-range and sparse large-lane normalization inputs. A numerical failure stops the run and records the exact case; tolerances are not relaxed.

Latency uses CUDA events around retained graphs containing 36 operator calls. Eight rounds reverse arm order, with 100 replays per arm per round. Inputs/caches reset outside timing; in-place normalization/rotation evolves the Q/K contents during repeated graph replay. Results are operator-only latency, not model or serving gains. Dependent GEMM/attention interactions require the later model gate.


## Result: rejected

Attempt 1 failed B1 initial layer 0 because the draft omitted the zero weight-shift option. All 5120 Q/K elements differed, maximum absolute difference 3.71875. No timing ran.

Attempt 2 explicitly supplied zero weight shift. B1 through B64 passed initial and changed-input graph bitwise checks for both axis mappings and cache modes. B128 failed initial contiguous-axis/no-cache layer 4: 64 differing elements, maximum absolute difference 1.71875. B128 timing did not run; no further diagnosis or model launch was performed. The larger completed batches also regressed in latency. This candidate is rejected, not a validated replacement.

Interleaved-axis medians (microseconds per layer):

| Batch | Original no store | Adapted no store | Original cache store | Adapted cache store |
|---:|---:|---:|---:|---:|
| 1 | 1.3103 | 1.2536 | 1.7077 | 1.5962 |
| 2 | 1.3150 | 1.2538 | 1.8247 | 1.6330 |
| 4 | 1.3127 | 1.2565 | 1.8243 | 1.7110 |
| 8 | 1.3718 | 1.3855 | 1.8548 | 1.7823 |
| 16 | 1.4859 | 1.6491 | 1.9293 | 1.8302 |
| 32 | 1.5673 | 2.4513 | 2.0015 | 2.4527 |
| 64 | 1.9968 | 3.8739 | 2.4310 | 3.8779 |

`raw-evidence.tar.gz` preserves the exact original scripts, logs and reports before local whitespace normalization. The complete remote originals remain under `/root/qvl/experiments/bf16-qk-norm-rotary`.
