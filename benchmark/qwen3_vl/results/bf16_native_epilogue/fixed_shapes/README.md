# Native BF16 TMEM gate/up epilogue: fixed-shape screen

The vector-store M8192 candidate is a credible standalone win; M128 is rejected. No production/model/serving edits or runs occurred. Existing native CUTLASS Blackwell TMA/tcgen05 producer is retained, with a custom epilogue pairing startup-interleaved gate/up columns in registers. Each accumulator first rounds to BF16, then FP32 SiLU/multiply executes, followed by BF16 output. No global gate/up intermediate is written. The first half of a full-sized output allocation holds the compact output; unused allocation space remains in this prototype.

One-tile mapping shows128 consecutive N accumulators per thread. Adjacent gate/up columns therefore pair locally, without lane shuffles. All four rotating weights compare bitwise against the same native GEMM materialized BF16 output plus existing silu_and_mul, for M128 and M8192. This is not evidence of bitwise equivalence to cuBLAS accumulation or model accuracy.

M128 vector candidate30.144us versus tuned public cuBLASLt+activation25.697us: rejected. M8192 vector initial five-path screen suggested a gain but had baseline clock/order variation. The subsequent two-path, two-GPU confirmation uses actual current _bf16_gemm_dispatch_impl; a temporary one-call spy proves its F.linear fallback and records source SHA0fa01b10b34bec46f4cca14b6dfb96fa67a07d94e68c1d08035b8c851740f431. Original weights feed production; only candidate weights are interleaved once at startup.

| GPU | Production median us | Fused median us | Reduction | Positive pairs |
| --- | ---: | ---: | ---: | ---: |
| 4 | 650.549 | 588.379 | 9.56% | 12/12 |
| 5 | 643.672 | 585.267 | 9.07% | 12/12 |

Six AB and six BA blocks per GPU, opposite first order across GPUs, follow40 common warmups. Each event interval contains20 graph replays; each graph rotates four99,614,720-byte weights (398,458,880bytes, exceeding L2). Candidate graph dumps prove four native kernels, not empty captures. Clock/power/temperature telemetry retained. Pair-order mean reductions remain positive: GPU4 candidate-first10.02%/production-first9.88%;GPU5 9.11%/8.50%.

The vector five-path M8192 screen measured native-only552.629us, native+activation659.724us, isolated activation111.679us, fused584.538us. Thus the fused epilogue adds~32us beyond the native producer while removing~75us against its own unfused chain. This explains why vector stores matter; it is not an extrapolated model gain. Earlier scalar epilogue lost badly (~1149us M8192), retained separately. Fixed tiles are128x128/1CTA for M128 and256x256/2CTA for M8192; no tile sweep.

Invalid-stream files are excluded: an initial fixed stream pointer escaped CUDA graph capture. Corrected calls fetch current stream each invocation. Vector-compile-alignment logs are also excluded: compiler could not infer128-bit store alignment; cute.assume on eight-BF16 offsets encodes alignment proved by guarded M/N multiples and16-byte base pointers. Corrected numerical gates and graph proof passed. Ragged rows are not admitted by this screen.

Interleaving four weights adds398,458,880bytes if originals retained. Observed setup cost6.4–9.8ms per four-weight group, excluded from steady-state timing. Native source/hash and compact modification patches retained; installed NVIDIA source is not duplicated. Remote root /root/qvl/experiments/native-epilogue; Python /root/qvl/venv-sgl/bin/python. All jobs and owned telemetry stopped. New ragged-row validation belongs in a separate subtree.

To reconstruct helper files, extract the installed cutlass.utils.gemm.sm100 epilogue function (including its @cute.jit), prepend that module's imports plus imports for transform_partitioned_tensor_layout and epilogue_tmem_copy_and_partition, then apply the matching gzip patch. Producer patches apply to the installed dense_gemm_persistent.py copy. Original producer SHA d59344faf902cb215a2cee3f2ae6415a14589c6ad8f93e5e74e2612c1e6a0810; helper-module SHA de76ff54cdae4b3f0412dc8878ac6c921d1f3d24f3f616cb53a20ea3cf46d364. Prototype SHA values are in prototype_hashes.json; patches preserve exact bytes.

`original-artifacts.tar.gz` preserves archived files before further formatting; recorded hashes refer to extracted originals.
