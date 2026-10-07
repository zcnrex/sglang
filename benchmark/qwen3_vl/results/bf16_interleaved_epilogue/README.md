# Interleaved BF16 gate/up epilogue screen

Rejected for integration. This bounded standalone experiment reuses the existing
Triton MoE GEMM with a single expert and its interleaved fused SwiGLU epilogue;
it does not implement a new TGV epilogue. Target dimensions are N=19456, K=2560.
The frozen production cuBLASLt patch was not changed.

| Tokens M | Torch GEMM + SiLU | Autotuned public Lt + SiLU | Best fused | Fused / Lt |
| --- | --- | --- | --- | --- |
| 128 | 28.06 us | 25.88 us | 30.30 us | 1.17x |
| 8192 | 643.20 us | 641.93 us | 1242.40 us | 1.94x |

Values are medians of five CUDA-event rounds, twenty graph replays per round.
Each graph rotates four independent weights (398,458,880 bytes, exceeding L2).
The three bounded tiles are 32x128x64, 64x128x64, and 64x256x64, all four warps
and three stages. No token sorting/routing or layout copies occur inside timed
calls. Baselines include both GEMM and reference SiLU-and-multiply. Measurements
are sequential screening, not randomized crossover; large-M clocks/timings drift
in the raw samples, but even the fastest fused sample loses substantially.

All six cases match the same GEMM's separately materialized BF16 outputs plus
reference activation bitwise. They also match Torch GEMM plus activation bitwise
on the seeded random input tested. This is not a proof for every input. The fused
source explicitly casts accumulators to BF16 before activation, preserving that
rounding boundary. No image/model accuracy or serving test was run.

The large-M compiled resource snapshots include unfused then fused entries for
each tile. All report zero spills. The best fused tile uses 86 registers and
73,744 shared-memory bytes, versus 96 registers for its unfused counterpart.
Resource collection was added after the initial M128 run, so M128 has no resource
snapshot. No further tuning or model integration is justified for this producer.
This result does not rule out a separately engineered tensor-memory/TGV epilogue.

Interleaving each weight once adds 99,614,720 bytes if the original layout is
retained. Observed one-time indexing/copy costs were 1.58–2.25 ms per weight;
these startup costs are excluded from steady-state timings and include first-use
effects. A production implementation would also need to preserve original-layout
consumers. No such integration was attempted.

Source: `/root/qvl/sglang-public-lt-clean/python`, clean baseline
466c9e0f4073bcad6e4403c6f54cfc8c1821b867 plus the two-file production patch recorded
in `../bf16_decode_lt/production_validation/`. Reused kernel:
`sglang/kernels/ops/moe/fused_moe_triton_kernels.py` (BF16 cast at line 621,
fused activation lines 626–662). Remote Python: `/root/qvl/venv-sgl/bin/python`.
Run on GPU6; unrelated regression workers on GPUs0–5 were preserved.

Reproduce with the archived `screen.py`:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/root/qvl/sglang-public-lt-clean/python \
  /root/qvl/venv-sgl/bin/python screen.py --m 128 --out m128.json
# Repeat with --m 8192 --out m8192.json.
```
