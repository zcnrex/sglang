# Lossless BF16 weight packing: rejected; separate raw-direct lead

The packed direct GEMM is slower. No production changes were made. A separate
uncompressed M1 direct-GEMM lead shows a small short-model gain with numerical
drift; M4 public FlashInfer loses at model level.

## Packed screen

Pinned Qwen3-VL-4B checkpoint `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
Twelve real Q/O/gate/down weights from layers 0/17/35 have 99.262–99.366% of
128-value groups fitting a four-bit exponent delta. Three-bit deltas fit only
17.115–21.111% of these groups. Full statistics are in `stats.json`.

The installed FlashInfer direct kernel loads BF16 weights into registers
(`dense_bf16_gemm_direct.py:227–236`). The isolated `packed_direct.py` changes
that prefetch only, reconstructing exact BF16 bits into the original register
array and retaining original arithmetic/reduction. Split-K uses cp.async/TMA
and was not modified. `prepare.py` records the transformation of installed source.

The codec uses 128-element groups: 128 sign/mantissa bytes, 64 exponent-nibble
bytes, and a separate uint16 base/flag; non-fitting groups retain all raw bits.
Each payload slot retains the original 256-byte stride, so this prototype reduces
read payload, not allocated storage. The codec and raw fallback preserve every
BF16 bit. Metadata adds two bytes per group. Three real concatenated gate/up
weights from layers 0/17/35 rotate across graph launches (298,844,160 bytes,
exceeding L2); N=19456, K=2560. Packing is startup-only and excluded from timing.

| M | Packed direct us | Raw direct us | Public FI us | Production Torch us |
| --- | --- | --- | --- | --- |
| 1 | 55.12 | 16.12 | 17.09 | 17.64 |
| 4 | 89.58 | 24.63 | 17.20 | 17.76 |
| 8 | 100.36 | 34.46 | 17.82 | 17.80 |

Medians of five rounds of 40 graph replays, three weights per replay. Packed
outputs are bitwise equal to raw direct outputs for all tested M/weights, and
complete packed/unpacked weight tensors compare bitwise. This does not mean
these GEMM reductions match Torch bitwise. `corrected.json`/`corrected.log` are
the valid results. `invalid-fixed-stream.log` records an excluded first run whose
candidate used a fixed pre-capture stream, producing empty graphs; its timings
are invalid. The corrected wrapper obtains the current stream on every call.

## Uncompressed confirmation

Eight randomized-order rounds of 80 graph replays use the same three real
weights, with graph DOT files proving kernel capture. M1 ran on GPU0 and M4 on
GPU1; comparisons are within each GPU. Baseline is Torch mm with BF16 output,
matching the production fallback for this shape (not eligible for current
TGV or tuned split-K selection). No packing is used in these measurements.

| M | Raw direct us | Public FI us | Torch us | Useful lead |
| --- | --- | --- | --- | --- |
| 1 | 16.04 | 17.74 | 17.72 | Raw direct saves 1.68 us/layer |
| 4 | 24.59 | 17.16 | 17.78 | Public FI saves 0.63 us/layer |

Both stated leads win all eight paired rounds. M1 direct tactic is
`DirectTactic(block_size=64, outputs_per_block=2, rows_per_block=1)`.
Public autotuning selects `CuteDSLWarpSplitKBf16Runner` with tuples
`(16,8,128,7,1)` for M1 and `(16,8,128,7,2)` for M4; full cache metadata is saved.
The public M1 path does not select the faster raw-direct tactic in this screen.

Neither uncompressed alternative is bitwise equal to Torch: maximum absolute
output difference is 0.015625, with normalized RMS at most 0.000160 on tested
inputs. Inputs and weights remain BF16 with no quantization. Saving 1.68 us over
36 layers suggests approximately 60 us per M1 decode forward before any runtime
cost; this is an estimate, not a measured model gain. No serving or accuracy
claim follows from these kernel-only results.

Run with `/root/qvl/venv-sgl/bin/python`,
`PYTHONPATH=/root/qvl/sglang-public-lt-clean/python`, and a free visible GPU.
`stats.py` measures weights, `prepare.py` generates the isolated direct loader,
`run.py` performs the packing screen, and `unpacked.py --m 1` (or `--m 4`)
performs randomized confirmation. Scripts retain original remote artifact paths.
The source baseline is 466c9e0f4073bcad6e4403c6f54cfc8c1821b867 plus the frozen
runtime Lt patch; installed package versions are recorded in tuner cache files.

## Same-loaded-model follow-up

`model.py` loads the verified 5f60fb8b67-equivalent source once per batch size.
It captures control and candidate graph backends using the same runner/input
buffers, then swaps the graph backend for eight counterbalanced pairs. Each run
uses identical seeded inputs, 8192 input tokens and 32 output tokens. Candidate
capture records 36 replaced GEMMs; only the selected batch/shape changes.
Configuration and authoritative process IDs are in `model-launch.json`.

| Batch | Baseline decode median ms | Candidate ms | Paired throughput geomean | Positive pairs |
| --- | --- | --- | --- | --- |
| 1, raw direct | 3.0164 | 2.9822 | +1.086% | 7/8 |
| 4, public FI | 3.4444 | 3.4533 | -0.282% | 1/8 |

The B1 32-token greedy sequence matches, but first-decode logits differ by up to
0.8125 (NRMS 0.02470); final-decode difference is 0.15625 (NRMS 0.00622). The B4
sequence diverges and throughput regresses, so that path is rejected. B1 is a
small kernel-only-derived model lead, not a validated serving improvement; the
numerical drift needs scrutiny before promotion. No serving/GSM test or
production integration was performed. Large logits/tokens tensors remain in
remote `model-m1/numerics.pt` and `model-m4/numerics.pt`, with archived hashes.

Both model processes exited successfully. Their startup used isolated caches;
future runs should share compiled-kernel caches separately from isolated tuner
records to avoid unnecessary recompilation. The initial CLI attempt used the
now-ambiguous `--cuda-graph-bs`; it exited before loading, and the measured runs
use explicit `--cuda-graph-bs-decode`.
