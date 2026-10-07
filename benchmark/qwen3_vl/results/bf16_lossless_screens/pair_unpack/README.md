# Pairwise exact BF16 reconstruction

GPU1, standalone external prototype. Existing format unchanged: sign/mantissa
bytes, exponent nibbles, per-row base/flag, exact raw fallback. Pairwise reader
loads two sign/mantissa bytes and one exponent byte, reconstructs two BF16
values in a uint32 with PRMT, then joins the halves for MMA consumption.
All65536 BF16 patterns, random bits, normal values and six real layer0/17/35
K/V captures reconstruct bitwise. Zero and nonfinite patterns are included.

Same attention compilation: BN128,8warps,2stages,8splits; one-request launch
for resource inspection. No spills in any case. Static PTX counts are not
runtime instruction counts.

| Reader | Registers | Shared bytes | Byte global loads | PRMT | Shared loads |
|---|---:|---:|---:|---:|---:|
|Raw|80|73728|0|0|6|
|Prior packed|146|139264|128|128|78|
|Pair packed|145|155648|0|128|52|

Byte loads disappear, but register demand barely changes and shared memory
increases16KiB. Final reconstruction-to-MMA layout remains costly. Rejected at
resource gate; no full-size attention timing or production integration.
Raw PTX retained remotely under `/root/qvl/experiments/pair-unpack/`.
Scripts use existing experimental packer and prior decode modules from the
lossless-kv/lossless-attention remote experiment directories.
