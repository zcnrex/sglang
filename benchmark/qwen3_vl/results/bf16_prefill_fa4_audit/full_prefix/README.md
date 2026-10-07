# Full-prefix packing standalone screen

Audited 5f60fb8b67 Python source (4229 hashes), GPUs 4/5, BF16 throughout.
The three prefix/query tuples come from the measured prefix observer. Queries
retain stride 6144; cache remains native HND page32. One fused copy assembles
cached prefixes and fresh K/V into contiguous unpaged BF16 buffers. Separate
cumulative Q/K lengths preserve causal alignment. Original cache is unchanged.

Packed K/V matched bitwise; FA4 versus TRT maximum absolute error was
0.00390625–0.0078125, NRMS approximately 0.0022, within atol=rtol=0.02.
The separate correctness-only proof checks actual selected CLC cache keys.
Constructor telemetry is empty because the compiled cache was already warm.

Across both GPUs and three shapes, median cold context time including packing
was 779–817 us versus TRT 899–924 us. Packing alone cost 37–39 us, using
83–99 MB scratch and 166–199 MB read/write traffic. Decode tails are excluded
from both paths. Four rounds alternate execution order; cold measurements
flush 256 MiB outside timing. GPU4 shows substantial first-round clock drift;
retain raw paired rounds rather than treating median speedups as end-to-end
claims. Data is synthetic; model validation is separate.

Original artifact bytes are preserved in `original-artifacts.tar.gz`;
`hashes.json` verifies its extracted files. Uncompressed copies are provided
for reading and may have whitespace normalized by repository hooks. No production source or serving result is included here.
