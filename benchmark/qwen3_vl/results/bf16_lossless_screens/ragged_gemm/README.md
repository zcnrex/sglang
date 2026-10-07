# Mixed-row alignment screen

Standalone GPU1 on chunan-b300-8, pinned `/root/qvl/venv-sgl` environment.
Seed20261007; BF16 inputs/weights/outputs, default torch CUDA BLAS backend.
Actual mixed row counts from measured epoch1 traces: M16331 (26 steps),
M16330 (4 steps), and early partial M8201. Source at trace:466c9e0f40.

`bench.py` tests actual model QKV/O/gate-up/down dimensions and nearest128-row
padding. All12 leading outputs match bitwise. Padding alone helps downproj
about6%, but explicit input copying erases the gain in all configurations.

`pair.py` uses the existing runtime SiLU producer directly into the leading
slice of padded storage, zeros the extra rows each iteration, then runs the
padded downprojection. No activation copy is required. All four activation
and output comparisons match bitwise. Eight alternating graph measurements
of50 pairs each include producer, zeroing and GEMM.

| M | Raw pair us | Padded pair us |
|---:|---:|---:|
|16331|796.376|789.586|
|16330|782.849|789.975|
|8201|401.737|400.854|
|16384 aligned control|803.921|798.033|

Timing drift was substantial; raw measurements are retained. Observed gains
are below1% and comparable to the already-aligned control. This bounded screen
does not establish a useful pair improvement. No production change promoted.
A custom producer that fuses padding-zero stores was not implemented.

Run with `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/qvl/sglang-perf/python
/root/qvl/venv-sgl/bin/python pair.py` (one shell line). Scripts write reports
to `/root/qvl/experiments/ragged-gemm/`.
