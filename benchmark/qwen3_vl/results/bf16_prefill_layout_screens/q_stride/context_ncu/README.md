# TRT context hardware counters

Nsight Compute2025.3.1.0, physicalGPU1 on chunan-b300-8. One warmed NVTX
`steady_context/` range, BF16 Q32/KV8/D128, HND page32, Q token stride6144.
Synthetic compatible Q/KV lengths8200/8007; individual production request
lengths were unavailable. See parent README. No production/system changes.

Attention kernel ID1: Q256/KV128 PersistentContext. ID0 is workspace fill.

| Counter | Value |
|---|---:|
|DRAM read bytes|201518336|
|DRAM write bytes|97402112|
|DRAM throughput percent sustained peak|3.42|
|SM throughput percent sustained peak|81.69|
|Tensor-pipe active percent sustained peak|78.34|
|Achieved occupancy percent|24.94|
|Profiled kernel duration us|1138.944|
|Separate unprofiled graph control us|784.780|

This invocation is dominated by compute/tensor activity rather than HBM
bandwidth. Profiler replay and clock conditions affect duration; the profiled
time is not substituted for the clean control or earlier paired measurements.

```sh
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/ncu --nvtx \
 --nvtx-include 'steady_context/' \
 --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum \
 --csv --log-file context_metrics.csv \
 /root/qvl/venv-sgl/bin/python context_profile.py
```

The clean control executes the same script in a fresh process without NCU.

## Comparison with the earlier paired screen

The clean control has identical geometry, seed, tensor creation order, Q stride,
KV layout, attention arguments and CUDA-event conversion to the first case in
`../bench.py`. Both capture five calls per graph and time ten graph replays
(50 calls). There is no identified shape or timing-unit mismatch.

The measurement histories differ: this control warms ten eager calls and five
graph replays, then records one sample. The earlier paired screen alternates
three paths for eight rounds, with three warm replays before each sample. Its
strided samples were768.49,910.13,945.70,934.17,926.72,939.15,932.66,938.91us;
933.42us is their median. The fresh784.78us control resembles the first earlier
sample, not its later sustained samples. This establishes timing-history drift,
but does not establish its hardware cause. No contemporaneous GPU clock/power
log was collected. Thermal/power/clock explanations remain unverified. Compare
paired variants within the same screen; do not interpret the fresh control as
a19% optimization. The profiler-attached pre-range control was804.48us, while
counter replay reported1138.94us; these are distinct measurements.
