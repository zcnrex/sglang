# Measured decode counters

Host: chunan-b300-8. Physical GPU1, exposed as CUDA device0. Nsight Compute
2025.3.1.0. Existing pinned FlashInfer environment; no production modifications.

The NVTX range selects one warmed decode call, including its small workspace
fill kernel. Reported attention counters refer to kernel ID1 in metrics.csv.
The control.log run executes the script entirely outside the profiler;
run.log contains an additional control outside the selected range with the
profiler attached. Initialization and warmup are excluded from counter capture.

```sh
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/qvl/sglang-perf/python \
/usr/local/cuda/bin/ncu --nvtx --nvtx-include 'steady_decode/' \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum \
  --csv --log-file metrics.csv /root/qvl/venv-sgl/bin/python profile.py
```

Kernel reads: 4,296,317,440 bytes. Writes: 4,785,920 bytes. Mandatory K+V
payload: 4,294,967,296 bytes. Read excess: 0.0314%. DRAM throughput: 94.15%
of sustained peak. SM throughput: 46.48%. Achieved occupancy: 24.89%.
Profiled duration: 595.488 us. Separate unprofiled graph control: 586.609 us.
