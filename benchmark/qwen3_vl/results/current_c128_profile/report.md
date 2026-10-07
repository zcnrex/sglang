# Current c128 workload profile

This is diagnostic profiling of commit `466c9e0f4073bcad6e4403c6f54cfc8c1821b867`, not a throughput claim. The clean detached remote worktree is `/root/qvl/sglang-current-profile`. The pinned Qwen3-VL-4B-Instruct revision is `ebb281ec70b05090aa6165b016eac8ec08e71b17`. GPU0 was otherwise idle. No production files were edited.

The server uses BF16 model computation and explicit BF16 KV cache, TRTLLM attention, HND cache, page 32, mixed chunking with 16384 prefill tokens, and disabled prefill graphs. No quantization or speculative decoding is enabled. Exact commands/environment and resolved server arguments are in `metadata.json`, `events-command.json`, and `server_info.json`. Kernel names independently identify BF16 attention inputs/outputs.

## Full workload accounting

The run used c128, 256 requests, 128 warmup requests, nominal 8192 input and 1024 output tokens, and a cache flush after warmup. All 256 requests completed; the benchmark reports 2,097,152 input tokens and 262,144 output tokens. Its input count is nominal text length before chat templating/retokenization; the first actual model input has 8200 tokens. No requests were changed for profiling.

The external `hooks/sitecustomize.py` wraps `ModelRunner.forward` with CUDA events and writes elapsed intervals asynchronously after event completion. Successful cache flush starts a new epoch; epoch1 is the measured workload, epoch2 is the separate trace run. Warmup is excluded. No per-forward synchronization is inserted. Sampling and scheduling outside `ModelRunner.forward` are excluded from event intervals; host launch gaps inside a forward are included.

| Component | GPU-event time | Share of forward intervals | Steps |
| --- | ---: | ---: | ---: |
| DECODE | 49.749 s | 72.36% | 1985 |
| MIXED | 18.940 s | 27.55% | 131 |
| EXTEND | 0.061 s | 0.09% | 1 |
| Total forward | 68.750 s | 100% | 2117 |

Benchmark wall time is 69.049 s. The 0.299 s difference is only 0.43% of wall time, but is not a rigorous attribution of CPU time: host work can overlap GPU execution. These measurements do not suggest a large idle/sampling/scheduling budget outside forward. Most decode steps are full batch 128 (1918 of 1985). Most frequent mixed shape is M16331/B126 (26 of 131 steps).

The diagnostic rate of 3796.5 output tokens/s would require a 9.53% wall-time reduction to reach 4196.5. If decode stayed fixed, prefill-containing forwards would need a 34.64% reduction. This estimate uses the measured diagnostic workload, not an assertion that the profiler is overhead-free.

## Representative kernel traces

Initial-admission and later mixed traces are separate from the full event-only workload. Each captures five prefill-containing and five decode steps. The later capture is armed after 600 decode steps, and its mixed spans have batch sizes 125, 125, 124, 125, 125; decode spans are all 128. Raw trace paths and hashes are in `trace-summary.json`; compressed traces stay on the remote host.

| Kernel family | Initial prefill GPU share | Later MIXED GPU share | Later DECODE GPU share |
| --- | ---: | ---: | ---: |
| BF16 GEMMs | 58.6% | 49.3% | 8.6% including split-K reduction |
| Context attention | 25.2% | 21.1% | — |
| Decode attention | below 1% | 16.5% | 87.9% |
| SiLU and multiply | 5.7% | 4.8% | below 1% |
| Fused residual RMSNorm | 3.5% | 2.9% | 1.2% |
| QK RMSNorm | 3.0% | 2.6% | included in fused prep |
| MRoPE | 1.8% | 1.5% | included in fused prep |

Mixed forwards retain substantial decode-attention work: they are not all removable prefill cost. Large-M GEMMs are the largest remaining prefill component. The measured ragged shapes were passed to the separate GEMM-padding investigation. The later decode remainder consists mainly of GEMMs plus split-K reduction (8.6%), residual RMSNorm (1.2%), SiLU (0.94%), and fused QK norm/MRoPE/cache write (0.81%). This does not expose an untested single large non-attention kernel; prior GEMM and fusion screens already cover these paths. No new optimization is claimed by this report.

Both `triage.txt` files preserve the skill's three automated tables: kernel, overlap, and fusion. **Their automated recommendations are not validated conclusions:** nvjet GEMMs here are BF16, so the FP8 recommendation is a false positive and incompatible with the task. The reported RoPE/cache fusion share incorrectly includes entire attention kernels (28.9%/86.8% initially and 40.1% later mixed); those shares are not removable preparation cost. Fused residual RMSNorm is also incorrectly categorized as GEMM. It is already fused. Single-trace overlap attribution is conservative, and no overlap candidate cleared the analyzer's reporting threshold.

## Reproduction and limitations

`run.py` launches the exact source, captures the event workload and initial-admission traces, and stops its own server. `late_mixed/run.py` plus its hook captures the second-wave trace in a bounded N256 run with no separate client warmup; this trace is for attribution only. `summarize.py` aggregates the primary event records. CUDA-event records, benchmark outputs, resolved metadata, scripts, and raw automated tables are retained here. The remote evidence root is `/root/qvl/experiments/current-c128-profile`.

All owned servers were stopped after completion. The launcher uses SIGTERM, which SGLang logs as child cleanup/SIGQUIT during shutdown; requests had already completed before this intentional shutdown.
