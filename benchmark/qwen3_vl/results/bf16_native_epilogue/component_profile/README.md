# Actual-model component diagnostic

The fused GPU kernel does save time on actual model data. Summed gate/up plus activation time falls by 2.613 ms on GPU4 and 5.517 ms on GPU5 across the 36 layers. Variation in unchanged work masks those savings in some forwards. This diagnostic supports a bounded paired serving screen after arbitrary eligible row counts are validated; it does not establish serving performance or identify another useful epilogue rewrite.

| Component, GPU time per profiled forward | GPU4 control | GPU4 candidate | GPU5 control | GPU5 candidate |
| --- | ---: | ---: | ---: | ---: |
| Gate/up + activation, ms | 41.829 | 39.216 | 42.793 | 37.277 |
| Down projection, ms | 18.414 | 20.045 | 18.672 | 18.862 |
| Other GEMMs, ms | 19.386 | 21.272 | 19.937 | 19.789 |
| Context FA4, ms | 26.244 | 28.800 | 27.067 | 26.739 |
| Decode-tail TRT attention, ms | 8.023 | 8.155 | 8.068 | 8.053 |
| All kernel duration sum, ms | 127.170 | 131.830 | 130.183 | 124.126 |

Each cell averages two traces, not a performance benchmark. GPU4 candidate saves 2.613 ms in the targeted work while unchanged components collectively grow about 7.27 ms. GPU5 unchanged components are nearly flat. All traces map 72 control gate/up+activation kernels versus 36 fused candidate kernels, with 36 normal down projections, 36 FA4 calls and 36 decode-tail TRT calls. No candidate setup, compilation or input DLPack conversion occurs in captured forwards. Warmup logits and greedy outputs match bitwise for the tested synthetic requests.

## CPU launch gaps and clocks

The median interval from host CUDA launch completion to GPU start is 25.3–35.2 ms across traces: most transformer kernels are queued well ahead. CPU gate/up scopes total approximately 6.2–6.8 ms for control and 2.7–3.1 ms for candidate, but these profiler-instrumented host savings cannot be added directly to critical-path time. The union of GPU-kernel intervals leaves 5.67–6.52 ms of gaps across the full extend operation; only 0.48–0.77 ms lies between the first transformer GEMM and last down projection. Most visible gaps precede transformer work, around request/position/cache metadata preparation. Gaps omit memcpy/memset intervals and are upper bounds on GPU idleness; sums across kernels can include overlap.

Contemporaneous 100-ms telemetry reports approximately 1260–2032 MHz SM clocks and 1054–1089 W power during these windows. There are only one or two samples inside each ~130-ms forward. Candidate GPU4 windows sampled lower clocks than its first control window, while GPU5's first candidate sampled 2032 MHz. This is consistent with power/clock variation affecting unrelated components, but sparse sampling cannot establish per-kernel clock causality or normalize durations reliably. The raw trace timestamps, converted epoch windows and samples are in components.json and telemetry.csv. GPU memory clocks remain at the existing setting; no hardware policy was modified.

## Capture and correlation

Same frozen current source, direct-Torch FFI candidate and exact B42/M16331 geometry as ../ffi_model. Both variants keep BF16 weights/queries/KV and current packed FA4. Four common warmup variants precede four captured forwards per GPU, ABBA on GPU4 and BAAB on GPU5. Each capture rebuilds the exact request/cache state and actual one-token tails, avoiding replay of a mutated prepared batch. Only the target extend forward is profiled; three decode steps afterward retain the prior numerical protocol. Profiles include CPU/CUDA events without stack/shape/memory collection. They are diagnostic and carry profiler overhead.

qvl_gateup_activation and qvl_down host scopes are joined to CUDA runtime/driver launch events by timestamp and thread, then to GPU kernels by CUPTI correlation ID. Attention and remaining families use exact kernel names. analyze_components.py reproduces compact tables from this directory (`python3 analyze_components.py .`). Full per-trace component counts, CPU scope totals, GPU sums/spans, gap lists and telemetry windows are retained. The two raw automated triage outputs provide the skill's three tables; their suggestions are not authoritative: nvjet here is BF16, not FP8; fused RMSNorm is misclassified as GEMM; the claimed QK/RoPE fusion share incorrectly includes the full decode attention kernel. Those rows do not represent removable work.

Remote root /root/qvl/experiments/native-epilogue-component-model preserves all raw traces. Their SHA256 manifest is included. The complete compressed raw evidence is only about 1.1 MB, so traces are also included locally. original-artifacts.tar.gz preserves pre-format bytes. Workers 415885/415886 completed, telemetry 415883 was stopped and GPUs4/5 released. No production edits or serving/GSM runs were performed.
