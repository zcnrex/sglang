# Qwen3-VL-4B B300 benchmark

`scheduler_bench.py` reproduces the handoff workload: text-only random 8192-token inputs and 1024-token outputs, concurrency 1/4/8/16/32/64/128, 30/40/80/80/160/320/640 requests, and `max(64, concurrency)` warmups followed by a cache flush. `--screen` uses only c1 (3 requests, 1 warmup) and c128 (256 requests, 128 warmups). Screening numbers are provisional because the full sweep uses more requests and warmup.

Run from the source checkout with a fresh output directory. Select an idle GPU and retain `pmon.log` to check for foreign processes. Each run records the GPU state, source revision/diff, installed packages, command, server log, per-point benchmark output, and a summary. It rejects incomplete request counts. Servers terminate on completion unless `--keep-server` is set.

```bash
HF_HOME=/root/qvl/hf PYTHONPATH=$PWD/python \
  python benchmark/qwen3_vl/scheduler_bench.py \
  --gpu 0 --port 32000 --output /root/qvl/experiments/example \
  --python /root/qvl/venv-sgl/bin/python \
  --bench /root/qvl/venv-bench/bin/sgl-bench --screen -- \
  --attention-backend trtllm_mha --page-size 32 \
  --enforce-disable-flashinfer-allreduce-fusion
```

Remove `--screen` for the complete sweep. Arguments after `--` pass directly to the server, allowing explicit comparisons of `--enable-mixed-chunk`, `--chunked-prefill-size`, and `--kv-cache-dtype`. The model remains multimodal capable; the benchmark sends no images.

Reference vLLM output tokens/s from `qwen3vl4b-b300-handoff.md` (2026-10-07):

| Concurrency | vLLM output tokens/s | 10% target |
|---|---:|---:|
| 1 | 334 | 367.4 |
| 4 | 1125 | 1237.5 |
| 8 | 1774 | 1951.4 |
| 16 | 2499 | 2748.9 |
| 32 | 3084 | 3392.4 |
| 64 | 3494 | 3843.4 |
| 128 | 3815 | 4196.5 |

The reference used one B300, vLLM `0.30.1rc1.dev648+g92044241a`, and sgl-bench `a9da34ad1f997ca05878d858d3d01970e4a49af9`. FP8 KV-cache results must be identified separately from BF16 and validated for accuracy before adoption.

`screening_results.json` records the short experiments at upstream revision `43b4abd857` before the split-K tuning change. `mixed-*` uses FlashInfer BF16; `trt-*` uses TRTLLM MHA with page size 32; `fp8` denotes E4M3 KV cache. `normal` means mixed chunking is disabled. All use one B300. NGRAM used draft length 4, BFS breadth 1, max running requests 128; its c1 retry followed a port shutdown race and completed normally. Monitor logs contain only the experiments' own launcher/scheduler processes (including outgoing processes during the NGRAM restart). Full logs remain under `/root/qvl/experiments/scheduler/` on the benchmark host.

## Validated deployment configuration

The candidate uses BF16 model weights, an FP8 E4M3 KV cache, and TRTLLM generation attention (including FP8 query conversion), page size 32, a 16384-token prefill chunk, and no mixed chunking. This is a precision/configuration change in addition to the split-K kernel optimization; comparisons should retain that distinction. The original multimodal model is loaded and remains available, but these performance and GSM8K evaluations send text only; image behavior was not evaluated.

```bash
HF_HOME=/root/qvl/hf CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/python \
  /root/qvl/venv-sgl/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --attention-backend trtllm_mha --page-size 32 \
  --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 16384 \
  --enforce-disable-flashinfer-allreduce-fusion \
  --host 127.0.0.1 --port 32000
```

The candidate source changes are committed in `cd4657fd5b`; the benchmark executable fix is `a2ab66d8f8`. The remote experiment checkout remained at upstream `43b4abd857` with the corresponding patches applied, so its recorded Git HEAD is the upstream base. Saved source diffs identify the actual tested implementation. The full sweep uses exactly the documented prompt counts, warmup counts, input/output lengths, and cache flush behavior.

## Final exact sweep

All seven points exceed the handoff vLLM throughput by at least 10%. Raw unmodified benchmark JSONL files are in `results/final/`; the same FP8 configuration before split-K tuning is in `results/fp8_old_splitk/`.

| Concurrency | vLLM tokens/s | Candidate tokens/s | Speedup | TTFT ms | TPOT ms |
|---|---:|---:|---:|---:|---:|
| 1 | 334 | 378.19 | 13.23% | 71.09 | 2.57 |
| 4 | 1125 | 1323.28 | 17.63% | 172.63 | 2.85 |
| 8 | 1774 | 2193.91 | 23.67% | 352.90 | 3.30 |
| 16 | 2499 | 3229.05 | 29.21% | 524.67 | 4.44 |
| 32 | 3084 | 4216.96 | 36.74% | 997.65 | 6.60 |
| 64 | 3494 | 5046.35 | 44.43% | 2011.10 | 10.72 |
| 128 | 3815 | 5600.03 | 46.79% | 3821.71 | 19.12 |

All 1,350 measured requests completed, with exactly 11,059,200 input tokens and 1,382,400 output tokens. The pmon audit found only the owned launcher PID 3100365 and scheduler PID 3100707 on GPU 3. TTFT at high concurrency remains higher than the handoff vLLM baseline; the optimization target met here is output throughput.

Full text GSM8K validation scored 1218/1314 (92.69%) for the final configuration versus 1215/1314 (92.47%) for the BF16 reference. See `accuracy_results.json` and `eval_gsm8k.py` for protocol and evidence; this is a single-run accuracy check, not a statistical equivalence claim.
