# Real-model cast-only gate

Both variants load frozen `/root/qvl/sglang-m16-down-production`. All 4,229 files in its expected source manifest match; the recursive manifest additionally includes two non-runtime `.claude/skills` scripts omitted by the earlier manifest. No production source was changed. PIDs 548262/548327 on GPUs 0/1 completed normally.

The external hook replaces only contiguous BF16 logits of exact shape `[64 or 128,151936]`, with no softcap. Matching FP32 buffers use startup-compiled `output.copy_(input)` and assert the returned object is the original caller buffer. The exact-shape no-buffer branch uses startup-compiled `.float()`. Both callables warm before candidate capture. Actual target capture used the buffer-preserving branch: one capture plus two eager warm calls, all three identity assertions passed. Unsupported shapes and softcap fall through. No vocabulary GEMM or sampling override occurs.

The real model uses 128 input tokens and 16 generated tokens per request. Prefill, first decode and last decode full logits are bitwise identical; all tokens agree. A changed input seed also produces bitwise final logits and identical tokens. These finite short-context checks do not establish full accuracy or long-context serving behavior.

| Batch | Baseline graph ms | Candidate graph ms | Median paired saving us |
| --- | ---: | ---: | ---: |
| 64 | 2.817595 | 2.804698 | 13.144 |
| 128 | 3.280603 | 3.249190 | 31.310 |

Six opposite-order paired rounds each replay 256 retained model graphs after common warmup. Both variants share identical populated input/KV-position state. All recorded runner-buffer snapshots remain unchanged, including the complete FP32 logits buffer. This is fixed-state graph timing, not throughput from 256 successive generated tokens. Separate actual 15-step timings are retained in the reports and are not substituted for the paired estimate.

One untimed warmed trace attributes the cast kernel at B64 from 24.256 to 13.248 us and B128 from 49.057 to 14.720 us. Profiler overhead and different cache conditions mean these single traces need not equal the repeated graph delta. Baseline kernel is Torch direct-copy with conversion; candidate is `triton_poi_fused_copy_copy__0`. Generated Inductor wrapper/kernel sources and hashes are archived in `compiled/`, with static element counts 9,723,904 and 19,447,808. Selected generated configurations are B64 XBLOCK512/8 warps/1 stage and B128 XBLOCK1024/4 warps/1 stage, retained as best_config JSON. Compilation config is fullgraph true/dynamic false, no global Tensor.float mutation. Clocks were not locked; post-run telemetry is included without claiming continuous clock stability.

Large `numerics.pt` tensors remain in the remote run directories; compact exact comparisons are in reports. Traces are losslessly compressed. No serving result is established here.
