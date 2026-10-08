# M8 LM-head C8 serving crossover

Both same-GPU pairs improve throughput slightly; the geometric mean is approximately **+0.3315%**. This is two short pairs, not a precise effect estimate or achievement of the overall target. TTFT is mixed.

| GPU | Control / candidate tok/s | Throughput change | TTFT change | TPOT change |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1885.032 / 1890.392 | +0.2843% | −1.9345% | −0.5802% |
| 3 | 1904.981 / 1912.195 | +0.3787% | +6.7344% | −0.7494% |

All four runs completed exactly 80 requests, 655360 input tokens and 81920 output tokens. Protocol: C8/N80, 64 warmup requests, flush before measurement, nominal 8192 input and 1024 output tokens. Two phases swap variants across GPUs 2–3. Both arms use `/root/qvl/sglang-lmhead-production`, including production M4 LM-head dispatch, BF16 weights/query/KV, TRT HND page32, mixed chunk16384 and disabled prefill graphs. Every one of the 4229 Python source hashes matches the local production tree; only candidate external exact-M8 hook differs.

Candidate public startup tuning independently selected M8 Lt tactic2 in both runs. Logs prove M8 graph capture and actual graph8 replay. M4 tactic2 and M128 gate/up tactic1 match across all runs. M128 down differs on GPU3: candidate4, control1; GPU2 uses1 in both. No hardcoded tactic or private tuning policy was used.

The M128 observer records startup/capture execution. Its first-occurrence markers alone do not exclude later execution. A separate audit of every logged prefill after the single cache-flush boundary checks conservative row intervals [new-token, new-token + running-requests]; none contains128 in any measured run. Prefill graphs are disabled and TP1 has no distributed row padding; decode batches are at most8. Thus logged measured geometry excludes the differing M128 down tactic. Exact intervals and manifest checks are in analysis.json; public tactic records are in tactics.json. The existing prefill logging provenance was established by the preceding production M4 audit.

Driver483200 completed both phases and cleaned owned servers483205/483206 and486459/486460. All handles were absent after PHASE B DONE. Raw archive retains driver/hook scripts, logs, JSON results, source manifests and cache JSON; compiled cache binaries are excluded. No production code or full accuracy run was part of this experiment. Prior M8 numerical and fixed-state graph evidence is in ../m8_gpu_event.
