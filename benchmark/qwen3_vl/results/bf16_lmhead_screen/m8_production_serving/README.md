# Production M8 LM-head C8 crossover

The committed M8 path retains a small positive serving signal: **+0.3347% / +0.3021%** same-GPU throughput, geometric mean approximately **+0.3184%**. These two short pairs do not establish a precise effect size or the overall target. TTFT and TPOT changes are mixed.

| GPU | Control / candidate tok/s | Throughput | TTFT | TPOT |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1887.925 / 1894.244 | +0.3347% | −11.4118% | +0.1990% |
| 3 | 1901.712 / 1907.457 | +0.3021% | +3.9513% | −0.6887% |

Control is frozen M4 production `/root/qvl/sglang-lmhead-production`; candidate `/root/qvl/sglang-lmhead-m8-production` is exactly the two-file overlay committed as5babd15f4f. Complete4229-file manifests are verified before each run. No external algorithm replacement or private tuning policy is used; observer records admission, public cache, first dispatch and replay only. Both arms retain BF16 weights/query/KV, TRT HND page32, mixed chunk16384 and disabled prefill graphs.

C8/N80, warm64, flush before measurement, nominal8192/1024 lengths. All four runs completed80 requests,655360 input tokens and81920 output tokens. Two phases swap variants across GPUs2–3. Public M128 gate/up tactic1, M128 down tactic4 and M4 LM-head tactic2 match in every run; candidates independently choose M8 tactic2. Startup READY and actual M8 capture/graph8 replay proof are retained. Measured post-flush prefill row intervals exclude128 in all runs as a supplementary audit, not required to discount a tactic difference here because M128 choices match.

Driver496611 completed both phases and cleaned owned servers496616/496617 and500550/500549. All handles were absent before starting separate accuracy work. Raw archive retains scripts, logs, results, expected manifests and cache JSON; compiled cache binaries are excluded. The paired model numerical gate is recorded separately in ../m8_production_model. No accuracy-equivalence claim follows from this performance screen.
