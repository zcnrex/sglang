# Production M32 vocabulary C32 crossover

The committed M32 extension shows only a tiny positive serving difference: **+0.0118% / +0.1017%** across two same-GPU pairs (geometric mean approximately **+0.0567%**). This does not establish a robust serving improvement. No accuracy run or PR promotion followed this screen.

| GPU | Control / candidate tok/s | Throughput | TTFT | TPOT |
| --- | ---: | ---: | ---: | ---: |
| 2 | 3149.637 / 3150.008 | +0.0118% | −2.2917% | +0.3937% |
| 3 | 3150.919 / 3154.122 | +0.1017% | +14.7906% | −1.3504% |

Control is frozen `/root/qvl/sglang-m16-down-production`, including validated M4/M8 vocabulary and M16 down projection. Candidate `/root/qvl/sglang-lmhead-m32-production` has exactly the two-file M32 extension d0d528db4f. All 4229 local Python hashes match that overlay; complete remote manifests also retain two identical inherited non-runtime skill scripts, documented in audit.json. The observer changes no algorithm or tuning policy. Candidate normal public startup READY, M32 optimized capture and actual graph32 replay are recorded.

This protocol deliberately caps decode graphs at **32** and pins KV capacity to **1,600,000 tokens in both arms**; every server-reported capacity is asserted. This differs from the earlier external-hook/default-graph screen, so absolute rates should not be compared as identical protocols. All runs use BF16 weights/query/KV, TRT HND page32, mixed chunk16384 and disabled prefill graphs. C32/N160, warm64, flush before measurement, nominal8192/1024 lengths. Each run completed exactly160 requests,1,310,720 input tokens and163,840 output tokens.

Common startup tactics match within each same-GPU pair: gate/up1, M4/M8 vocabulary2; M128 down1 on GPU2 and4 on GPU3. Candidates independently select M32 vocabulary2. Per-run cache and dispatch evidence is preserved in analysis.json. No private matched-tactic policy was needed or used.

Driver539161 completed both swapped phases and cleaned servers539168/539169 and543139/543140. All handles were absent after PHASE B DONE; GPUs2–3 are released. Raw archive preserves scripts, logs, source manifests, public cache JSON and complete benchmark results; compiled cache binaries are excluded. Two-seed short production model correctness is in ../m32_production_model.

The raw gzip archive is stored in numbered parts to meet the repository file-size limit. Concatenate `raw_evidence.tar.gz.part00` and `raw_evidence.tar.gz.part01` in that order to recover the original archive byte-for-byte; `raw_archive_parts.json` records sizes and SHA256 hashes.
