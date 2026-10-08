# BF16 activation serving screen

The activation-only candidate was tested against the frozen base on physical GPUs 6 and 7, swapping variant order on each GPU. All six server sanity checks scored 19/20. Every measured request completed with exact nominal 8192 input and 1024 output token accounting. The tokenizer adds chat framing on the server; nominal benchmark lengths are not raw native token-ID lengths.

| Concurrency | Requests / warmup | GPU 6 throughput change | GPU 7 throughput change | Geometric mean |
|---:|---:|---:|---:|---:|
| 1 | 8 / 2 | +0.687% | +0.907% | +0.797% |
| 8 | 40 / 8 | +0.087% | +0.445% | +0.266% |
| 128 | 128 / 128 | +0.230% | +0.189% | +0.209% |

These are bounded two-pair screens, not a full workload result or proof of isolated causal gains. Public startup tuning chose different M128 down-projection tactics. At concurrency 128 the control/candidate tactic pairs were 1/4 on GPU 6 and 6/4 on GPU 7, so its throughput difference cannot be attributed solely to activation. Gate/up tactic 1 and vocabulary M4/M8 tactic 2 matched throughout. Measured M128 projection usage was not separately instrumented at lower concurrency; its absence must not be inferred from client concurrency alone.

Mean TTFT regressed at concurrency 1: 73.52 to 76.10 ms and 75.02 to 80.45 ms. At concurrency 8 it improved from 364.46 to 362.36 ms and 360.13 to 358.84 ms. At concurrency 128 it was mixed: 4688.49 to 4628.70 ms and 4607.25 to 4684.09 ms. Full TTFT/TPOT distributions and raw request details are retained in each run archive; the compact summary contains all reported metrics.

The protocol retained BF16 weights/query/KV, HND page size 32, mixed chunk size 16384, twenty graph buckets through 128, and a 1.6-million-token pool. Each server used fresh normal public tuning, shared compiled artifacts, and cache flushing after warmup through the pinned benchmark client. No algorithm replacement hooks were installed. The larger-concurrency screen began only after the separate matched-startup model correctness gate passed. All owned servers exited and GPUs 6/7 were verified empty afterward; unrelated GPUs were untouched.

Each raw archive contains original unnormalized files, including source admission, launch arguments, resolved server configuration, normal startup tactic cache, sanity details, benchmark commands/results, and clock/power/temperature telemetry. Archives larger than 1.4 MB are split: concatenate numbered parts before extraction. `raw-file-hashes.json` verifies the extracted original bytes. To reproduce the summary, extract all run directories here and run `summarize.py.txt` with Python. Readable summaries may receive newline normalization; raw archives are immutable.

The candidate source was frozen before a final formatting-only blank-line removal in the activation header. `source-provenance.json` records both tested and committed hashes, and compressed full manifests preserve the exact admitted source snapshots. No normalization changes were present in either arm.
