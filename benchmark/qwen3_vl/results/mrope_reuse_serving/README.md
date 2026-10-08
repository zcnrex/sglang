# Shared MRoPE kernel serving screen

The shared-kernel implementation preserved the validated numerical behavior. This bounded serving screen measured effectively flat throughput at C16/C128, with a small decrease in the C1 smoke. It does not demonstrate a throughput improvement or the full performance target.

Both variants used the same pinned BF16 model, BF16 queries/cache, TRT attention, HND page 32 cache, mixed prefill chunks of 16,384, 20 decode graph buckets through 128, and a 1,600,000-token pool. Input/output settings were nominal 8192/1024 with random seed 42 and range ratio 1. Fresh public startup tuning was retained; no algorithm replacement hook was installed. Warmups were followed by cache flush. Existing compiled caches were shared; only the prior owned Rust compilation artifacts were reused inside otherwise fresh worker caches.

GPU 6 ran control then candidate; GPU 7 ran candidate then control. Each loaded server passed GSM sanity 19/20. All 684 measured requests completed with 5,603,328 nominal input tokens and 700,416 output tokens; raw per-request arrays and logs are retained losslessly.

| Concurrency | Requests / warmup per run | GPU 6 throughput change | GPU 7 throughput change | Paired geomean |
|---:|---:|---:|---:|---:|
|1|3 /1|−0.3441%|−0.5421%|−0.4432%|
|16|40 /16|+0.5311%|−0.3520%|+0.0886%|
|128|128 /128|−0.0755%|−0.0986%|−0.0870%|

C1 is a three-request smoke, insufficient for a stable speed conclusion. C16 pairs disagree in sign. GPU 7 startup tactics matched fully. GPU 6 selected M128 down-projection tactic 4 for control and 1 for candidate; that pair cannot isolate the shared-kernel change. Measured M128 execution was not instrumented at lower concurrency, so its absence is not assumed.

Mean TTFT (control→candidate, milliseconds) was 78.339→78.700 and 79.482→86.496 at C1; 569.713→559.960 and 570.851→566.461 at C16; 4613.413→4623.520 and 4751.229→4642.874 at C128. These are means, not medians. Full mean/median/p90/p95/p99 and TPOT fields are in `summary.json`; raw request distributions are in the archives.

The control source contains 10,245 files; candidate 10,243, with the five intended modified files and two standalone files removed. Before/after full-manifest checks passed for every worker. Model source and evaluator source revisions are distinct: sanity metrics retain the evaluator's recorded revision without relabeling it as runtime source.

All owned servers terminated through their driver cleanup, and GPUs 6/7 were verified empty. Scripts and launch arguments are retained. Join each phase's ordered `raw-*.tar.gz.part*` files to reconstruct the exact original compressed archive, then verify its SHA256 against `raw_archives.json`. Archives exclude only runtime compilation caches; public startup tuner records, all raw benchmarks, sanity results, telemetry, and server/driver logs are included. Text copies outside these archives may be whitespace-normalized by repository hooks.
