# Next prefill admission audit

**No new scheduling change is recommended from the retained evidence.** The current logs show different prefill grouping and TTFT distributions, but do not identify a removable admission delay or a defensible per-request scheduling rule. This audit ran no GPU jobs and changed no production source.

## What the current evidence establishes

The integrated C8 results have all 80 per-request TTFTs and inter-token latency arrays. They do not contain absolute request submission, scheduler admission, first execution, or first-token timestamps, and the batch logs have no matching request IDs. Therefore, a request's TTFT cannot be joined to its actual scheduler batches. Request-order groups of eight are not known admission batches.

After each measured cache flush:

| Integrated C8 run | Prefill records | One-request records | Queue-zero records | Records with ≤512 new tokens | Mean / median TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU2 control | 51 | 13 | 30 | 1 | 356.056 / 345.155 |
| GPU2 candidate | 54 | 20 | 39 | 4 | 355.397 / 379.222 |
| GPU3 control | 53 | 17 | 36 | 3 | 353.458 / 370.675 |
| GPU3 candidate | 59 | 25 | 44 | 5 | 339.366 / 366.009 |

These counts support changed grouping, not a causal explanation. A queue-zero report is a post-admission observation, not proof that no requests arrived during the forward. A prefill record is not a unique request; continuations are counted again. Mean/median divergence does not show a uniform TTFT regression.

The current mixed on/off C8/C16 comparison also has retained per-request arrays. C8 has 50–56 prefill records per run; C16 has 45–46. Mixed lowers mean TTFT in every same-GPU pair, while C8 median and tail changes are mixed. Existing results already establish the throughput/TPOT tradeoff; repeating that switch is not a new mechanism.

The earlier C128 forward-event profile attributes 72.36% of forward time to decode and 27.55% to mixed steps. Forward intervals sum to 68.750 s versus 69.049 s benchmark wall time. The 0.43% difference is not a rigorous CPU-idle estimate because host work overlaps GPU execution, but it does not reveal a large unexploited idle period for an admission-only optimization. Later mixed kernels include substantial attention and GEMM work that queue reordering cannot remove by itself.

## Source checks

Inspected current source is hashed in `source_provenance.json`; line references refer to that snapshot.

- `scheduler.py:3764` chooses a ready prefill batch before decode. `_get_new_batch_prefill_raw` handles explicit delay policies at 3819–3888; recorded runs have FCFS, prefill interval 0, prefill delayer disabled and no minimum-free-slot delay.
- `scheduler.py:3939–3951` admits the existing chunked continuation before iterating fresh waiting requests. A proposal to prioritize finishing the current chunk is already implemented.
- `scheduler.py:3965–4070` iterates the waiting queue and stops on admission exhaustion. Resource and slot checks must remain valid; bypassing them is not a latency optimization.
- `schedule_policy.py:656–674` subtracts mixed decode tokens from the compute budgets. `schedule_policy.py:1534–1587` rounds/truncates chunk admission to pages under the current path.
- `schedule_policy.py:90–98` restricts exact-token chunk filling to gfx95. Simply enabling its environment flag does not change B300 behavior. Extending it would alter the already-investigated page/chunk admission boundary, not expose a demonstrated new delay.
- Crucially, `schedule_policy.py:1176–1190` calls `_update_prefill_budget` with prefix length zero for a continuing chunk, while 937–938 adds that argument to logged hit tokens. Consequently, a small batch with `#cached-token: 0` is **not evidence of a fresh prefix-zero request**. The integrated C8 logs cannot establish how often a proposed minimum-fresh-chunk threshold would be eligible.

## History and rejected directions

The root experiment index and relevant result notes were searched for admission, delay, chunk, interval, queue, and resource-cap experiments before considering a recommendation.

- `bf16_scheduler_results.json` and the root README already cover chunk sizes, prefill/decode intervals and context limits; no rescreen is proposed.
- `bf16_prefill_fa4_audit/max_running96_screen` reduced C128 throughput by approximately 8.1% without improving TTFT. Lowering the running cap is not proposed.
- `chunk_boundary_screen` already rejected 16640 versus 16384. A small overshoot/page-boundary adjustment would revisit that same mechanism.
- That boundary observer counted only one fresh prefix-zero ≤512-token chunk per control run (64 tokens), versus three per enlarged-budget run. It explicitly left a minimum-new-chunk threshold untested because immediate eligibility was tiny. Current aggregate logs cannot establish greater fresh-chunk eligibility, so they do not justify advancing it now.
- `production_lower_regression/c64_latency_audit.md` already identified refill/admission phase relationships as an unresolved possibility. It found no request-level timestamps to distinguish admission from client arrival or context execution. The newer retained TTFT arrays improve distribution analysis but still do not close that attribution gap.

Artificial coalescing waits would intentionally delay an already eligible request and might shift latency to other requests. Skipping a partially fitting new request would leave part of the current chunk budget unused, effectively reducing work per prefill; that is outside the requested non-shrinking-chunk direction. Reordering FCFS arrivals risks fairness/starvation and changes which request benefits, with no measured eligibility criterion here. None is recommended.

## Reproducibility and limits

`extract.py.txt` reads the twelve successful saved benchmark/log pairs, uses the last cache-flush marker, recomputes mean/median/p90/p95/p99 from request TTFTs, and records batch/queue histograms and resolved flags. `extraction.json` includes input file hashes. Run it from the repository root with Python; it uses no network or GPU.

The smallest missing attribution is a correlated request timeline linking receive/admission/first-forward/first-token events and per-forward GPU spans. This is an evidence gap, not an optimization proposal or a request to start another run. No latency improvement is predicted, and no additional experiment was launched.
