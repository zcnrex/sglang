# Read-only phase drift audit

Measurement windows use the first logged prefill after the successful cache flush plus benchmark duration. Logs have one-second timestamps; five seconds at each edge are excluded for telemetry statistics. All post-flush prefill rows are counted. Raw 1 Hz telemetry contains clocks, power and temperature; it does not contain GPU memory allocation, CPU utilization, background process inventory or a kernel trace.

| Phase/GPU | Median SM MHz | Memory MHz | Median power W | Median temperature C | Samples below 1000 W |
| --- | ---: | ---: | ---: | ---: | ---: |
| A/0 | 2017 | 3996 | 1092.87 | 62 | 0/161 |
| A/1 | 1995 | 3996 | 1090.22 | 63 | 0/161 |
| B/0 | 2017 | 3996 | 1092.90 | 61 | 4/163 |
| B/1 | 2002 | 3996 | 1090.62 | 62 | 4/163 |

Memory clocks remain 3996 MHz in every sampled measurement window. SM minima/maxima overlap strongly, and temperatures are slightly lower in phase B. There is no evidence here of a sustained clock reduction explaining the common slowdown. Four lower-power samples occur in each phase-B arm, but power can respond to work mix or idle gaps; no CPU timeline exists to identify their cause or critical-path cost.

Each run has 121 logged decode batches, all marked CUDA graph true. Startup and final public tactic records are identical within every worker, and common selected tactics agree across workers. Candidate markers confirm actual production cast capture at 64/128 rows with caller identity, plus measured-epoch B128 decode. No missing dispatch or cache-policy failure is visible.

Post-flush prefill log counts are 329/330 in phase A and 331/331 in phase B. All four runs have the same 5,212,745 logged new-token total. The new-sequence field sums 949/950/949/949 because chunk continuations can be counted again; these are not unique-request totals. Median running requests at prefill are 121/121/121/122. Grouping differs slightly, but this coarse log cannot quantify attention/GEMM costs or establish that grouping caused the roughly 1.1% phase change.

Conclusion: no concrete protocol/source/tactic failure is identified, and no supported causal explanation rescues a cast gain from this full-workload result. Retain it as inconclusive and do not repeat merely to seek a positive outcome.
