# Chunk boundary 16384 versus 16640: rejected short screen

Increasing the mixed-chunk budget to 16640 did not improve throughput. Both arms use the exact frozen `/root/qvl/sglang-m16-down-production` source, BF16 weights/Q/KV, HND page 32, TRT attention, graph maximum 128 and server-verified KV capacity 1,600,000. No production files changed.

| GPU | Control 16384 tok/s | Candidate 16640 tok/s | Change | Median TTFT change | Median TPOT change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 3823.983 | 3797.273 | -0.6985% | +0.0677% | +0.6254% |
| 3 | 3823.228 | 3816.847 | -0.1669% | +1.1627% | +0.0012% |

Each run completed exactly 128 requests, 1,048,576 nominal input tokens and 131,072 output tokens, concurrency 128, nominal 8192/1024 lengths. This is a short N128 screen with warmup 64 and cache flush, not the standard C128 warmup 128 protocol. Normal public startup tuning was retained. GPU2 naturally matched M128 gate/up 1, down 1, M4/M8 vocabulary 2. GPU3 differed: candidate gate/up 2/down 1 versus control gate/up 1/down 4; its small regression does not isolate the budget. Actual rows 128 DECODE and graph 128 replay were observed in every run. No measured rows 128 MIXED/EXTEND were observed.

## Rationale and observed admissions

Frozen scheduler subtracts mixed decode tokens from the chunk budget. Page rounding normally charges 8224 for an 8200-token prompt, so two such prompts plus 128 tails fit 16576, below 16640 but above 16384. `max_prefill_tokens` remains 16384; with chunking it checks exhaustion after admission, rather than clamping a second full request while its input budget remains positive. No 16640/16528/16896/16400 configuration was found in prior inspected metadata, launch or resolved-server JSONs.

Observed retokenized lengths vary, so the ideal two-identical-prompts premise is incomplete. For example control admitted extend lengths [8200,8126,1] at rows 16327; candidate admitted [8200,8126,256,1] at rows 16583. The extra budget often starts a small third chunk. The candidate produced 64/65 prefill-containing forwards versus 65/65 control; continuation counts 59/60 versus 62/62. Full CPU extend/prefix lists are in observer JSONL by cache-flush epoch; epoch 1 is measured, epoch 0 is probe/warmup.

## Tiny-chunk follow-up audit

New prefix-zero chunks of at most 512 tokens occur only once per control run (64 tokens), and three times per candidate run (256,416,480 tokens). They account for 64/1152 input tokens, respectively, or 0.006%/0.110% of nominal measured input. This is low immediate eligibility for a proposed minimum-new-chunk admission threshold. Changing admission can cascade into later grouping, so these counts neither prove a win nor precisely bound it. No threshold experiment was run.

Continuation prefix-token sums fall from 402,016 in each control run to 192,608/197,152 candidate, but these are geometry counters, not saved attention FLOPs. Causal chunking partitions necessary attention work; prefix lengths must not be presented as redundant QK computation. Small initial chunks can create extra launch/metadata work and alter optimized context routing, but this screen contains no timed attribution proving a worthwhile removable cost.

## Evidence

`analysis.json` contains rates, geometry counts and complete public tactic records. The raw archive preserves exact manifests, commands, logs, startup caches and benchmark JSONL. Readable driver/observer copies are `.py.txt`; observers record shapes and tuning state without replacing algorithms. Remote root: `/root/qvl/experiments/chunk-boundary`. Driver 606098, phase A 606103/606104 and phase B 611385/611384 are terminal; GPUs 2/3 released. No full follow-up or production change is warranted from this negative screen.
