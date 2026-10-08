# Integrated B1/B2 production serving crossovers

Frozen M4 production control versus immutable /root/qvl/sglang-qkv-small-production, exact three-file extension. Public dispatch only, no algorithm replacement. C1/N30 on GPUs2/3 and C2/N40 on GPUs4/5, each two phases with opposite assignment. Both candidate pairs improve throughput: C1 +4.897226%/+4.813891%, geometric +4.855550%; C2 +4.143425%/+4.226621%, geometric +4.185015%. C1 exceeds historical target367.4 tok/s; C2 has no historical target and no goal claim.

Warm64 then flush, input8192/output1024/range1/clientseed42; serverseed0, BF16 model/query/KV/output, TRT HND32, mixed chunk16384, explicit20 graph buckets capped128, pool1,600,000. Identical reported buckets/capacity asserted. Every C1 run completed30 requests/245760 input/30720 output; C2 completed40/327680/40960. Full per-request arrays retained using --output-details.

| C | Phase/GPU | Arm | tok/s | TTFT mean ms | median ms | p95 ms | p99 ms | TPOT median ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | A/2 | control | 391.500786 | 80.349631 | 78.300721 | 90.613859 | 93.275041 | 2.475450 |
| 1 | A/3 | candidate | 413.342609 | 80.446019 | 80.090874 | 86.312299 | 88.794159 | 2.340983 |
| 1 | B/2 | candidate | 410.673465 | 82.024265 | 82.673143 | 87.974653 | 88.067455 | 2.354657 |
| 1 | B/3 | control | 394.358617 | 80.375489 | 82.048302 | 89.634760 | 91.200799 | 2.456982 |
| 2 | A/4 | control | 716.629557 | 117.982808 | 119.672666 | 136.583934 | 147.804768 | 2.675032 |
| 2 | A/5 | candidate | 747.122662 | 121.342733 | 121.543680 | 143.744442 | 146.938959 | 2.562370 |
| 2 | B/4 | candidate | 746.322568 | 120.298227 | 119.226358 | 144.554087 | 147.562746 | 2.563225 |
| 2 | B/5 | control | 716.825181 | 120.246472 | 120.876195 | 143.961536 | 147.976796 | 2.675923 |

C1 TTFT mean rises1.675ms/0.071ms; median rises4.372ms on GPU2 and falls1.957ms on GPU3, with p95/p99 improved both. C2 mean rises2.315ms/1.096ms; medians−0.446ms/+0.667ms. These finite samples do not establish a broad TTFT improvement. Full TPOT means/tails are in analysis.json.

Normal public startup uses fresh isolated caches. Ordered M128 gateup/down/M4-vocab/M8-vocab tactics: C1 A2=1/1/2/2,A3=3/4/2/2,B2=1/4/2/2,B3=1/4/2/2; C2 A4=1/1/2/2,A5=1/4/2/2,B4=1/4/2/2,B5=1/1/2/2. Warm and measured M128 counts are zero all8 runs, so differing M128 tactics had no observed participation. Each candidate admission verifies target-qualified B1 or B2 successful eager/capture36 with stride(128,1). This serving observer does not count target graph replays; full measured B1/B2 counters were added independently for subsequent accuracy, without altering these completed runs.

Drivers689506/689507 terminal. Exact commands/manifests/server_info/observers/raw request data retained. Complete raw archive remains remote /root/qvl/experiments/qkv-small-production-serving.tar.gz and /tmp/qvl-qkv-small-production/serving-original.tar.gz. Ordered1MiB parts and SHA256 reconstruction manifest preserve archive; oversized extracted readable files are losslessly gzipped. Python scripts use.py.txt suffix in extracted copy only.
