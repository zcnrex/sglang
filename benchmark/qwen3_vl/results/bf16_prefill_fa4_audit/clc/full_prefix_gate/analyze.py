import collections
import json
import pathlib
import re
import statistics

r = pathlib.Path(__file__).parent
out = {"pairs": [], "coverage": {}}
for i in range(4):
    a = json.loads((r / f"fa4-pack-serving/A/gpu{i}/summary.json").read_text())
    b = json.loads((r / f"fa4-pack-serving/B/gpu{i}/summary.json").read_text())
    c, t = (a, b) if i % 2 == 0 else (b, a)
    out["pairs"].append(
        {
            "gpu": i,
            "control": c,
            "candidate": t,
            "throughput_delta_percent": 100
            * (t["output_throughput"] / c["output_throughput"] - 1),
            "ttft_delta_percent": 100 * (t["median_ttft_ms"] / c["median_ttft_ms"] - 1),
        }
    )
d = [x["throughput_delta_percent"] for x in out["pairs"]]
mean = statistics.mean(d)
h = 3.182446 * statistics.stdev(d) / 2
out["mean_delta_percent"] = mean
out["paired_t95_percent"] = [mean - h, mean + h]
for p in r.glob("*/**/layer0-contexts-pid*.jsonl"):
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    log = (p.parent / "server.log").read_text()
    mt = re.search(r"\[(.*?)\] Cache flushed successfully", log)
    scope = "whole_run"
    if mt:
        n = 0
        for i in range(len(rows) - 1, -1, -1):
            n += rows[i]["prefix_lens"].count(0)
            if n == 128:
                break
        assert n == 128
        rows = rows[i:]
        scope = "final128fresh_requests_cross_checked_flush"
    c = collections.Counter()
    reason = collections.Counter()
    for x in rows:
        c["fa4_query_tokens" if x["used_fa4"] else "fallback_query_tokens"] += x[
            "query_tokens"
        ]
        if not x["used_fa4"]:
            reason[x["fallback_reason"]] += x["query_tokens"]
    out["coverage"][str(p.relative_to(r))] = {
        "scope": scope,
        "contexts": len(rows),
        **c,
        "fallback_reasons": dict(reason),
        "max_scratch_bytes": max(x["max_scratch_bytes"] for x in rows),
        "fresh_context_requests": sum(
            sum(v == 0 for v in x["prefix_lens"]) for x in rows
        ),
    }
(r / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
