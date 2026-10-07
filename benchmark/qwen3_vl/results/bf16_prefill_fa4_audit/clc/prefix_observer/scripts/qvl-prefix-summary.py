import collections
import json
import pathlib

r = pathlib.Path("/root/qvl/experiments/prefix-observer")
p = r / "A/gpu0"
allrows = [
    json.loads(s)
    for f in p.glob("contexts-*.jsonl")
    for s in f.read_text().splitlines()
]
rows = [x for x in allrows if x["kind"] == "context" and x["epoch"] == 1]
assert len([x for x in allrows if x["kind"] == "flush_cache"]) == 1
hist = collections.Counter()
weight = collections.Counter()
lens = collections.Counter()
records = []
for x in rows:
    assert sum(x["context_extend_lens"]) == x["context_tokens"]
    for pre, ext in zip(x["context_prefix_lens"], x["context_extend_lens"]):
        hist[pre] += 1
        weight[pre] += ext
        lens[ext] += 1
    records.append(
        {
            k: x[k]
            for k in [
                "context_requests",
                "context_tokens",
                "context_prefix_lens",
                "context_extend_lens",
                "suffix_requests",
            ]
        }
    )
res = {
    "measurement_epoch": 1,
    "flush_events": [x for x in allrows if x["kind"] == "flush_cache"],
    "context_calls": len(rows),
    "context_request_chunks": sum(hist.values()),
    "fresh_request_chunks": hist[0],
    "continuation_request_chunks": sum(v for k, v in hist.items() if k > 0),
    "prefix_histogram": dict(sorted(hist.items())),
    "extend_tokens_by_prefix": dict(sorted(weight.items())),
    "extend_histogram": dict(sorted(lens.items())),
    "context_query_tokens": sum(weight.values()),
    "fresh_query_tokens": weight[0],
    "cached_query_tokens": sum(v for k, v in weight.items() if k > 0),
    "all_fresh_calls": sum(not any(x["context_prefix_lens"]) for x in rows),
    "all_fresh_call_tokens": sum(
        x["context_tokens"] for x in rows if not any(x["context_prefix_lens"])
    ),
    "fresh_tokens_in_mixed_prefix_calls": sum(
        sum(
            ext
            for pre, ext in zip(x["context_prefix_lens"], x["context_extend_lens"])
            if pre == 0
        )
        for x in rows
        if any(x["context_prefix_lens"])
    ),
    "cached_prefix_total_tokens": sum(k * v for k, v in hist.items()),
    "suffix_request_counts": dict(
        sorted(collections.Counter(x["suffix_requests"] for x in rows).items())
    ),
    "all_measured_contexts": records,
    "phase_counts": dict(
        collections.Counter(
            str((x.get("epoch"), x.get("phase_marker")))
            for x in allrows
            if x["kind"] == "context"
        )
    ),
}
(r / "prefix-summary.json").write_text(json.dumps(res, indent=2))
print(
    json.dumps(
        {
            k: v
            for k, v in res.items()
            if k
            not in [
                "all_measured_contexts",
                "prefix_histogram",
                "extend_tokens_by_prefix",
                "extend_histogram",
                "suffix_request_counts",
                "flush_events",
            ]
        }
    )
)
