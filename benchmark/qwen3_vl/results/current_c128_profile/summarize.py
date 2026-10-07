import collections
import json
import pathlib

root = pathlib.Path("/root/qvl/experiments/current-c128-profile")
rows = []
for p in root.glob("events-*.jsonl"):
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except:
            pass
summary = {}
for epoch in sorted({r["epoch"] for r in rows}):
    group = [r for r in rows if r["epoch"] == epoch]
    total = sum(r["gpu_ms"] for r in group)
    modes = {}
    for mode in sorted({r["mode"] for r in group}):
        g = [r for r in group if r["mode"] == mode]
        ms = sum(r["gpu_ms"] for r in g)
        modes[mode] = {
            "steps": len(g),
            "gpu_ms": ms,
            "forward_gpu_fraction": ms / total,
            "input_tokens": sum(r["input_tokens"] for r in g),
            "batch_size_min": min(r["batch_size"] for r in g),
            "batch_size_max": max(r["batch_size"] for r in g),
            "batch_sizes": dict(collections.Counter(r["batch_size"] for r in g)),
        }
    summary[epoch] = {
        "total_forward_gpu_ms": total,
        "host_span_s": max(r["host_end"] for r in group)
        - min(r["wall_start"] for r in group),
        "modes": modes,
    }
(root / "event-summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
