import collections
import gzip
import hashlib
import json
import pathlib

root = pathlib.Path("/root/qvl/experiments/current-c128-profile")
out = []
for p in sorted(root.rglob("*.trace.json.gz")):
    d = json.load(gzip.open(p))
    events = d["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    total = sum(e.get("dur", 0) for e in kernels)
    counts = collections.Counter()
    dur = collections.Counter()
    for e in kernels:
        counts[e["name"]] += 1
        dur[e["name"]] += e.get("dur", 0)
    out.append(
        {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "size_bytes": p.stat().st_size,
            "kernel_gpu_ms": total / 1000,
            "steps": [
                e["name"]
                for e in events
                if e.get("cat") == "user_annotation"
                and e.get("name", "").startswith("step[")
            ],
            "kernels": [
                {
                    "name": name,
                    "gpu_ms": ms / 1000,
                    "share": ms / total,
                    "launches": counts[name],
                }
                for name, ms in dur.most_common()
            ],
        }
    )
(root / "trace-summary.json").write_text(json.dumps(out, indent=2))
print(
    json.dumps(
        [
            {"path": x["path"], "gpu_ms": x["kernel_gpu_ms"], "steps": x["steps"]}
            for x in out
        ],
        indent=2,
    )
)
