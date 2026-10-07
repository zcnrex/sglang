import json
import math
import pathlib
import statistics

r = pathlib.Path("/root/qvl/experiments/m1-production-full-accuracy")
manifest = json.loads((r / "shards.json").read_text())
result = {"variants": {}}
for v in ["control", "candidate"]:
    outcomes = {}
    tokens = 0
    parts = []
    for i in range(2):
        p = r / f"{v}{i}" / "results/metrics.json"
        if not p.exists():
            break
        d = json.loads(p.read_text())
        if d.get("status") != "complete":
            break
        assert (
            d["total"] == 657
            and d["dataset_sha256"] == manifest["shards"][str(i)]["sha256"]
        )
        for row in d["outcomes"]:
            globalrow = manifest["shards"][str(i)]["original_rows"][
                row["dataset_row"] - 5
            ]
            assert globalrow not in outcomes
            outcomes[globalrow] = row["correct"]
        tokens += d["completion_tokens"]
        parts.append(d["correct"])
    result["variants"][v] = {
        "correct": sum(outcomes.values()),
        "total": len(outcomes),
        "completion_tokens": tokens,
        "shard_correct": parts,
        "outcomes": outcomes,
    }
a = result["variants"]["control"]["outcomes"]
b = result["variants"]["candidate"]["outcomes"]
if len(a) == len(b) == 1314:
    assert set(a) == set(b) == set(range(5, 1319))
    result["control_only"] = [k for k in a if a[k] and not b[k]]
    result["candidate_only"] = [k for k in a if b[k] and not a[k]]
(r / "paired-summary.json").write_text(json.dumps(result, indent=2))
print(
    "accuracy",
    json.dumps(
        {
            **result,
            "variants": {
                v: {k: x for k, x in d.items() if k != "outcomes"}
                for v, d in result["variants"].items()
            },
        }
    ),
)
s = pathlib.Path("/root/qvl/experiments/m1-production-serving")
pairs = []
for gpu in range(4):
    row = {"gpu": gpu}
    for ph in ["A", "B"]:
        p = s / ph / f"gpu{gpu}"
        if not (p / "summary.json").exists():
            continue
        d = json.loads((p / "summary.json").read_text())
        m = json.loads((p / "metadata.json").read_text())
        assert (
            d["completed"] == 30
            and d["total_input_tokens"] == 245760
            and d["total_output_tokens"] == 30720
        )
        row[m["variant"]] = d["output_throughput"]
    if "candidate" in row and "control" in row:
        row["ratio"] = row["candidate"] / row["control"]
    pairs.append(row)
x = {"pairs": pairs}
if all("ratio" in p for p in pairs):
    x["geomean_ratio"] = math.exp(statistics.mean(math.log(p["ratio"]) for p in pairs))
(s / "paired-summary.json").write_text(json.dumps(x, indent=2))
print("serving", json.dumps(x))
