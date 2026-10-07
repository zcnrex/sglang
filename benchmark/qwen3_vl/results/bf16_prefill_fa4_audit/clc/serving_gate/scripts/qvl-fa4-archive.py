import json
import math
import pathlib
import tarfile

import numpy as np

s = pathlib.Path("/root/qvl/experiments/fa4-prefill-serving")
d = json.loads((s / "paired-summary.json").read_text())
logs = np.log([x["ratio"] for x in d["pairs"]])
half = 3.182446305 * logs.std(ddof=1) / 2
d["t_ci95_percent"] = [
    (math.exp(logs.mean() - half) - 1) * 100,
    (math.exp(logs.mean() + half) - 1) * 100,
]
d["coverage_qualification"] = (
    "sparse snapshots; final counters may be overwritten by other server processes; serving preserved positive CLC snapshots, accuracy final zero unverified"
)
(s / "paired-summary.json").write_text(json.dumps(d, indent=2))
r = pathlib.Path("/root/qvl/experiments/fa4-prefill-accuracy")
a = json.loads((r / "control/results/metrics.json").read_text())
b = json.loads((r / "candidate/results/metrics.json").read_text())
x = {
    "control_correct": a["correct"],
    "candidate_correct": b["correct"],
    "total": 128,
    "coverage_status": "candidate final counter zero; unverified",
    "control_only": [],
    "candidate_only": [],
}
for p, q in zip(a["outcomes"], b["outcomes"]):
    if p["correct"] and not q["correct"]:
        x["control_only"].append(p["dataset_row"])
    if q["correct"] and not p["correct"]:
        x["candidate_only"].append(q["dataset_row"])
(r / "paired-summary.json").write_text(json.dumps(x, indent=2))
with tarfile.open("/tmp/qvl-fa4-gate.tar.gz", "w:gz") as t:
    for root, label in [(s, "serving"), (r, "accuracy")]:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if "isolated-cache" in rel.parts:
                continue
            if "startup-cache" in rel.parts and not (
                "flashinfer" in rel.parts and p.suffix == ".json"
            ):
                continue
            if p.suffix in {".json", ".jsonl", ".log", ".csv", ".patch"}:
                t.add(p, arcname=str(pathlib.Path(label) / rel))
    for p in [
        "/tmp/qvl-fa4-serving.py",
        "/tmp/qvl-fa4-accuracy.py",
        "/tmp/qvl-fa4-launch.py",
        "/tmp/qvl-fa4-archive.py",
        "/root/qvl/experiments/fa4_clc_serving_hook.py",
    ]:
        t.add(p, arcname="scripts/" + pathlib.Path(p).name)
print(json.dumps(x))
print(d["t_ci95_percent"])
