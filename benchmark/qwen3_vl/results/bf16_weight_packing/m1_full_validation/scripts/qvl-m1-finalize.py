import hashlib
import json
import math
import pathlib
import tarfile

import numpy as np

roots = [
    pathlib.Path("/root/qvl/experiments/m1-gateup-serving"),
    pathlib.Path("/root/qvl/experiments/m1-gateup-full-accuracy"),
]
paths = [
    "/root/qvl/experiments/m1-gateup-accuracy/hooks/sitecustomize.py",
    "/tmp/qvl-m1-serving.py",
    "/tmp/qvl-m1-full-gsm.py",
]
provenance = {
    "source_equivalent_commit": "5f60fb8b67",
    "source_audit": json.loads(
        pathlib.Path(
            "/root/qvl/experiments/m1-gateup-accuracy/source-audit.json"
        ).read_text()
    ),
    "file_sha256": {
        p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths
    },
    "candidate_switch": "QVL_M1_CANDIDATE=1 for candidate only",
    "concurrent_work": "four independent c1 accuracy workers on GPUs4-7 during serving GPUs0-3; no GPUsharing",
}
for r in roots:
    (r / "provenance.json").write_text(json.dumps(provenance, indent=2))
r = roots[0]
d = json.loads((r / "paired-summary.json").read_text())
logs = np.log([p["ratio"] for p in d["pairs"]])
half = 3.182446305 * logs.std(ddof=1) / 2
rng = np.random.default_rng(42)
boot = np.exp(rng.choice(logs, size=(100000, 4), replace=True).mean(axis=1)) - 1
d["uncertainty"] = {
    "t_ci95_percent": [
        (math.exp(logs.mean() - half) - 1) * 100,
        (math.exp(logs.mean() + half) - 1) * 100,
    ],
    "bootstrap_ci95_percent": (np.quantile(boot, [0.025, 0.975]) * 100).tolist(),
    "qualification": "descriptive four GPU pairs/two periods, not repeated-day validation",
}
d["runs"] = []
for p in sorted(r.glob("*/gpu*/summary.json")):
    x = json.loads(p.read_text())
    m = json.loads((p.parent / "metadata.json").read_text())
    d["runs"].append(
        {"phase": m["phase"], "gpu": m["gpu"], "variant": m["variant"], **x}
    )
(r / "paired-summary.json").write_text(json.dumps(d, indent=2))
print(json.dumps(d))
with tarfile.open("/tmp/qvl-m1-final.tar.gz", "w:gz") as t:
    for r, label in zip(roots, ["serving", "full-accuracy"]):
        for p in r.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(r)
            if "isolated-cache" in rel.parts:
                continue
            if "startup-cache" in rel.parts and not (
                "flashinfer" in rel.parts and p.suffix == ".json"
            ):
                continue
            if p.name.startswith("shard") and p.suffix == ".jsonl":
                continue
            if p.suffix in {".json", ".jsonl", ".py", ".log", ".csv", ".patch"}:
                t.add(p, arcname=str(pathlib.Path(label) / rel))
    for p in paths + [
        "/tmp/qvl-m1-summary.py",
        "/tmp/qvl-m1-launch.py",
        "/tmp/qvl-m1-finalize.py",
    ]:
        t.add(p, arcname="scripts/" + pathlib.Path(p).name)
