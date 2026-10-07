import hashlib
import json
import math
import pathlib
import statistics

import numpy as np

root = pathlib.Path("/root/qvl/experiments/decode-lt-production-full")
runs = []
for phase in ["A", "B"]:
    for gpu in range(8):
        p = root / phase / f"gpu{gpu}"
        if not (p / "summary.json").exists():
            continue
        meta = json.loads((p / "metadata.json").read_text())
        row = {**meta, **json.loads((p / "summary.json").read_text())}
        probe = json.loads((p / "probe.json").read_text())
        assert len(probe) == 128
        tokens = [
            [t[1] for t in r["meta_info"]["output_token_logprobs"]] for r in probe
        ]
        logprobs = [
            [t[0] for t in r["meta_info"]["output_token_logprobs"]] for r in probe
        ]
        assert all(len(x) == 2 for x in tokens)
        row["probe_tokens"] = tokens
        row["probe_logprobs"] = logprobs
        row["probe_sha256"] = hashlib.sha256(
            (p / "probe.json").read_bytes()
        ).hexdigest()
        runs.append(row)
result = {"runs": runs, "paired": []}
if len(runs) == 16:
    reference = runs[0]
    result["probe_all_token_ids_equal"] = all(
        r["probe_tokens"] == reference["probe_tokens"] for r in runs
    )
    result["probe_all_logprobs_equal"] = all(
        r["probe_logprobs"] == reference["probe_logprobs"] for r in runs
    )
    for gpu in range(8):
        c = next(r for r in runs if r["gpu"] == gpu and r["variant"] == "candidate")
        b = next(r for r in runs if r["gpu"] == gpu and r["variant"] == "control")
        result["paired"].append(
            {
                "gpu": gpu,
                "candidate_tps": c["output_throughput"],
                "control_tps": b["output_throughput"],
                "ratio": c["output_throughput"] / b["output_throughput"],
                "candidate_phase": c["phase"],
            }
        )
    ratios = [r["ratio"] for r in result["paired"]]
    result["geomean_ratio"] = math.exp(statistics.mean(math.log(x) for x in ratios))
    result["median_ratio"] = statistics.median(ratios)
    result["positive_pairs"] = sum(x > 1 for x in ratios)
    result["phase_means"] = {
        ph: {
            v: statistics.mean(
                r["output_throughput"]
                for r in runs
                if r["phase"] == ph and r["variant"] == v
            )
            for v in ["candidate", "control"]
        }
        for ph in ["A", "B"]
    }
    result["precision_verified"] = all(
        r["server_command"][r["server_command"].index("--dtype") + 1] == "bfloat16"
        and r["server_command"][r["server_command"].index("--kv-cache-dtype") + 1]
        == "bfloat16"
        for r in runs
    )
if len(runs) == 16:
    logs = np.log(np.array(ratios))
    rng = np.random.default_rng(42)
    boot = np.exp(rng.choice(logs, size=(100000, 8), replace=True).mean(axis=1)) - 1
    half = 2.364624251 * float(logs.std(ddof=1)) / math.sqrt(8)
    result["uncertainty"] = {
        "method": "GPU-paired log-ratio; descriptive intervals from8pairs across2periods, not independent repeated-day validation",
        "bootstrap_seed": 42,
        "bootstrap_resamples": 100000,
        "bootstrap_gain_percent_ci95": [
            float(x * 100) for x in np.quantile(boot, [0.025, 0.975])
        ],
        "t_df": 7,
        "t_gain_percent_ci95": [
            (math.exp(float(logs.mean()) - half) - 1) * 100,
            (math.exp(float(logs.mean()) + half) - 1) * 100,
        ],
        "interpretation": "Both intervals must be checked before claiming gain; experiment is small and has hardware/period variation.",
    }
(root / "crossover_summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
print("completed_runs", len(runs))
