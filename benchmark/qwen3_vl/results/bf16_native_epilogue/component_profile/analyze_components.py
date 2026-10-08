import collections
import csv
import datetime
import gzip
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
tele = []
with open(root / "telemetry.csv") as f:
    for r in csv.DictReader(f):
        r = {k.strip(): v.strip() for k, v in r.items()}
        try:
            ts = (
                datetime.datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
            )
        except ValueError:
            continue
        tele.append((ts, r))
results = []
for path in sorted(root.glob("gpu*/trace*.json.gz")):
    d = json.load(gzip.open(path))
    ev = d["traceEvents"]
    scopes = [
        e
        for e in ev
        if e.get("cat") == "user_annotation"
        and e["name"] in ("qvl_gateup_activation", "qvl_down")
    ]
    launches = {
        e.get("args", {}).get("correlation"): e
        for e in ev
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "Launch" in e["name"]
    }
    groups = collections.defaultdict(
        lambda: {
            "count": 0,
            "gpu_us": 0.0,
            "cpu_scope_us": 0.0,
            "names": collections.Counter(),
        }
    )
    for s in scopes:
        groups[s["name"]]["cpu_scope_us"] += s["dur"]
    kernels = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda x: x["ts"])
    delays = []
    gaps = []
    last = None
    for k in kernels:
        launch = launches.get(k.get("args", {}).get("correlation"))
        scope = None
        if launch:
            scope = next(
                (
                    s
                    for s in scopes
                    if s["pid"] == launch["pid"]
                    and s["tid"] == launch["tid"]
                    and s["ts"] <= launch["ts"] < s["ts"] + s["dur"]
                ),
                None,
            )
            delays.append(k["ts"] - launch["ts"] - launch["dur"])
        n = k["name"]
        group = (
            scope["name"]
            if scope
            else (
                "context_attention"
                if "FlashAttentionForward" in n
                else "decode_attention"
                if "fmhaSm" in n
                else "other_gemm"
                if "nvjet" in n
                else "norm"
                if "rmsnorm" in n.lower()
                else "qknorm_rope"
                if "qknorm" in n.lower() or "mrope" in n.lower()
                else "cache_pack"
                if "pack_kv" in n or "reshape_and_cache" in n
                else "other"
            )
        )
        k["_group"] = group
        g = groups[group]
        g["count"] += 1
        g["gpu_us"] += k["dur"]
        g["names"][n] += 1
        if last is not None and k["ts"] > last:
            gap = k["ts"] - last
            late = launch and launch["ts"] + launch["dur"] >= last
            gaps.append(
                {
                    "us": gap,
                    "late_cpu_launch": bool(late),
                    "kernel": n,
                    "cpu_end_minus_prev_gpu_end_us": launch["ts"] + launch["dur"] - last
                    if launch
                    else None,
                }
            )
        last = max(last or 0, k["ts"] + k["dur"])
    main_start = next(
        k["ts"] for k in kernels if k["_group"] == "other_gemm" and "nvjet" in k["name"]
    )
    main_end = max(k["ts"] + k["dur"] for k in kernels if k["_group"] == "qvl_down")
    main_kernels = [k for k in kernels if main_start <= k["ts"] < main_end]
    main_cursor = main_start
    main_gap = 0.0
    for z in main_kernels:
        main_gap += max(0, z["ts"] - main_cursor)
        main_cursor = max(main_cursor, min(main_end, z["ts"] + z["dur"]))
    main_gap /= 1000
    span = max(k["ts"] + k["dur"] for k in kernels) - kernels[0]["ts"]
    total = sum(k["dur"] for k in kernels)
    start = (d["baseTimeNanoseconds"] + kernels[0]["ts"] * 1000) / 1e9
    end = (d["baseTimeNanoseconds"] + last * 1000) / 1e9
    gpu = int(path.parent.name[3:])
    samples = [r for ts, r in tele if start <= ts <= end and int(r["index"]) == gpu]
    clocks = {}
    for field in ["clocks.current.sm [MHz]", "power.draw [W]", "temperature.gpu"]:
        values = [float(r[field].split()[0]) for r in samples if field in r]
        if values:
            clocks[field] = {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "samples": len(values),
            }
    out = {
        "trace": str(path.relative_to(root)),
        "gpu": gpu,
        "variant": int(path.name.split("-")[-1].split(".")[0]),
        "groups": dict(groups),
        "transformer_region_gap_ms": main_gap,
        "gpu_sum_ms": total / 1000,
        "gpu_span_ms": span / 1000,
        "gpu_interkernel_gap_ms": sum(g["us"] for g in gaps) / 1000,
        "gaps_over_5us": sum(g["us"] > 5 for g in gaps),
        "late_launch_gap_ms": sum(g["us"] for g in gaps if g["late_cpu_launch"]) / 1000,
        "launch_to_gpu_median_us": statistics.median(delays),
        "launch_to_gpu_min_us": min(delays),
        "clocks": clocks,
        "epoch_window": [start, end],
        "top_gaps": sorted(gaps, key=lambda x: -x["us"])[:10],
    }
    results.append(out)
(root / "components.json").write_text(json.dumps(results, indent=2) + "\n")
for r in results:
    print(
        r["trace"],
        {k: round(v["gpu_us"] / 1000, 3) for k, v in r["groups"].items()},
        "sum",
        round(r["gpu_sum_ms"], 3),
        "gap",
        round(r["gpu_interkernel_gap_ms"], 3),
        "launchlead",
        round(r["launch_to_gpu_median_us"], 1),
        "clocks",
        r["clocks"],
    )
