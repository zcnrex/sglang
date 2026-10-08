import collections
import gzip
import json
import pathlib

root = pathlib.Path("benchmark/qwen3_vl/results/bf16_native_epilogue/component_profile")
out = []


def union(a):
    end = -1e99
    total = 0
    for s, e in sorted(a):
        total += max(0, e - max(s, end))
        end = max(end, e)
    return total


for p in sorted(root.glob("gpu*/trace*.gz")):
    es = json.load(gzip.open(p))["traceEvents"]
    ks = sorted([e for e in es if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    start = ks[0]["ts"]
    first = next(e["ts"] for e in ks if "nvjet" in e["name"])
    pos = next(e for e in ks if e["name"] == "compute_position_kernel")
    posend = pos["ts"] + pos["dur"]
    copy = next(
        e
        for e in sorted(es, key=lambda x: x.get("ts", 0))
        if e.get("cat") == "gpu_memcpy" and e["ts"] > posend and "Pageable" in e["name"]
    )
    stop = copy["ts"]
    gpu = [
        e
        for e in es
        if e.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"]
        and start <= e.get("ts", 0) < first
    ]
    cpu = [e for e in es if e.get("cat") == "cpu_op" and posend <= e["ts"] < stop]
    sync = [
        e
        for e in es
        if e.get("cat") in ["cuda_runtime", "cuda_driver"]
        and start <= e["ts"] < first
        and ("Synchronize" in e["name"] or "Malloc" in e["name"])
    ]
    out.append(
        {
            "trace": str(p.relative_to(root)),
            "pre_gemm_span_us": first - start,
            "pre_gemm_gpu_busy_us": union(
                [(e["ts"], min(first, e["ts"] + e["dur"])) for e in gpu]
            ),
            "pre_gemm_gpu_memcpy_sum_us": sum(
                e["dur"] for e in gpu if e["cat"] == "gpu_memcpy"
            ),
            "position_to_pageable_copy_gap_us": stop - posend,
            "cpu_op_counts_in_gap": dict(collections.Counter(e["name"] for e in cpu)),
            "sync_or_malloc_before_gemm": [
                {k: e[k] for k in ["name", "dur"]} for e in sync
            ],
            "pre_gemm_gpu_events": [
                {
                    "name": e["name"],
                    "category": e["cat"],
                    "start_us": e["ts"] - start,
                    "duration_us": e["dur"],
                }
                for e in sorted(gpu, key=lambda e: e["ts"])
            ],
        }
    )
print(json.dumps(out, indent=2))
