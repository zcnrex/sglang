"""Offline Chrome trace metrics; explicit GPU windows and source-backed annotations."""

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


def union(intervals):
    total = 0.0
    end = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def measure(events, window, device, annotations):
    lo, hi = window["start_us"], window["end_us"]
    assert hi > lo
    selected = []
    membership = (
        set(window["kernel_event_indices"])
        if "kernel_event_indices" in window
        else None
    )
    boundary = []
    for i, e in enumerate(events):
        if membership is not None and i not in membership:
            continue
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        args = e.get("args", {})
        if str(args.get("device")) != str(device):
            continue
        a, b = e["ts"], e["ts"] + e["dur"]
        if b <= lo or a >= hi:
            continue
        if a < lo or b > hi:
            boundary.append(i)
        selected.append((e, max(a, lo), min(b, hi)))
    if membership is not None:
        assert len(selected) == len(membership), (
            "Membership indices outside device/window or nonkernel events"
        )
    assert selected, (
        "No kernels in explicit device/window; inspect trace category/device keys"
    )
    assert not boundary, f"GPU window cuts kernels: {boundary[:10]}"
    rows = collections.defaultdict(list)
    for e, a, b in selected:
        rows[e["name"]].append((e, a, b))
    table = []
    for name, group in rows.items():
        evidence = annotations.get(name, {})
        if evidence:
            assert (
                evidence.get("source")
                and evidence.get("source_revision")
                and evidence.get("reason")
            ), name
        launches = collections.Counter(
            json.dumps(
                {
                    k: v
                    for k, v in e.get("args", {}).items()
                    if k
                    in [
                        "grid",
                        "block",
                        "stream",
                        "registers per thread",
                        "shared memory",
                        "External id",
                        "correlation",
                    ]
                },
                sort_keys=True,
            )
            for e, a, b in group
        )
        table.append(
            dict(
                name=name,
                count=len(group),
                sum_us=sum(b - a for e, a, b in group),
                union_us=union([(a, b) for e, a, b in group]),
                annotation=evidence
                or {"classification": "unverified", "dtype": "unknown"},
                launches=[
                    dict(args=json.loads(k), count=v) for k, v in launches.items()
                ],
            )
        )
    intervals = [(a, b) for e, a, b in selected]
    summed = sum(b - a for a, b in intervals)
    occupied = union(intervals)
    span = max(b for a, b in intervals) - min(a for a, b in intervals)
    return dict(
        window=window,
        kernel_count=len(selected),
        kernel_sum_us=summed,
        kernel_interval_union_us=occupied,
        kernel_critical_span_us=span,
        within_span_gap_us=span - occupied,
        concurrent_kernel_excess_us=summed - occupied,
        kernels=sorted(table, key=lambda x: -x["sum_us"]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    manifest_path = Path(a.manifest).resolve()
    m = json.loads(manifest_path.read_text())
    results = []
    for run in m["runs"]:
        for key in [
            "framework",
            "source_revision",
            "versions",
            "stage",
            "batch_size",
            "graph_on",
            "windows",
            "trace_sha256",
            "device",
            "input_contract_evidence",
        ]:
            assert key in run, key
        path = Path(run["trace"])
        path = path if path.is_absolute() else manifest_path.parent / path
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == run["trace_sha256"]
        data = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
        assert not isinstance(data, dict) or data.get("displayTimeUnit", "us") in [
            "ms",
            "us",
            "ns",
        ]
        # Kineto Chrome ts/dur are microseconds; displayTimeUnit is UI-only.
        events = data["traceEvents"] if isinstance(data, dict) else data
        assert run["stage"] in ["decode", "prefill"]
        if run["stage"] == "decode":
            assert run["graph_on"] and len(run["windows"]) == 5
            assert len(run["sequence_lengths"]) == 5
            assert all(len(x) == run["batch_size"] for x in run["sequence_lengths"])
        windows = sorted(run["windows"], key=lambda x: x["start_us"])
        assert all(
            x["end_us"] <= y["start_us"] for x, y in zip(windows, windows[1:])
        ), "Overlapping step windows"
        results.append(
            dict(
                metadata=run,
                steps=[
                    measure(events, w, run["device"], run.get("kernel_annotations", {}))
                    for w in windows
                ],
            )
        )
    Path(a.out).write_text(
        json.dumps(
            dict(
                runs=results,
                warning="GPU interval span is not dependency-DAG critical path or end-to-end latency. Kernel sum minus union is observed concurrency, not removable overhead.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
