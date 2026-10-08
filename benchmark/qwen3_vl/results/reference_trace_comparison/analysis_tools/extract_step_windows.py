"""Associate marked model forwards with GPU kernels by CUDA launch correlation."""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path


def extract(events, device, expected_steps=5):
    scopes = []
    for i, e in enumerate(events):
        match = re.fullmatch(r"QVL_decode_step(\d+)_B(\d+).*", e.get("name", ""))
        if match and e.get("ph") == "X" and e.get("cat") == "user_annotation":
            scopes.append((int(match[1]), int(match[2]), i, e))
    scopes.sort()
    assert len(scopes) == expected_steps, (
        f"Expected {expected_steps} marked scopes, found {len(scopes)}"
    )
    assert [x[0] for x in scopes] == list(range(expected_steps)), (
        "Scope indices must be unique 0..4"
    )
    assert len({x[1] for x in scopes}) == 1, "Actual batch differs across scopes"
    launches = [
        (i, e)
        for i, e in enumerate(events)
        if e.get("ph") == "X"
        and e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "launch" in e.get("name", "").lower()
    ]
    kernels = [
        (i, e)
        for i, e in enumerate(events)
        if e.get("ph") == "X"
        and e.get("cat") == "kernel"
        and str(e.get("args", {}).get("device")) == str(device)
    ]
    windows = []
    assigned = set()
    seen_correlations = set()
    for step, batch, index, scope in scopes:
        lo = scope["ts"]
        hi = lo + scope["dur"]
        nested = [
            (i, e)
            for i, e in launches
            if (e.get("pid"), e.get("tid")) == (scope.get("pid"), scope.get("tid"))
            and lo <= e["ts"]
            and e["ts"] + e["dur"] <= hi
        ]
        assert nested, f"No same-thread CUDA launch in step {step}"
        correlations = []
        for i, e in nested:
            c = e.get("args", {}).get("correlation")
            assert c is not None, f"Missing launch correlation {i}"
            # Refuse reused IDs even if all reuse happens on another CPU thread.
            matches = [
                j for j, v in launches if v.get("args", {}).get("correlation") == c
            ]
            assert matches == [i], f"Ambiguous launch correlation {c}: {matches}"
            correlations.append(c)
        assert not (seen_correlations & set(correlations)), (
            "Correlation shared between steps"
        )
        seen_correlations.update(correlations)
        joined = [
            (i, e)
            for i, e in kernels
            if e.get("args", {}).get("correlation") in correlations
        ]
        assert joined, f"No GPU kernels join step {step}"
        for correlation in correlations:
            assert any(
                e["args"].get("correlation") == correlation for i, e in joined
            ), f"Launch has no GPU kernel: {correlation}"
        graph_launches = [e for i, e in nested if "graphlaunch" in e["name"].lower()]
        assert graph_launches, f"No graph launch in decode step {step}"
        graph_ids = set()
        for launch in graph_launches:
            c = launch["args"]["correlation"]
            g = [e for i, e in joined if e["args"].get("correlation") == c]
            assert g and all("graph id" in e["args"] for e in g), (
                f"Missing graph kernel proof correlation {c}"
            )
            graph_ids.update(str(e["args"]["graph id"]) for e in g)
        ids = {i for i, e in joined}
        assert not ids & assigned
        assigned.update(ids)
        windows.append(
            dict(
                step=step,
                batch_size=batch,
                start_us=min(e["ts"] for i, e in joined),
                end_us=max(e["ts"] + e["dur"] for i, e in joined),
                kernel_event_indices=sorted(ids),
                cpu_scope_event_index=index,
                launch_event_indices=[i for i, e in nested],
                correlations=correlations,
                graph_ids=sorted(graph_ids),
            )
        )
    # Scope time and GPU time are same trace clock, but launch-to-kernel correlation,
    # not CPU scope boundaries, establishes membership and GPU bounds.
    unrelated = [
        dict(
            event_index=i,
            name=e["name"],
            ts=e["ts"],
            dur=e["dur"],
            args=e.get("args", {}),
        )
        for i, e in kernels
        if i not in assigned
    ]
    for w in windows:
        w["unassigned_overlapping_indices"] = [
            x["event_index"]
            for x in unrelated
            if x["ts"] < w["end_us"] and x["ts"] + x["dur"] > w["start_us"]
        ]
    return dict(
        windows=windows,
        unassigned_gpu_kernels=unrelated,
        limitation="Marked scope includes model launches only as instrumented. Unassigned kernels may include sampling or housekeeping; classify from source/correlation, never assume all are sampling. Last-step sampling can be outside capture.",
    )


def extract_v2(events, device, decoder_pattern, logits_pattern):
    # First validate five outer execute_model scopes and their graph evidence.
    outer = extract(events, device)
    samples = []
    for i, e in enumerate(events):
        m = re.fullmatch(
            r"QVL_(?:decode_)?sample_step(\d+)_B(\d+).*", e.get("name", "")
        )
        if m and e.get("ph") == "X" and e.get("cat") == "user_annotation":
            samples.append((int(m[1]), int(m[2]), i, e))
    samples.sort()
    assert len(samples) == 5 and [x[0] for x in samples] == list(range(5)), (
        "Need five sample scopes"
    )
    launch_events = [
        (i, e)
        for i, e in enumerate(events)
        if e.get("ph") == "X"
        and e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "launch" in e.get("name", "").lower()
    ]
    gpu = {
        i: e
        for i, e in enumerate(events)
        if e.get("ph") == "X"
        and e.get("cat") == "kernel"
        and str(e.get("args", {}).get("device")) == str(device)
    }

    def inside(child, parent):
        return (
            (child.get("pid"), child.get("tid"))
            == (parent.get("pid"), parent.get("tid"))
            and parent["ts"] <= child["ts"]
            and child["ts"] + child["dur"] <= parent["ts"] + parent["dur"]
        )

    def nested(parent, pattern):
        found = [
            (i, e)
            for i, e in enumerate(events)
            if e.get("ph") == "X"
            and e.get("cat") == "user_annotation"
            and re.fullmatch(pattern, e.get("name", ""))
            and inside(e, parent)
        ]
        assert len(found) == 1, f"Expected one nested {pattern}, found {len(found)}"
        return found[0]

    def members(scope):
        ls = [(i, e) for i, e in launch_events if inside(e, scope)]
        ids = set()
        for i, l in ls:
            c = l.get("args", {}).get("correlation")
            assert c is not None
            assert [
                j for j, e in launch_events if e.get("args", {}).get("correlation") == c
            ] == [i], f"Ambiguous correlation {c}"
            joined = {
                j for j, e in gpu.items() if e.get("args", {}).get("correlation") == c
            }
            assert joined, f"Launch has no GPU kernel: {l['name']}/{c}"
            ids.update(joined)
        return ids

    def window(ids, step, label):
        assert ids, f"Empty required {label}"
        return dict(
            step=step,
            label=label,
            kernel_event_indices=sorted(ids),
            start_us=min(gpu[i]["ts"] for i in ids),
            end_us=max(gpu[i]["ts"] + gpu[i]["dur"] for i in ids),
        )

    combined = []
    components = []
    covered = set()
    for w, (step, batch, sindex, sample) in zip(outer["windows"], samples):
        assert step == w["step"] and batch == w["batch_size"]
        parent = events[w["cpu_scope_event_index"]]
        di, dec = nested(parent, decoder_pattern)
        li, logits = nested(sample, logits_pattern)
        for scope in (dec, logits):
            suffix = re.search(r"_step(\d+)_B(\d+)$", scope["name"])
            if suffix:
                assert (int(suffix[1]), int(suffix[2])) == (step, batch), (
                    "Nested scope step/batch mismatch"
                )
        assert parent["ts"] + parent["dur"] <= sample["ts"], (
            "Sample must follow its execute scope"
        )
        d = members(dec)
        l = members(logits)
        sp = members(sample)
        worker = set(w["kernel_event_indices"])
        assert d <= worker and l <= sp and not (d & l) and not (worker & sp)
        assert any(
            "graphlaunch" in e["name"].lower() and inside(e, dec)
            for i, e in launch_events
        ), "Decoder nested scope lacks graph launch"
        merged = window(d | l, step, "decoder_plus_logits")
        merged["batch_size"] = batch
        merged["graph_ids"] = w["graph_ids"]
        combined.append(merged)
        groups = {
            "decoder": d,
            "logits": l,
            "worker_bookkeeping": worker - d,
            "sample_outside_logits": sp - l,
        }
        # Classify by CPU launch order, without claiming semantic gather/sampler identity.
        before = {
            j
            for i, e in launch_events
            if inside(e, sample) and e["ts"] + e["dur"] <= logits["ts"]
            for j, g in gpu.items()
            if g.get("args", {}).get("correlation")
            == e.get("args", {}).get("correlation")
        }
        groups["sample_before_logits"] = before
        groups["sample_after_or_other"] = sp - l - before
        components.append(
            dict(
                step=step,
                scope_event_indices=dict(
                    decoder=di,
                    logits=li,
                    sample=sindex,
                    worker=w["cpu_scope_event_index"],
                ),
                groups={
                    name: window(ids, step, name)
                    if ids
                    else dict(step=step, label=name, kernel_event_indices=[])
                    for name, ids in groups.items()
                },
            )
        )
        covered.update(worker | sp)
    return dict(
        windows=combined,
        components=components,
        unassigned_gpu_kernels=[
            dict(event_index=i, **e) for i, e in gpu.items() if i not in covered
        ],
        limitation="Combined decoder+logits kernel membership excludes worker bookkeeping and sample outside logits. Their intervening GPU span can contain excluded work/host gaps; do not call it pure model latency. Sample-before/after labels are temporal, not proven gather/sampler semantic classifications. Overlapping component groups must not be summed.",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--device", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layout", choices=["model", "vllm-v2"], default="model")
    p.add_argument("--decoder-pattern", default=r"QVL_decoder_model(?:_step\d+_B\d+)?")
    p.add_argument("--logits-pattern", default=r"QVL_compute_logits(?:_step\d+_B\d+)?")
    a = p.parse_args()
    path = Path(a.trace)
    raw = path.read_bytes()
    data = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    result = (
        extract(data["traceEvents"], a.device)
        if a.layout == "model"
        else extract_v2(
            data["traceEvents"], a.device, a.decoder_pattern, a.logits_pattern
        )
    )
    result.update(trace=str(path), trace_sha256=hashlib.sha256(raw).hexdigest())
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
