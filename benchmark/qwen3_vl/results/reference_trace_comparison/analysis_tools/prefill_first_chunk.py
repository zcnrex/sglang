"""Extract an explicit first chunk; no pure-prefill or identical-boundary assumption."""

import gzip
import json
from pathlib import Path

from measure_trace import measure

BASE = Path(__file__).resolve().parents[1]
inputs = {
    "sglang": BASE / "analysis_inputs/sglang-b128/prefill-TP0.trace.json.gz",
    "vllm": BASE
    / "vllm/b128-v2/traces/prefill_rank0.1791491168898869904.pt.trace.json.gz",
}
for fw, p in inputs.items():
    es = json.loads(gzip.decompress(p.read_bytes()))["traceEvents"]
    scopes = [
        e
        for e in es
        if e.get("cat") == "user_annotation"
        and e.get("name") == "QVL_prefill_step0_B2_T16384"
    ]
    assert len(scopes) == 1
    scope = scopes[0]
    launches = [
        (i, e)
        for i, e in enumerate(es)
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and e.get("ph") == "X"
        and "launch" in e.get("name", "").lower()
    ]
    selected = [
        (i, e)
        for i, e in launches
        if (e["pid"], e["tid"]) == (scope["pid"], scope["tid"])
        and scope["ts"] <= e["ts"]
        and e["ts"] + e["dur"] <= scope["ts"] + scope["dur"]
    ]
    correlations = set()
    for i, e in selected:
        c = e["args"]["correlation"]
        assert [
            j for j, l in launches if l.get("args", {}).get("correlation") == c
        ] == [i]
        correlations.add(c)
    ids = [
        i
        for i, e in enumerate(es)
        if e.get("cat") == "kernel"
        and str(e.get("args", {}).get("device")) == "0"
        and e.get("args", {}).get("correlation") in correlations
    ]
    assert all(
        any(es[i]["args"].get("correlation") == c for i in ids) for c in correlations
    )
    window = {
        "step": 0,
        "kernel_event_indices": ids,
        "start_us": min(es[i]["ts"] for i in ids),
        "end_us": max(es[i]["ts"] + es[i]["dur"] for i in ids),
    }
    result = measure(es, window, 0, {})
    result["boundary"] = (
        "SG forward including projection"
        if fw == "sglang"
        else "vLLM execute_model including bookkeeping, excluding subsequent logits and sampling"
    )
    result["trace"] = str(p.relative_to(BASE))
    result["geometry"] = {
        "batch": 2,
        "input_tokens": 16384,
        "seq_lens": [8192, 8192],
        "query_lens": [8192, 8192],
    }
    (BASE / f"prefill-first-chunk-{fw}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        fw,
        result["kernel_count"],
        result["kernel_sum_us"],
        result["kernel_interval_union_us"],
        result["kernel_critical_span_us"],
    )
    for k in result["kernels"][:9]:
        print(k["count"], round(k["sum_us"], 1), k["name"])
