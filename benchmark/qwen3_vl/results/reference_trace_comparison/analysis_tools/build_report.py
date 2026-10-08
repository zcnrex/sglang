"""Render bounded capture metrics without treating summed durations as latency."""

import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def group(n):
    if n.startswith("fmha"):
        return "Attention"
    if "splitk_epilogue" in n:
        return "QKV GEMM + norm/MRoPE/cache"
    if "nvjet" in n or "Gemm" in n or "splitKreduce" in n:
        return "Other GEMM/reduction"
    if "rmsnorm" in n.lower() or "rms_norm" in n:
        return "Residual/RMSNorm"
    if "act_and_mul" in n or "silu" in n:
        return "SiLU × up"
    if (
        "qk_norm_mrope" in n
        or "reshape_and_cache" in n
        or "arange_bitwise" in n
        or n in ("triton_red_fused_4",)
    ):
        return "QK/MRoPE/cache preparation"
    return "Other (includes first-layer compiled variants)"


def render():
    rows = []
    for fw in ["sglang", "vllm"]:
        for b in [8, 16, 128]:
            d = (
                BASE / "analysis_inputs" / f"sglang-b{b}"
                if fw == "sglang"
                else BASE / "vllm" / f"b{b}-v2"
            )
            if (d / "decode-metrics.json").exists():
                rows.append(
                    (fw, b, d, json.loads((d / "decode-metrics.json").read_text()))
                )
    text = [
        "# Five-step profiler comparison",
        "",
        "These are instrumented GPU traces, not a new serving throughput benchmark or a direct TTFT measurement. All five steps are retained. SGLang measures `ModelRunner.forward`, including graph/attention metadata; vLLM measures nested decoder plus vocabulary projection, with worker bookkeeping and sampling separate. The two scope boundaries are not automatically identical.",
        "",
        "## Decode timings",
        "",
        "Microseconds; median across five steps. Union counts occupied kernel intervals once; sum counts overlapping kernels repeatedly. Span is first kernel start to last kernel end, not a dependency critical path.",
        "",
        "|Framework|B|Kernels/step|Kernel sum|Interval union|Span|Sum − union|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fw, b, d, m in rows:
        s = m["steps"]
        med = lambda k: statistics.median(x[k] for x in s)
        text.append(
            f"|{fw}|{b}|{s[0]['kernel_count']}|{med('kernel_sum_us'):.1f}|{med('kernel_interval_union_us'):.1f}|{med('kernel_critical_span_us'):.1f}|{med('concurrent_kernel_excess_us'):.1f}|"
        )
    text += [
        "",
        "## Operation counts and kernel duration sums",
        "",
        "These sums are attribution, not additive wall-clock savings. QKV fused kernels include GEMM and preparation and cannot be split into separate costs. Generated first-layer variants remain explicitly in Other.",
        "",
        "|Framework|B|Operation|Count/step|Median sum µs|",
        "|---|---:|---|---:|---:|",
    ]
    for fw, b, d, m in rows:
        names = sorted({group(k["name"]) for s in m["steps"] for k in s["kernels"]})
        for n in names:
            vals = [
                (
                    sum(k["count"] for k in s["kernels"] if group(k["name"]) == n),
                    sum(k["sum_us"] for k in s["kernels"] if group(k["name"]) == n),
                )
                for s in m["steps"]
            ]
            label = (
                "Other (metadata/gather/copy)"
                if fw == "sglang" and n.startswith("Other (")
                else n
            )
            text.append(
                f"|{fw}|{b}|{label}|{vals[0][0]}|{statistics.median(x[1] for x in vals):.1f}|"
            )
    text += [
        "",
        "## All steps",
        "",
        "|Framework|B|Step|Count|Sum µs|Union µs|Span µs|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fw, b, d, m in rows:
        for i, s in enumerate(m["steps"]):
            text.append(
                f"|{fw}|{b}|{i}|{s['kernel_count']}|{s['kernel_sum_us']:.1f}|{s['kernel_interval_union_us']:.1f}|{s['kernel_critical_span_us']:.1f}|"
            )
    text += ["", "## Raw triage and exact-name data", ""]
    for fw, b, d, m in rows:
        rel = d.relative_to(BASE)
        text.append(
            f"- {fw} B{b}: [exact-name metrics]({rel}/decode-metrics.json), [correlation windows]({rel}/decode-windows.json)."
        )
        for p in sorted(d.glob("*.triage.txt")):
            text.append(
                f"  - [Unmodified three-table triage: {p.name}]({p.relative_to(BASE)})"
            )
    (BASE / "report-metrics.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    render()
