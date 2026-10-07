import json
import statistics
from pathlib import Path

import flashinfer
import torch

torch.manual_seed(20261007)
from compact_rope import triton_mrope_fused as compact_mrope

from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

rows = []
# Trace preserves M/B, not individual lengths. These are synthetic compatible geometries.
for qlens, klens in [([8200, 8007], [8200, 8007]), ([8200, 8007], [8200, 8200])]:
    m = sum(qlens)
    tail = 124
    total = m + tail
    backing = torch.randn(total, 6144, device="cuda", dtype=torch.bfloat16)
    qs = backing[:, :4096].view(total, 32, 128)
    qc = qs.contiguous()
    lens = torch.tensor(klens, device="cuda", dtype=torch.int32)
    maxpages = (max(klens) + 31) // 32
    k = torch.randn(2 * maxpages, 8, 32, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.arange(2 * maxpages, device="cuda", dtype=torch.int32).reshape(
        2, maxpages
    )
    cuq = torch.tensor([0, qlens[0], m], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, klens[0], sum(klens)], device="cuda", dtype=torch.int32)
    ws = torch.empty(512 * 1024**2, device="cuda", dtype=torch.uint8)
    out = torch.empty(m, 32, 128, device="cuda", dtype=torch.bfloat16)

    def call(q):
        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q[:m],
            kv_cache=(k, v),
            workspace_buffer=ws,
            block_tables=table,
            seq_lens=lens,
            max_q_len=max(qlens),
            max_kv_len=262144,
            bmm1_scale=128**-0.5,
            bmm2_scale=1.0,
            batch_size=2,
            cum_seq_lens_q=cuq,
            cum_seq_lens_kv=cuk,
            out=out,
        )

    def copycall():
        qc[:m].copy_(qs[:m])
        call(qc)

    original = backing.clone()
    kw = backing[:, 4096:5120].view(total, 8, 128)
    normweight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    positions = (
        torch.arange(total, device="cuda", dtype=torch.int64).repeat(3, 1) % 8192
    )
    angles = torch.randn(8192, 64, device="cuda", dtype=torch.float32)
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1).bfloat16()

    def chain(compact):
        backing.copy_(original)
        fused_inplace_qknorm(qs, kw, normweight, normweight, eps=1e-6, head_dim=128)
        args = (
            qs.view(total, 4096),
            kw.view(total, 1024),
            cos_sin,
            positions,
            [24, 20, 20],
            128,
            128,
            True,
            False,
            True,
            None,
        )
        if compact:
            compact_mrope(*args, qc.view(total, 4096))
        else:
            triton_mrope_fused(*args)
        call(qc if compact else qs)

    chain(False)
    qref = qs.clone()
    kref = kw.clone()
    oref = out.clone()
    chain(True)
    assert torch.equal(qref, qc) and torch.equal(kref, kw) and torch.equal(oref, out)
    funcs = {"normal_chain": lambda: chain(False), "compact_chain": lambda: chain(True)}
    chain(False)
    ref = out.clone()
    graphs = {}
    for name, fn in funcs.items():
        fn()
        assert torch.equal(ref, out)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(5):
                fn()
        graphs[name] = g
    times = {name: [] for name in funcs}
    for rep in range(8):
        for name in list(funcs)[:: 1 if rep % 2 == 0 else -1]:
            g = graphs[name]
            for _ in range(3):
                g.replay()
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(10):
                g.replay()
            b.record()
            b.synchronize()
            times[name].append(a.elapsed_time(b) * 20)
    result = dict(
        qlens=qlens,
        klens=klens,
        trace_total_m=16331,
        trace_batch=126,
        geometry="synthetic compatible, individual lengths not captured",
        q_strides=list(qs.stride()),
        contiguous_strides=list(qc.stride()),
        bitwise_output=True,
        bitwise_qk=True,
        common_reset_copy_included=True,
        us=times,
        median_us={name: statistics.median(v) for name, v in times.items()},
    )
    rows.append(result)
    print(json.dumps(result), flush=True)
    Path("/root/qvl/experiments/qstride/chain_report.json").write_text(
        json.dumps(rows, indent=2)
    )
