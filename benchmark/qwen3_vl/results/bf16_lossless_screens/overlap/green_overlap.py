import json
import math

import flashinfer
import torch

torch.manual_seed(12)
b, n = 128, 8192
q = torch.randn(b, 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(b * n // 32, 8, 32, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
w = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
blocks = torch.arange(b * n // 32, device="cuda", dtype=torch.int32).reshape(b, -1)
lens = torch.full((b,), n, device="cuda", dtype=torch.int32)
out = torch.empty_like(q)


def attn():
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q,
        (k, v),
        w,
        blocks,
        lens,
        n,
        bmm1_scale=1 / math.sqrt(128),
        bmm2_scale=1.0,
        kv_layout="HND",
        out=out,
    )


def measure(g, reps=30):
    a = torch.cuda.Event(enable_timing=True)
    z = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    z.record()
    z.synchronize()
    return a.elapsed_time(z) / reps


def run(m, attention_sms, gemm_sms):
    x = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(19456, 2560, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(m, 19456, device="cuda", dtype=torch.bfloat16)

    def gemm():
        torch.mm(x, weight.T, out=y)

    for _ in range(5):
        attn()
        gemm()
    torch.cuda.synchronize()
    refa = out.clone()
    refg = y.clone()
    from sgl_kernel import spatial

    astream, gstream = spatial.create_greenctx_stream_by_value(
        attention_sms, gemm_sms, 0
    )
    with torch.cuda.stream(astream):
        attn()
    with torch.cuda.stream(gstream):
        gemm()
    torch.cuda.synchronize()
    graphs = {}
    main = torch.cuda.Stream()
    side = gstream
    with torch.cuda.stream(main):
        for mode in ["attention", "gemm", "serial", "concurrent"]:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=main):
                if mode == "attention":
                    attn()
                elif mode == "gemm":
                    gemm()
                elif mode == "serial":
                    attn()
                    gemm()
                else:
                    side.wait_stream(main)
                    astream.wait_stream(main)
                    with torch.cuda.stream(astream):
                        attn()
                    with torch.cuda.stream(side):
                        gemm()
                    main.wait_stream(astream)
                    main.wait_stream(side)
            graphs[mode] = g
    torch.cuda.synchronize()
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, refa, rtol=0, atol=0)
    torch.testing.assert_close(y, refg, rtol=0, atol=0)
    for i in range(5):
        order = (
            ["attention", "gemm", "serial", "concurrent"]
            if i % 2 == 0
            else ["concurrent", "serial", "gemm", "attention"]
        )
        result = {mode: measure(graphs[mode]) for mode in order}
        print(
            json.dumps(
                {
                    "attention_sms": attention_sms,
                    "gemm_sms": gemm_sms,
                    "m": m,
                    "iteration": i,
                    "ms": result,
                    "serial_over_concurrent": result["serial"] / result["concurrent"],
                    "outputs_bitwise": True,
                }
            ),
            flush=True,
        )


run(8192, 112, 32)
run(8192, 96, 48)
