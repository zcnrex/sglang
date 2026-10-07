import ctypes
import json
import sys

import torch
import triton

sys.path.insert(0, "/root/qvl/experiments/lossless-kv")
from lossless import pack

lib = ctypes.CDLL("/root/qvl/experiments/xqa-producer/producer.so")
fn = lib.run
fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]


def run(p, m, o, rows, packed, check):
    fn(
        p.data_ptr(),
        m.data_ptr(),
        o.data_ptr(),
        rows,
        packed,
        check,
        torch.cuda.current_stream().cuda_stream,
    )


for name, x in [
    (
        "bits",
        torch.arange(65536, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(-1, 128),
    ),
    (
        "random",
        torch.randint(
            -32768, 32768, (4096, 128), device="cuda", dtype=torch.int16
        ).view(torch.bfloat16),
    ),
] + [
    (
        f"{a}{b}",
        torch.load(
            f"/root/qvl/experiments/real-kv-capture/layer{a}_{b}.pt",
            map_location="cuda",
            weights_only=True,
        ).reshape(-1, 128),
    )
    for a in [0, 17, 35]
    for b in "kv"
]:
    p, m = pack(x)
    o = torch.empty_like(x)
    for packed in [False, True]:
        run(p if packed else x, m, o, x.shape[0], packed, 1)
        assert torch.equal(o.view(torch.int16), x.view(torch.int16))
    print(name, "bitwise", flush=True)
x = (
    torch.load(
        "/root/qvl/experiments/real-kv-capture/layer17_k.pt",
        map_location="cuda",
        weights_only=True,
    )
    .reshape(-1, 128)
    .repeat(256, 1)
)
p, m = pack(x)
o = torch.empty(x.shape[0] // 32 * 256, device="cuda", dtype=torch.int32)
run(x, m, o, x.shape[0], 0, 0)
ref = o.clone()
run(p, m, o, x.shape[0], 1, 0)
assert torch.equal(ref, o)
print(
    "logical_bytes",
    x.numel() * 2,
    "hit",
    float(((m.to(torch.int32) & 256) != 0).float().mean()),
    flush=True,
)
for rep in range(3):
    for packed in [False, True][:: 1 if rep % 2 == 0 else -1]:
        ms = triton.testing.do_bench_cudagraph(
            lambda: run(p if packed else x, m, o, x.shape[0], packed, 0), rep=100
        )
        print(json.dumps(dict(rep=rep, packed=packed, ms=ms)), flush=True)
