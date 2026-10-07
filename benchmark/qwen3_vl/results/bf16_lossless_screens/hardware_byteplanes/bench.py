import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from cuda.bindings import driver as d

from sglang.srt.utils.cuda_vmm_utils import (
    VmmReservation,
    check_drv,
    make_device_allocation_prop,
    tensor_from_pointer,
)


@triton.jit
def convert(S, P, N: tl.constexpr, B: tl.constexpr, INVERSE: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    if INVERSE:
        sm = tl.load(P + i, i < N, 0).to(tl.uint16)
        ex = tl.load(P + N + i, i < N, 0).to(tl.uint16)
        bits = ((sm & 128) << 8) | (sm & 127) | (ex << 7)
        tl.store(S + i, bits, i < N)
    else:
        bits = tl.load(S + i, i < N, 0).to(tl.uint16)
        tl.store(P + i, ((bits >> 8) & 128) | (bits & 127), i < N)
        tl.store(P + N + i, (bits >> 7) & 255, i < N)


@triton.jit
def read(S, P, O, N: tl.constexpr, B: tl.constexpr, PLANES: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    if PLANES:
        sm = tl.load(P + i, i < N, 0).to(tl.uint32)
        ex = tl.load(P + N + i, i < N, 0).to(tl.uint32)
        bits = ((sm & 128) << 8) | (sm & 127) | (ex << 7)
    else:
        bits = tl.load(S + i, i < N, 0).to(tl.uint16).to(tl.uint32)
    tl.store(O + tl.program_id(0), tl.sum(bits, 0))


torch.cuda.set_device(0)
n = 1 << 31
nbytes = n * 2
allocs = []
props = []


def alloc(comp):
    p = make_device_allocation_prop(0, handle_types=None)
    p.allocFlags.compressionType = comp
    g = int(
        check_drv(
            d.cuMemGetAllocationGranularity(
                p, d.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ),
            "gran",
        )
    )
    r = VmmReservation(nbytes, p, 0)
    h = r.map(0, nbytes, retain_handle=True)
    allocs.append(r)
    a = check_drv(d.cuMemGetAllocationPropertiesFromHandle(h), "prop")
    props.append(
        dict(
            requested=comp,
            actual=int(a.allocFlags.compressionType),
            size=nbytes,
            granularity=g,
        )
    )
    return tensor_from_pointer(
        r.base, nbytes, shape=(nbytes,), dtype=torch.uint8, device_id=0
    )


raw = alloc(1).view(torch.int16)
planes = alloc(1)
plainplanes = alloc(0)
o = torch.empty(triton.cdiv(n, 4096), device="cuda", dtype=torch.int64)
report = dict(
    allocations=props,
    logical_bytes=nbytes,
    layout="two global byte planes: sign/mantissa then full exponent",
    cases={},
)
# Include exhaustive patterns independently of real capture values.
bits = torch.arange(65536, device="cuda").to(torch.int16)
p = torch.empty(131072, device="cuda", dtype=torch.uint8)
back = torch.empty_like(bits)
convert[(16,)](bits, p, 65536, 4096, False)
convert[(16,)](back, p, 65536, 4096, True)
assert torch.equal(bits, back)
for case in ["0k", "17k", "17v"]:
    layer = case[:-1]
    kv = case[-1]
    x = (
        torch.load(
            f"/root/qvl/experiments/real-kv-capture/layer{layer}_{kv}.pt",
            map_location="cuda",
            weights_only=True,
        )
        .reshape(-1)
        .view(torch.int16)
    )
    raw.copy_(x.repeat(n // x.numel()))
    convert[(triton.cdiv(n, 4096),)](raw, planes, n, 4096, False)
    plainplanes.copy_(planes)
    back = torch.empty_like(raw)
    convert[(triton.cdiv(n, 4096),)](back, planes, n, 4096, True)
    assert torch.equal(raw, back)
    del back
    funcs = {
        "native_compressed": lambda: read[(triton.cdiv(n, 4096),)](
            raw, planes, o, n, 4096, False
        ),
        "planes_compressed": lambda: read[(triton.cdiv(n, 4096),)](
            raw, planes, o, n, 4096, True
        ),
        "planes_plain": lambda: read[(triton.cdiv(n, 4096),)](
            raw, plainplanes, o, n, 4096, True
        ),
        "producer": lambda: convert[(triton.cdiv(n, 4096),)](
            raw, planes, n, 4096, False
        ),
    }
    graphs = {}
    for key, fn in funcs.items():
        fn()
        torch.cuda.synchronize()
        if key == "native_compressed":
            ref = o.clone()
        elif key != "producer":
            assert torch.equal(ref, o)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        graphs[key] = g
    samples = {key: [] for key in funcs}
    for rep in range(4):
        for key in list(funcs)[:: 1 if rep % 2 == 0 else -1]:
            g = graphs[key]
            for _ in range(2):
                g.replay()
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(5):
                g.replay()
            b.record()
            b.synchronize()
            samples[key].append(a.elapsed_time(b) / 5)
    result = dict(
        bitwise_roundtrip=True,
        checksum_exact=True,
        ms=samples,
        median_ms={k: statistics.median(v) for k, v in samples.items()},
    )
    report["cases"][case] = result
    print(case, json.dumps(result), flush=True)
    Path("/root/qvl/experiments/byteplanes/report.json").write_text(
        json.dumps(report, indent=2)
    )
