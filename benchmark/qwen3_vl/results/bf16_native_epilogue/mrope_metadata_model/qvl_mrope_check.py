import json
import time
import types

import qvl_mrope_fastpath as h
import torch

import sglang.srt.model_executor.forward_batch_info as mod
from sglang.srt.model_executor.forward_batch_info import ForwardMode

saved = mod.get_exec
mod.get_exec = lambda: types.SimpleNamespace(
    deterministic=types.SimpleNamespace(rl_on_policy_target=None)
)
runner = types.SimpleNamespace(device="cuda")
tests = []
cases = [
    ([0], [8192]),
    ([7104, 0, 0] + [8200] * 39, [1096, 8188, 7008] + [1] * 39),
    ([7, 100], [0, 3]),
    ([4], [0]),
    ([3, 9], [1, 1]),
    ([100000], [17]),
]
for pre, lens in cases:
    p = torch.cat(
        [
            torch.arange(a, a + b, device="cuda", dtype=torch.int64)
            for a, b in zip(pre, lens)
        ]
    )
    b = types.SimpleNamespace(
        multimodal_inputs=[None] * len(pre),
        prefix_lens=pre,
        extend_lens=lens,
        reqs=[types.SimpleNamespace(session=None)] * len(pre),
    )
    f = types.SimpleNamespace(
        positions=p,
        seq_lens_cpu=torch.tensor([a + b for a, b in zip(pre, lens)]),
        forward_mode=ForwardMode.EXTEND,
        spec_info=None,
    )
    h.original(f, runner, b)
    ref = f.mrope_positions.clone()
    h.enabled = True
    h.fast(f, runner, b)
    assert torch.equal(ref, f.mrope_positions)
    tests.append({"prefix": pre, "extend": lens, "bitwise": True})
# Nonzero multimodal delta must fall back even when positions appear text-like.
mm = types.SimpleNamespace(
    mrope_positions=torch.arange(30).reshape(3, 10),
    mrope_position_delta=torch.tensor(7),
)
b = types.SimpleNamespace(
    multimodal_inputs=[mm],
    prefix_lens=[2],
    extend_lens=[3],
    reqs=[types.SimpleNamespace(session=None)],
)
f = types.SimpleNamespace(
    positions=torch.arange(2, 5, device="cuda"),
    seq_lens_cpu=torch.tensor([5]),
    forward_mode=ForwardMode.EXTEND,
    spec_info=None,
)
n = h.calls["fallback"]
h.fast(f, runner, b)
assert h.calls["fallback"] == n + 1
assert torch.equal(f.mrope_positions, mm.mrope_positions[:, 2:5].cuda())
# B42 metadata-only alternating eager latency, inclusive device completion.
pre, lens = cases[1]
p = torch.cat([torch.arange(a, a + b, device="cuda") for a, b in zip(pre, lens)])
b = types.SimpleNamespace(
    multimodal_inputs=[None] * 42, prefix_lens=pre, extend_lens=lens, reqs=[]
)
f = types.SimpleNamespace(
    positions=p,
    seq_lens_cpu=torch.tensor([a + b for a, b in zip(pre, lens)]),
    forward_mode=ForwardMode.EXTEND,
    spec_info=None,
)
for _ in range(10):
    h.original(f, runner, b)
    h.fast(f, runner, b)
rows = []
for i in range(8):
    for candidate in [False, True] if i % 2 == 0 else [True, False]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(100):
            (h.fast if candidate else h.original)(f, runner, b)
        torch.cuda.synchronize()
        rows.append({"candidate": candidate, "us": (time.perf_counter() - t) * 1e4})
mod.get_exec = saved
print(
    json.dumps(
        {
            "tests": tests,
            "nonzero_multimodal_fallback": True,
            "timings": rows,
            "calls": h.calls,
        },
        indent=2,
    )
)
