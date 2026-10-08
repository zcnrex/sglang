import json
import os
import pathlib
import types

import torch

os.environ["QVL_NATIVE_OUT"] = "/root/qvl/experiments/mrope-mixed-check"
os.environ["QVL_NATIVE_VARIANT"] = "candidate"
pathlib.Path(os.environ["QVL_NATIVE_OUT"]).mkdir(exist_ok=True)
import mrope_serving_v2_hook as h

import sglang.srt.model_executor.forward_batch_info as mod
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    compute_position,
)

mod.get_exec = h.get_exec = lambda: types.SimpleNamespace(
    deterministic=types.SimpleNamespace(rl_on_policy_target=None)
)
g = json.load(open("/tmp/native_model_geometry.json"))
pre = g["prefix_lens"]
lens = g["extend_lens"]
m = sum(lens)
p, start = compute_position(
    "trtllm_mha",
    torch.tensor(pre, device="cuda", dtype=torch.int32),
    torch.tensor(lens, device="cuda", dtype=torch.int32),
    m,
)
f = ForwardBatch(
    ForwardMode.MIXED,
    42,
    torch.zeros(m, device="cuda", dtype=torch.int64),
    torch.arange(42, device="cuda"),
    torch.tensor([a + b for a, b in zip(pre, lens)], device="cuda"),
    torch.arange(m, device="cuda"),
    sum(a + b for a, b in zip(pre, lens)),
)
f.positions = p
f.seq_lens_cpu = f.seq_lens.cpu()
b = types.SimpleNamespace(
    multimodal_inputs=[None] * 42,
    prefix_lens=pre,
    extend_lens=lens,
    reqs=[types.SimpleNamespace(session=None)] * 42,
    dllm_config=None,
)
runner = types.SimpleNamespace(device="cuda")
h.orig(f, runner, b)
ref = f.mrope_positions.clone()
h.mrope(f, runner, b)
assert torch.equal(ref, f.mrope_positions)
assert h.counts["fast"] == 1
# Custom-position signal must use original rebuilding, even if positions disagree.
h.state["custom_positions"] = True
f.positions = p + 17
h.mrope(f, runner, b)
assert torch.equal(ref, f.mrope_positions)
assert h.counts["fast"] == 1
print(
    json.dumps(
        {
            "actual_forward_batch_type": type(f).__name__,
            "mode": "MIXED",
            "rows": m,
            "prefixes": pre,
            "extend_lens": lens,
            "positions_kernel": "compute_position(trtllm_mha)",
            "bitwise": True,
            "custom_position_fallback": True,
            "counts": dict(h.counts),
        }
    )
)
