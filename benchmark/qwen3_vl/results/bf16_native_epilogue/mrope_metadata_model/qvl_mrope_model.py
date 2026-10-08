import atexit
import copy
import hashlib
import json
import os
import pathlib
import time

import numpy as np
import qvl_mrope_fastpath as mrope
import torch

from sglang.benchmark import one_batch as bench
from sglang.srt.model_executor.model_runner import ModelRunner

source = pathlib.Path("/root/qvl/sglang-prefix-production")
expected = json.load(
    open("/root/qvl/experiments/prefix-production/candidate_expected.json")
)
actual = {
    str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (source / "python").rglob("*.py")
}
assert actual == expected
root = pathlib.Path(os.environ["QVL_MODEL_OUT"])
root.mkdir(parents=True, exist_ok=True)
geometry = json.load(open("/tmp/native_model_geometry.json"))
assert sum(geometry["extend_lens"]) == 16331
state = {"enabled": False, "capture": False, "variant": False}
results = []
captures = {}
weights = {}
kernel = {}
coverage = {}
setup = []
dispatch_counts = {
    "compile": 0,
    "input_descriptor": 0,
    "weight_descriptor": 0,
    "weight_transform": 0,
}
forward = ModelRunner.forward


def forward_hook(self, b, *a, **kw):
    r = forward(self, b, *a, **kw)
    if state["capture"]:
        captures[state["variant"]].append(
            r.logits_output.next_token_logits.detach().clone()
        )
    return r


ModelRunner.forward = forward_hook
from sglang.kernels.ops.attention import dllm_kv_pack

pack_original = dllm_kv_pack.pack_prefix_current
pack_calls = 0


def pack_observer(*a, **kw):
    global pack_calls
    pack_calls += 1
    return pack_original(*a, **kw)


dllm_kv_pack.pack_prefix_current = pack_observer
call = 0
order = [False, True, True, False] * 2
if int(os.environ.get("PAIR_OFFSET", "0")):
    order = [not x for x in order]


def once(*args, **kwargs):
    global call
    variants = [False, True, True, False] if call == 0 else [order[call - 1]]
    runner = args[1]
    for variant in variants:
        mrope.enabled = False
        state.update(enabled=False, capture=False)
        runner.clear()
        reqs = copy.deepcopy(args[3])
        assert len(reqs) == 42
        for i, req in enumerate(reqs):
            n = geometry["prefix_lens"][i] + (
                geometry["extend_lens"][i] if i < 3 else 0
            )
            req.origin_input_ids = req.origin_input_ids[:n]
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.set_extend_range(0, n)
        for start in range(3, 42, 8):
            group = reqs[start : start + 8]
            ids, _, _ = runner.extend(group)
            for index, (req, tok) in enumerate(zip(group, ids.tolist()), start):
                n = geometry["prefix_lens"][index]
                req.prefix_indices = runner.torch_runner.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :n
                ].to(req.prefix_indices.dtype)
                req.full_untruncated_fill_ids.append(tok)
                req.set_extend_range(n, n + 1)
        full = reqs[0].origin_input_ids[:]
        n = 7104
        reqs[0].origin_input_ids = full[:n]
        reqs[0].full_untruncated_fill_ids = reqs[0].origin_input_ids
        reqs[0].set_extend_range(0, n)
        runner.extend([reqs[0]])
        reqs[0].full_untruncated_fill_ids.extend(full[n:])
        reqs[0].prefix_indices = runner.torch_runner.req_to_token_pool.req_to_token[
            reqs[0].kv.req_pool_idx, :n
        ].to(reqs[0].prefix_indices.dtype)
        reqs[0].set_extend_range(n, 8200)
        assert (
            sum(len(req.get_fill_ids()) - len(req.prefix_indices) for req in reqs)
            == 16331
        )
        mrope.enabled = variant
        state.update(enabled=variant, capture=call == 0, variant=variant)
        if call == 0:
            captures[variant] = []
        before = mrope.calls["fast"]
        pack_before = pack_calls
        dispatch_before = dispatch_counts.copy()
        runner.synchronize()
        t = time.perf_counter()
        ids, logits, batch = runner.extend(reqs)
        runner.synchronize()
        dt = time.perf_counter() - t
        invoked = mrope.calls["fast"] - before
        assert invoked == (1 if variant else 0)
        for _ in range(3):
            ids, logits = runner.decode(ids, batch)
        runner.synchronize()
        state["capture"] = False
        r = {
            "candidate": variant,
            "prefill_latency": dt,
            "m": 16331,
            "batch_size": 42,
            "mrope_fast_calls": invoked,
            "packed_fa4_calls": pack_calls - pack_before,
            "dispatch_counts": {
                k: v - dispatch_before[k] for k, v in dispatch_counts.items()
            },
        }
        if call > 0:
            assert all(v == 0 for v in r["dispatch_counts"].values())
    if call > 0:
        results.append(r)
    call += 1
    print("MIXED_RESULT", json.dumps(r), flush=True)
    return r


bench.latency_test_run_once = once


def save():
    checks = []
    if len(captures) == 2:
        for a, b in zip(captures[False], captures[True]):
            checks.append(
                {
                    "bitwise": torch.equal(a, b),
                    "max_abs": (a - b).abs().max().item(),
                    "nrmse": ((a.float() - b.float()).norm() / a.float().norm()).item(),
                    "greedy_equal": torch.equal(a.argmax(-1), b.argmax(-1)),
                }
            )
    (root / "report.json").write_text(
        json.dumps(
            {
                "verified_python_files": len(actual),
                "geometry": geometry,
                "results": results,
                "checks": checks,
                "coverage": mrope.calls,
                "weight_setup": setup,
                "dispatch_counts": dispatch_counts,
                "scratch_bytes": 0,
            },
            indent=2,
        )
    )


atexit.register(save)
np.random.seed(42)
torch.manual_seed(42)
bench.cli_main()
