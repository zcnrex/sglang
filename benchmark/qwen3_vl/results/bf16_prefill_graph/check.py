import atexit
import copy
import json
import os
import pathlib

import numpy as np
import torch

from sglang.srt.configs import model_config

model_config.multimodal_breakable_cuda_graph_supported_model_archs.append(
    "Qwen3VLForConditionalGeneration"
)
from sglang.benchmark import one_batch as bench
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)

root = pathlib.Path(os.environ["QVL_GRAPH_OUT"])
root.mkdir(parents=True, exist_ok=True)
state = {"graph": False, "call": 0}
events = []
captures = {}
results = []
original_execute = PrefillCudaGraphRunner.execute


def execute(self, *a, **kw):
    events.append({"executed": True})
    return original_execute(self, *a, **kw)


PrefillCudaGraphRunner.execute = execute
original_can = PrefillCudaGraphRunner.can_run_graph


def can(self, batch):
    eligible = not batch.contains_mm_inputs() and original_can(self, batch)
    events.append(
        {
            "tokens": len(batch.input_ids),
            "batch": batch.batch_size,
            "eligible": eligible,
            "enabled": state["graph"],
            "buckets": self.capture_num_tokens,
        }
    )
    return state["graph"] and eligible


PrefillCudaGraphRunner.can_run_graph = can
orig_forward = ModelRunner.forward


def forward(self, batch, *a, **kw):
    r = orig_forward(self, batch, *a, **kw)
    if state["call"] == 0 and batch.forward_mode.is_extend():
        key = "graph" if state["graph"] else "eager"
        captures[key] = r.logits_output.next_token_logits.detach().clone()
    return r


ModelRunner.forward = forward
orig_once = bench.latency_test_run_once


def once(*a, **kw):
    variants = (
        [False, True, False, True] if state["call"] == 0 else [bool(state["call"] % 2)]
    )
    for v in variants:
        state["graph"] = v
        aa = list(a)
        aa[3] = copy.deepcopy(a[3])
        if os.environ.get("QVL_HET"):
            for i, req in enumerate(aa[3]):
                n = 8200 if i == 0 else (8007 if i == 1 else 1)
                req.origin_input_ids = req.origin_input_ids[:n]
                req.full_untruncated_fill_ids = req.origin_input_ids
                req.set_extend_range(0, n)
        r = orig_once(*aa, **kw)
    r["graph"] = state["graph"]
    results.append(r.copy())
    state["call"] += 1
    return r


bench.latency_test_run_once = once


def save():
    report = {"events": events, "results": results}
    if len(captures) == 2:
        a, b = captures.values()
        report["check"] = {
            "bitwise": torch.equal(a, b),
            "max_abs": (a - b).abs().max().item(),
            "greedy_equal": torch.equal(a.argmax(-1), b.argmax(-1)),
        }
    (root / "report.json").write_text(json.dumps(report, indent=2, default=str))


atexit.register(save)
np.random.seed(42)
torch.manual_seed(42)
bench.cli_main()
