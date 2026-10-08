import hashlib
import json
import os
import pathlib
import time

import torch
from flashinfer.autotuner import AutoTuner

from sglang.srt.layers.quantization import unquant
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.model_runner import ModelRunner

root = pathlib.Path(os.environ["QVL_NATIVE_OUT"])
candidate = os.environ["QVL_NATIVE_VARIANT"] == "candidate"
original_search = AutoTuner.search_cache
startup_calls = []
seen = set()


def pinned_search(self, custom_op, runners, input_shapes, *args, **kwargs):
    eligible = (
        custom_op == "bf16_gemm"
        and tuple(input_shapes[0]) in [(128, 2560), (128, 9728)]
        and tuple(input_shapes[1]) in [(2560, 19456), (9728, 2560)]
    )
    if eligible and self.is_tuning_mode:
        with self._lock:
            previous = self.is_tuning_mode
            try:
                self.is_tuning_mode = False
                result = original_search(
                    self, custom_op, runners, input_shapes, *args, **kwargs
                )
            finally:
                self.is_tuning_mode = previous
        assert result[0] and result[2] == 1, result
        startup_calls.append(
            {
                "weight_shape": list(input_shapes[1]),
                "hit": result[0],
                "tactic": result[2],
            }
        )
        return result
    return original_search(self, custom_op, runners, input_shapes, *args, **kwargs)


AutoTuner.search_cache = pinned_search
old_lt = unquant._try_tuned_bf16_cublaslt


def observe_lt(x, w, *args, **kwargs):
    result = old_lt(x, w, *args, **kwargs)
    if (
        result is not None
        and torch.cuda.is_current_stream_capturing()
        and tuple(w.shape) not in seen
    ):
        seen.add(tuple(w.shape))
        if len(seen) == 2:
            AutoTuner.search_cache = original_search
    return result


unquant._try_tuned_bf16_cublaslt = observe_lt

from collections import Counter

from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    get_exec,
)

state = {"epoch": 0, "custom_positions": False}
counts = Counter()
hist = Counter()


def write(kind):
    if not counts:
        return
    p = root / f"mrope-counts-pid{os.getpid()}.jsonl"
    with p.open("a") as f:
        f.write(
            json.dumps(
                {
                    "kind": kind,
                    "pid": os.getpid(),
                    "epoch": state["epoch"],
                    "counts": dict(counts),
                    "rows_hist": dict(hist),
                    "time_ns": time.time_ns(),
                }
            )
            + "\n"
        )


init_batch = ForwardBatch.init_new.__func__


@classmethod
def init_new(cls, *args, **kwargs):
    previous = state["custom_positions"]
    state["custom_positions"] = kwargs.get("extend_position_info") is not None or (
        len(args) > 5 and args[5] is not None
    )
    try:
        return init_batch(cls, *args, **kwargs)
    finally:
        state["custom_positions"] = previous


ForwardBatch.init_new = init_new
orig = ForwardBatch._compute_mrope_positions_extend


def mrope(self, runner, batch):
    p = self.positions
    mm = batch.multimodal_inputs
    eligible = (
        not state["custom_positions"]
        and self.forward_mode == ForwardMode.EXTEND
        and self.spec_info is None
        and getattr(batch, "dllm_config", None) is None
        and mm is not None
        and all(x is None for x in mm)
        and get_exec().deterministic.rl_on_policy_target is None
        and p is not None
        and p.is_cuda
        and p.dtype == torch.int64
        and p.ndim == 1
        and p.is_contiguous()
        and len(mm) == len(batch.extend_lens) == len(batch.prefix_lens)
        and p.numel() == sum(batch.extend_lens)
    )
    counts["calls"] += 1
    counts["rows"] += p.numel() if p is not None else 0
    hist[str(p.numel() if p is not None else -1)] += 1
    if eligible:
        counts["eligible"] += 1
    if candidate and eligible:
        self.mrope_positions = p.unsqueeze(0).repeat(3, 1)
        counts["fast"] += 1
    else:
        counts["original"] += 1
        if not eligible:
            counts["fallback"] += 1
        orig(self, runner, batch)


ForwardBatch._compute_mrope_positions_extend = mrope
init_original = ModelRunner.init_cuda_graphs


def initialize(self, *args, **kwargs):
    source = pathlib.Path("/root/qvl/sglang-prefix-production")
    expected = json.load(
        open("/root/qvl/experiments/prefix-production/candidate_expected.json")
    )
    actual = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source / "python").rglob("*.py")
    }
    assert actual == expected
    result = init_original(self, *args, **kwargs)
    assert len(seen) == 2 and AutoTuner.search_cache == original_search
    info = {
        "candidate": candidate,
        "python_files_verified": len(actual),
        "compile_count": 0,
        "weight_transforms": 0,
        "startup_tactics": startup_calls,
        "ready": sorted(unquant._CUBLASLT_BF16_READY),
        "captured_lt_shapes": sorted(seen),
        "startup_policy_restored": True,
    }
    (root / f"native-startup-pid{os.getpid()}.json").write_text(
        json.dumps(info, indent=2)
    )
    return result


ModelRunner.init_cuda_graphs = initialize
flush_original = Scheduler.flush_cache


def flush(self, *a, **kw):
    result = flush_original(self, *a, **kw)
    write("flush")
    counts.clear()
    hist.clear()
    state["epoch"] += 1
    return result


Scheduler.flush_cache = flush
import atexit

atexit.register(lambda: write("atexit"))
