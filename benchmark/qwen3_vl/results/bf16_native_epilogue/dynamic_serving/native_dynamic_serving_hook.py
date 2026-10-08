import hashlib
import json
import os
import pathlib
import sys
import time

import torch
from flashinfer.autotuner import AutoTuner

from sglang.srt.layers.quantization import unquant
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen2 import Qwen2MLP

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
state = {
    "ready": False,
    "epoch": 0,
    "prefill": False,
    "mm": False,
    "calls": 0,
    "batch": None,
}
indices = {}
weights = {}
scratch = None
op = None
path = root / f"native-context-pid{os.getpid()}.jsonl"


def write(record):
    record.update(pid=os.getpid(), time_ns=time.time_ns(), epoch=state["epoch"])
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


init_original = ModelRunner.init_cuda_graphs


def initialize(self, *args, **kwargs):
    global scratch, op
    if not state["ready"]:
        source = pathlib.Path("/root/qvl/sglang-prefix-production")
        expected = json.load(
            open("/root/qvl/experiments/prefix-production/candidate_expected.json")
        )
        actual = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source / "python").rglob("*.py")
        }
        assert actual == expected
        modules = [
            m
            for m in self.model.modules()
            if isinstance(m, Qwen2MLP)
            and tuple(m.gate_up_proj.weight.shape) == (19456, 2560)
        ]
        assert len(modules) == 36
        for i, m in enumerate(modules):
            indices[id(m)] = i
        info = {
            "candidate": candidate,
            "python_files_verified": len(actual),
            "compile_count": 0,
            "weight_transforms": 0,
            "weight_bytes": 0,
            "scratch_bytes": 0,
        }
        t = time.perf_counter()
        if candidate:
            sys.path.insert(0, "/root/qvl/experiments/native-epilogue-dynamic-m")
            import native_dynamic_m

            assert (
                hashlib.sha256(
                    pathlib.Path(native_dynamic_m.__file__).read_bytes()
                ).hexdigest()
                == "952741f3d6239e5c2c2dbd1037b39828ad4416f9a8461d3af2fce85384e5d924"
            )
            assert (
                hashlib.sha256(
                    pathlib.Path(
                        "/root/qvl/experiments/native-epilogue-dynamic-m/native_fused_ffi.py"
                    ).read_bytes()
                ).hexdigest()
                == "0a0897443d7a4f41361ef24a030ea6356dd6e406d510d5c3f419542bfaf19b30"
            )
            assert (
                hashlib.sha256(
                    pathlib.Path(
                        "/root/qvl/experiments/native-epilogue-dynamic-m/native_epi_subtile64.py"
                    ).read_bytes()
                ).hexdigest()
                == "68b7746ce1e6a785124cf2fc86d8de732e754419874c88a2c26b2d221a794d6f"
            )
            assert torch.cuda.get_device_capability() == (10, 3)
            for m in modules:
                w = m.gate_up_proj.weight
                assert w.dtype == torch.bfloat16 and w.is_cuda
                wi = (
                    torch.stack((w[:9728], w[9728:]), 1)
                    .reshape(19456, 2560)
                    .contiguous()
                )
                weights[id(m)] = (wi, wi.t().unsqueeze(0))
            scratch = torch.empty(
                16384 * 19456,
                device=modules[0].gate_up_proj.weight.device,
                dtype=torch.bfloat16,
            )
            op = native_dynamic_m.DynamicGateUp()
            x = torch.zeros(
                (1, 16384, 2560), device=scratch.device, dtype=scratch.dtype
            )
            for m in [8192, 16331, 16384]:
                op(
                    x.flatten()[: m * 2560].view(1, m, 2560),
                    next(iter(weights.values()))[1],
                    scratch[: m * 19456].view(1, m, 19456),
                )
            torch.cuda.synchronize()
            del x
            info.update(
                compile_count=op.compile_count,
                weight_transforms=len(weights),
                weight_bytes=sum(w[0].numel() * 2 for w in weights.values()),
                scratch_bytes=scratch.numel() * 2,
                scratch_ptr=scratch.data_ptr(),
            )
        info["setup_seconds"] = time.perf_counter() - t
        state["ready"] = True
    else:
        info = {"candidate": candidate, "reinitialization": True}
    result = init_original(self, *args, **kwargs)
    assert len(seen) == 2 and AutoTuner.search_cache == original_search
    info.update(
        startup_tactics=startup_calls,
        ready=sorted(unquant._CUBLASLT_BF16_READY),
        captured_lt_shapes=sorted(seen),
        startup_policy_restored=True,
    )
    (root / f"native-startup-pid{os.getpid()}.json").write_text(
        json.dumps(info, indent=2)
    )
    return result


ModelRunner.init_cuda_graphs = initialize
forward_original = ModelRunner.forward


def forward(self, b, *a, **kw):
    old = (state["prefill"], state["mm"])
    state["prefill"] = b.forward_mode.is_extend()
    state["mm"] = b.contains_mm_inputs()
    try:
        return forward_original(self, b, *a, **kw)
    finally:
        state["prefill"], state["mm"] = old


ModelRunner.forward = forward
original = Qwen2MLP.forward


def mlp(self, x, forward_batch=None):
    index = indices.get(id(self))
    if index is None:
        return original(self, x, forward_batch)
    m = x.shape[0]
    eligible = (
        state["ready"]
        and state["prefill"]
        and not state["mm"]
        and x.ndim == 2
        and 8192 <= m <= 16384
        and x.shape[1] == 2560
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
    )
    if index == 0 and state["prefill"]:
        state["batch"] = {
            "kind": "prefill",
            "m": m,
            "eligible": eligible,
            "candidate": candidate,
            "fused_layers": 0,
            "fallback": "none" if eligible else "shape_or_layout_or_multimodal",
        }
    if candidate and eligible:
        assert op.compile_count == 1 and len(weights) == 36
        op(x.unsqueeze(0), weights[id(self)][1], scratch[: m * 19456].view(1, m, 19456))
        activated = scratch[: m * 9728].view(m, 9728)
        result, _ = self.down_proj(activated, forward_batch=forward_batch)
        if state["batch"] is not None:
            state["batch"]["fused_layers"] += 1
    else:
        result = original(self, x, forward_batch)
    if index == 35 and state["prefill"]:
        record = state["batch"]
        assert record is not None and record["fused_layers"] == (
            36 if candidate and eligible else 0
        )
        write(record)
        state["batch"] = None
    return result


Qwen2MLP.forward = mlp
flush_original = Scheduler.flush_cache


def flush(self, *a, **kw):
    result = flush_original(self, *a, **kw)
    state["epoch"] += 1
    write({"kind": "flush_cache", "result": str(result)})
    return result


Scheduler.flush_cache = flush
