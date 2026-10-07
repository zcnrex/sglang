import atexit
import copy
import json
import pathlib

import numpy as np
import torch

import sglang.srt.layers.quantization.unquant as uq
from sglang.benchmark import one_batch as bench
from sglang.srt.model_executor.model_runner import ModelRunner

root = pathlib.Path("/root/qvl/experiments/cublaslt-persistent")
out = root / "model_direct"
out.mkdir(exist_ok=True)
state = {"candidate": False, "calls": 0, "phase": "warmup"}
checks = []
captures = {}
cache = {}
runner = None
raw = None
workspace = None
handle = None
original = uq._bf16_gemm_dispatch_impl
T = {(8192, 6144, 2560): 2, (8192, 2560, 4096): 2, (8192, 2560, 9728): 6}


def dispatch(x, w, bias, addend=None):
    global runner, raw, workspace, handle
    shape = (x.numel() // x.shape[-1], w.shape[0], w.shape[1])
    if (
        state["candidate"]
        and shape in T
        and x.dtype == torch.bfloat16
        and bias is None
        and addend is None
    ):
        if runner is None:
            from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module
            from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

            runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
            raw = gen_mm_bf16_cublaslt_module().build_and_load()
            workspace = torch.empty(
                32 * 1024 * 1024, dtype=torch.uint8, device=x.device
            )
            handle = torch.cuda.current_blas_handle()
        y = torch.empty(shape[0], shape[1], dtype=x.dtype, device=x.device)
        if shape not in cache:
            cache[shape] = runner._get_algos(
                [x.view(-1, x.shape[-1]), w.T, None, False, y, workspace]
            )[0]
        raw.mm_bf16_cublaslt_run_with_algo(
            x.view(-1, x.shape[-1]),
            w,
            None,
            y,
            workspace,
            handle,
            cache[shape],
            T[shape],
        )
        if shape not in seen:
            ref = torch.nn.functional.linear(x, w, bias).reshape_as(y)
            torch.testing.assert_close(y, ref, rtol=0.02, atol=0.02)
            checks.append(
                {
                    "shape": shape,
                    "tactic": T[shape],
                    "bitwise": torch.equal(y, ref),
                    "max_abs": (y - ref).abs().max().item(),
                }
            )
            seen.add(shape)
        return y.view(*x.shape[:-1], shape[1])
    return original(x, w, bias, addend)


seen = set()
uq._bf16_gemm_dispatch_impl = dispatch
orig_forward = ModelRunner.forward


def forward(self, batch, *a, **kw):
    r = orig_forward(self, batch, *a, **kw)
    key = "candidate" if state["candidate"] else "control"
    if (
        state["phase"] == "warmup"
        and batch.forward_mode.is_extend()
        and key not in captures
    ):
        captures[key] = r.logits_output.next_token_logits.detach().clone()
    return r


ModelRunner.forward = forward
original_once = bench.latency_test_run_once
measurements = []


def once(*args, **kwargs):
    if state["calls"] == 0:
        # Same original prompts for both numerical captures and repeated common warmup.
        for variant in [False, True, False, True, False, True]:
            state["candidate"] = variant
            new = list(args)
            new[3] = copy.deepcopy(args[3])
            result = original_once(*new, **kwargs)
        state["calls"] += 1
        state["phase"] = "measure"
        return result
    state["candidate"] = bool(state["calls"] % 2 == 0)
    result = original_once(*args, **kwargs)
    result["variant"] = "candidate" if state["candidate"] else "control"
    result["pair"] = (state["calls"] - 1) // 2
    measurements.append(result.copy())
    state["calls"] += 1
    return result


bench.latency_test_run_once = once


def save():
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    torch.save({k: v.cpu() for k, v in captures.items()}, out / "logits.pt")
    (out / "paired_results.json").write_text(json.dumps(measurements, indent=2))
    if len(captures) == 2:
        a = captures["candidate"]
        b = captures["control"]
        (out / "logits_check.json").write_text(
            json.dumps(
                {"bitwise": torch.equal(a, b), "max_abs": (a - b).abs().max().item()},
                indent=2,
            )
        )


atexit.register(save)
np.random.seed(42)
torch.manual_seed(42)
bench.cli_main()
