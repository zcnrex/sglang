import atexit
import json
import os
import pathlib

import numpy as np
import torch

import sglang.srt.layers.quantization.unquant as uq
from sglang.srt.model_executor.model_runner import ModelRunner

root = pathlib.Path("/root/qvl/experiments/decode-lt")
variant = os.environ["QVL_VARIANT"]
out = root / variant
out.mkdir(exist_ok=True)
checks = []
captures = {}
graph_flags = []
dispatch_counts = {}
profiled = False
runner = None
workspace = None
raw = None
handle = None
original = uq._bf16_gemm_dispatch_impl
T = {(128, 19456, 2560): 1, (128, 2560, 9728): 4}


def dispatch(x, w, bias, addend=None):
    global runner, workspace, raw, handle
    shape = (x.numel() // x.shape[-1], w.shape[0], w.shape[1])
    if (
        variant.startswith("candidate")
        and shape in T
        and x.dtype == torch.bfloat16
        and bias is None
        and addend is None
    ):
        dispatch_counts[str(shape)] = dispatch_counts.get(str(shape), 0) + 1
        if runner is None:
            from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module
            from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

            raw = gen_mm_bf16_cublaslt_module().build_and_load()
            handle = torch.cuda.current_blas_handle()
            runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
            workspace = torch.empty(
                40 * 1024 * 1024, dtype=torch.uint8, device=x.device
            )
        y = torch.empty(shape[0], shape[1], dtype=x.dtype, device=x.device)
        if shape == (128, 2560, 9728) and os.getenv("QVL_TGV_DOWN") == "1":
            from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import _run_tgv

            _run_tgv(x.view(-1, x.shape[-1]), w.T, None, y, True, 23)
        else:
            algos, _ = runner._get_algos(
                [x.view(-1, x.shape[-1]), w.T, None, False, y, workspace]
            )
            raw.mm_bf16_cublaslt_run_with_algo(
                x.view(-1, x.shape[-1]), w, None, y, workspace, handle, algos, T[shape]
            )
        if shape not in seen:
            ref = torch.nn.functional.linear(x, w, bias).reshape_as(y)
            torch.testing.assert_close(y, ref, rtol=0.02, atol=0.02)
            diff = y.float() - ref.float()
            checks.append(
                {
                    "shape": shape,
                    "tactic": T[shape],
                    "bitwise": torch.equal(y, ref),
                    "nrms": (
                        diff.square().mean().sqrt() / ref.float().square().mean().sqrt()
                    ).item(),
                    "max_abs": diff.abs().max().item(),
                }
            )
            seen.add(shape)
        return y.view(*x.shape[:-1], shape[1])
    return original(x, w, bias, addend)


seen = set()
uq._bf16_gemm_dispatch_impl = dispatch
orig_forward = ModelRunner.forward


def forward(self, batch, *a, **kw):
    global profiled
    if batch.forward_mode.is_decode() and not profiled:
        profiled = True
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            r = orig_forward(self, batch, *a, **kw)
        prof.export_chrome_trace(str(out / "warm_decode_trace.json"))
    else:
        r = orig_forward(self, batch, *a, **kw)
    if batch.forward_mode.is_decode():
        graph_flags.append(bool(r.can_run_graph))
    key = batch.forward_mode.name + "_" + str(batch.input_ids.numel())
    if key not in captures:
        captures[key] = r.logits_output.next_token_logits.detach().clone()
    return r


ModelRunner.forward = forward


def save():
    (out / "graph_verification.json").write_text(
        json.dumps(
            {
                "decode_graph_flags": graph_flags,
                "capture_dispatch_counts": dispatch_counts,
                "profiled_warmup_only": profiled,
            },
            indent=2,
        )
    )
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    torch.save({k: v.cpu() for k, v in captures.items()}, out / "logits.pt")


atexit.register(save)
np.random.seed(42)
torch.manual_seed(42)
from sglang.benchmark.one_batch import cli_main

cli_main()
