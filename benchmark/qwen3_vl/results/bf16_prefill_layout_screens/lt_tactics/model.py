import atexit
import json
import os
import pathlib

import numpy as np
import torch

import sglang.srt.layers.quantization.unquant as uq
from sglang.srt.model_executor.model_runner import ModelRunner

root = pathlib.Path("/root/qvl/experiments/cublaslt-tactics")
variant = os.environ["QVL_VARIANT"]
out = root / variant
out.mkdir(exist_ok=True)
checks = []
captures = {}
runner = None
workspace = None
original = uq._bf16_gemm_dispatch_impl
T = {
    (8192, 6144, 2560): 2,
    (8192, 2560, 4096): 2,
    (8192, 2560, 9728): 6,
    (16331, 2560, 9728): 2,
}


def dispatch(x, w, bias, addend=None):
    global runner, workspace
    shape = (x.numel() // x.shape[-1], w.shape[0], w.shape[1])
    if (
        variant.startswith("candidate")
        and shape in T
        and x.dtype == torch.bfloat16
        and bias is None
        and addend is None
    ):
        if runner is None:
            from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module

            runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
            workspace = torch.empty(
                32 * 1024 * 1024, dtype=torch.uint8, device=x.device
            )
        y = torch.empty(shape[0], shape[1], dtype=x.dtype, device=x.device)
        runner.forward(
            [x.view(-1, x.shape[-1]), w.T, None, False, y, workspace], tactic=T[shape]
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
    r = orig_forward(self, batch, *a, **kw)
    if batch.forward_mode.is_extend() and batch.input_ids.numel() not in captures:
        captures[batch.input_ids.numel()] = (
            r.logits_output.next_token_logits.detach().clone()
        )
    return r


ModelRunner.forward = forward


def save():
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    torch.save({k: v.cpu() for k, v in captures.items()}, out / "logits.pt")


atexit.register(save)
np.random.seed(42)
torch.manual_seed(42)
from sglang.benchmark.one_batch import cli_main

cli_main()
