import json
import os
import pathlib
import runpy

import torch

runpy.run_path("/root/qvl/experiments/lmhead-production-observer/sitecustomize.py")
from sglang.srt.layers.quantization import unquant
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)

root = pathlib.Path(os.environ["QVL_LMHEAD_OUT"])
pairs = []
steps = 0
original = unquant._try_tuned_bf16_cublaslt


def dispatch(x, w, *a, **kw):
    y = original(x, w, *a, **kw)
    if (
        y is not None
        and tuple(x.shape) == (4, 2560)
        and tuple(w.shape) == (151936, 2560)
    ):
        ref = torch.mm(x, w.T)
        if torch.cuda.is_current_stream_capturing():
            pairs.append((y, ref))
    return y


unquant._try_tuned_bf16_cublaslt = dispatch
execute = DecodeCudaGraphRunner.execute


def replay(self, *a, **kw):
    global steps
    y = execute(self, *a, **kw)
    if getattr(self, "bs", None) == 4 and (root / "image-active").exists():
        assert pairs
        candidate, ref = pairs[-1]
        diff = candidate.float() - ref.float()
        record = {
            "step": steps,
            "bitwise": torch.equal(candidate, ref),
            "max_abs": diff.abs().max().item(),
            "nrmse": (diff.norm() / ref.float().norm()).item(),
            "argmax_equal": torch.equal(candidate.argmax(-1), ref.argmax(-1)),
        }
        with (root / "same-input-checks.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        steps += 1
    return y


DecodeCudaGraphRunner.execute = replay
