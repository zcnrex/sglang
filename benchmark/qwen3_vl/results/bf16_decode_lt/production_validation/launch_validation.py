import runpy

import torch

from sglang.srt.layers.quantization import unquant
from sglang.srt.model_executor.runner.base_runner import BaseRunner

old = BaseRunner.warmup
old_try = unquant._try_tuned_bf16_cublaslt
seen = set()


def attempt(x, w, *a, **kw):
    out = old_try(x, w, *a, **kw)
    if (
        out is not None
        and (tuple(w.shape), torch.cuda.is_current_stream_capturing()) not in seen
    ):
        seen.add((tuple(w.shape), torch.cuda.is_current_stream_capturing()))
        print(
            "PUBLIC_LT_DISPATCH",
            tuple(w.shape),
            "capture",
            torch.cuda.is_current_stream_capturing(),
            flush=True,
        )
    return out


unquant._try_tuned_bf16_cublaslt = attempt


def warm(self, *a, **kw):
    old(self, *a, **kw)
    print("PUBLIC_LT_READY", sorted(unquant._CUBLASLT_BF16_READY), flush=True)
    assert len(unquant._CUBLASLT_BF16_READY) == 2


BaseRunner.warmup = warm
if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__")
