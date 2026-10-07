import json
import os
import pathlib

if os.getenv("QVL_M1_MARKER"):
    import torch

    import sglang.srt.layers.quantization.unquant as u
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    path = pathlib.Path(os.environ["QVL_M1_MARKER"])
    weights = set()
    capture_count = 0
    seen = False
    origexec = DecodeCudaGraphRunner.execute

    def emit(d):
        with path.open("a") as f:
            f.write(json.dumps(d) + "\n")

    if hasattr(u, "_try_m1_bf16_direct"):
        original = u._try_m1_bf16_direct

        def observe(x, w, *a, **kw):
            global capture_count
            y = original(x, w, *a, **kw)
            if y is not None:
                weights.add(w.data_ptr())
                if torch.cuda.is_current_stream_capturing():
                    capture_count += 1
            return y

        u._try_m1_bf16_direct = observe

    def execute(self, fb, *a, **kw):
        global seen
        y = origexec(self, fb, *a, **kw)
        if self.bs == 1 and not seen:
            seen = True
            emit(
                {
                    "kind": "graph1_replay",
                    "raw_batch": int(fb.batch_size),
                    "distinct_weight_buffers_including_precompile": len(weights),
                    "capture_calls": capture_count,
                }
            )
        return y

    DecodeCudaGraphRunner.execute = execute
