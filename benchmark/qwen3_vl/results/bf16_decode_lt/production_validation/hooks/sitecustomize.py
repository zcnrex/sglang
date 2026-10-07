import json
import os
from pathlib import Path

if os.getenv("QVL_ACCURACY_MARKER"):
    import flashinfer.gemm

    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    output = Path(os.environ["QVL_ACCURACY_MARKER"])
    original_mm = flashinfer.gemm.mm_bf16
    original_execute = DecodeCudaGraphRunner.execute
    seen = set()

    def record(kind, **kwargs):
        key = (kind, str(kwargs))
        if key not in seen:
            seen.add(key)
            with output.open("a") as f:
                f.write(json.dumps({"kind": kind, **kwargs}) + "\n")

    def mm(*args, **kwargs):
        a, b = args[:2]
        if tuple(a.shape) in {(128, 2560), (128, 9728)}:
            record(
                "public_mm",
                a=list(a.shape),
                b=list(b.shape),
                backend=kwargs.get("backend"),
            )
        return original_mm(*args, **kwargs)

    def execute(self, forward_batch, *args, **kwargs):
        result = original_execute(self, forward_batch, *args, **kwargs)
        if self.bs == 128:
            record("decode_graph_128", raw_batch=int(forward_batch.batch_size))
        return result

    flashinfer.gemm.mm_bf16 = mm
    DecodeCudaGraphRunner.execute = execute
