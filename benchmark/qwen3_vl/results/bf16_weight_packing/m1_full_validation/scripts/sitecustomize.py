import json
import os
import pathlib

if os.getenv("QVL_M1_MARKER"):
    import torch
    from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
        default_tactic,
        run_direct_dense,
    )

    import sglang.srt.layers.quantization.unquant as uq
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    path = pathlib.Path(os.environ["QVL_M1_MARKER"])
    original = uq._bf16_gemm_dispatch_impl
    execute_original = DecodeCudaGraphRunner.execute
    weights = set()
    seen_replay = False
    capture_count = 0

    def emit(d):
        with path.open("a") as f:
            f.write(json.dumps(d) + "\n")

    def dispatch(x, w, bias, addend=None):
        global capture_count
        if (
            os.getenv("QVL_M1_CANDIDATE") == "1"
            and x.numel() == 2560
            and tuple(w.shape) == (19456, 2560)
            and bias is None
            and addend is None
        ):
            assert (
                x.dtype == w.dtype == torch.bfloat16
                and x.is_contiguous()
                and w.is_contiguous()
            )
            y = torch.empty(1, 19456, device=x.device, dtype=x.dtype)
            run_direct_dense(
                x.reshape(1, 2560), w.T, y, False, default_tactic(1, 19456, 2560)
            )
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                capture_count += 1
            if w.data_ptr() not in weights:
                weights.add(w.data_ptr())
                emit(
                    {
                        "kind": "m1_gateup",
                        "unique_layers": len(weights),
                        "capturing": capturing,
                    }
                )
            return y.view(*x.shape[:-1], 19456)
        return original(x, w, bias, addend)

    def execute(self, fb, *args, **kw):
        global seen_replay
        y = execute_original(self, fb, *args, **kw)
        if self.bs == 1 and not seen_replay:
            seen_replay = True
            emit(
                {
                    "kind": "graph1_replay",
                    "raw_batch": int(fb.batch_size),
                    "unique_weights": len(weights),
                    "capture_dispatches": capture_count,
                }
            )
        return y

    uq._bf16_gemm_dispatch_impl = dispatch
    DecodeCudaGraphRunner.execute = execute
