import os

if os.getenv("QVL_LT_PATCH") == "1":
    import json
    import pathlib

    import torch

    import sglang.srt.layers.quantization.unquant as uq

    original = uq._bf16_gemm_dispatch_impl
    runner = raw = workspace = handle = None
    cache = {}
    seen = set()
    tactics = {(128, 19456, 2560): 1, (128, 2560, 9728): 4}

    def dispatch(x, w, bias, addend=None):
        global runner, raw, workspace, handle
        shape = (x.numel() // x.shape[-1], w.shape[0], w.shape[1])
        if (
            shape not in tactics
            or x.dtype != torch.bfloat16
            or bias is not None
            or addend is not None
        ):
            return original(x, w, bias, addend)
        if runner is None:
            from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module
            from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

            runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
            raw = gen_mm_bf16_cublaslt_module().build_and_load()
            workspace = torch.empty(
                40 * 1024 * 1024, device=x.device, dtype=torch.uint8
            )
            handle = torch.cuda.current_blas_handle()
        out = torch.empty(shape[0], shape[1], device=x.device, dtype=x.dtype)
        a = x.view(-1, x.shape[-1])
        if shape not in cache:
            cache[shape] = runner._get_algos([a, w.T, None, False, out, workspace])[0]
        raw.mm_bf16_cublaslt_run_with_algo(
            a, w, None, out, workspace, handle, cache[shape], tactics[shape]
        )
        if shape not in seen and not torch.cuda.is_current_stream_capturing():
            ref = torch.nn.functional.linear(a, w)
            torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)
            record = {
                "shape": shape,
                "tactic": tactics[shape],
                "bitwise": torch.equal(out, ref),
                "max_abs": (out - ref).abs().max().item(),
                "workspace_mib": 40,
                "pid": os.getpid(),
            }
            with (
                pathlib.Path(os.environ["QVL_EVIDENCE_DIR"]) / "hook-checks.jsonl"
            ).open("a") as f:
                f.write(json.dumps(record) + "\n")
            seen.add(shape)
        return out.view(*x.shape[:-1], shape[1])

    uq._bf16_gemm_dispatch_impl = dispatch
