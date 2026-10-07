import os

if os.getenv("QVL_LT_PATCH") == "1":
    import json
    import pathlib
    import time

    import flashinfer
    import torch
    from flashinfer.autotuner import AutoTuner
    from flashinfer.gemm import mm_bf16

    import sglang.srt.layers.quantization.unquant as uq
    from sglang.srt.model_executor.model_runner import ModelRunner

    original_dispatch = uq._bf16_gemm_dispatch_impl
    original_graph_init = ModelRunner.init_cuda_graphs
    shapes = {(128, 19456, 2560), (128, 2560, 9728)}
    tuned = False

    def initialize(self, *args, **kwargs):
        global tuned
        if not tuned:
            outdir = pathlib.Path(os.environ["QVL_EVIDENCE_DIR"])
            records = []
            for m, n, k in sorted(shapes):
                matches = [
                    (name, w)
                    for name, w in self.model.named_parameters()
                    if tuple(w.shape) == (n, k)
                ]
                assert matches, (n, k)
                name, w = matches[0]
                assert w.dtype == torch.bfloat16 and w.is_cuda and w.is_contiguous()
                generator = torch.Generator(device=w.device).manual_seed(42)
                x = (
                    torch.randn(
                        m, k, device=w.device, dtype=torch.bfloat16, generator=generator
                    )
                    * 0.1
                )
                y = torch.empty(m, n, device=w.device, dtype=torch.bfloat16)
                start = time.monotonic()
                with (
                    torch.inference_mode(),
                    flashinfer.autotune(tuning_buckets=(128,), round_up=False),
                ):
                    mm_bf16(x, w.T, out=y, backend="cublaslt")
                mm_bf16(x, w.T, out=y, backend="cublaslt")
                ref = torch.nn.functional.linear(x, w)
                torch.testing.assert_close(y, ref, rtol=0.02, atol=0.02)
                records.append(
                    {
                        "shape": [m, n, k],
                        "weight_name": name,
                        "weight_dtype": str(w.dtype),
                        "bitwise_vs_torch": torch.equal(y, ref),
                        "max_abs": (y - ref).abs().max().item(),
                        "nrms": (
                            (y.float() - ref.float()).square().mean()
                            / ref.float().square().mean()
                        )
                        .sqrt()
                        .item(),
                        "tuning_seconds": time.monotonic() - start,
                    }
                )
            AutoTuner.get().save_configs(str(outdir / "autotune-cache.json"))
            (outdir / "startup_tuning.json").write_text(json.dumps(records, indent=2))
            tuned = True
        return original_graph_init(self, *args, **kwargs)

    def dispatch(x, w, bias, addend=None):
        shape = (x.numel() // x.shape[-1], w.shape[0], w.shape[1])
        if (
            shape not in shapes
            or x.dtype != torch.bfloat16
            or bias is not None
            or addend is not None
        ):
            return original_dispatch(x, w, bias, addend)
        assert tuned, "Public BF16 tactics must be tuned before CUDA graph capture"
        out = torch.empty(shape[0], shape[1], device=x.device, dtype=x.dtype)
        mm_bf16(x.view(-1, x.shape[-1]), w.T, out=out, backend="cublaslt")
        return out.view(*x.shape[:-1], shape[1])

    ModelRunner.init_cuda_graphs = initialize
    uq._bf16_gemm_dispatch_impl = dispatch
