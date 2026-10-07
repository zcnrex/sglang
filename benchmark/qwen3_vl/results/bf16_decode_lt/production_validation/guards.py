from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import unquant as u
from sglang.srt.model_executor.runner import flashinfer_autotune as f

x = torch.zeros(128, 2560, device="cuda", dtype=torch.bfloat16)
w = torch.zeros(19456, 2560, device="cuda", dtype=torch.bfloat16)
o = torch.empty(128, 19456, device="cuda", dtype=torch.bfloat16)
state = NS(
    kernel=NS(disable_flashinfer_autotune=False),
    deterministic=NS(enable_deterministic_inference=False),
)
backend = NS(is_cutedsl=lambda: True)
u._CUBLASLT_BF16_READY.add((0, 19456, 2560))


def fake(a, b, out=None, **kw):
    return (
        out
        if out is not None
        else torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    )


with (
    patch.object(u, "get_exec", return_value=state),
    patch.object(u, "get_bf16_gemm_backend", return_value=backend),
    patch("flashinfer.gemm.mm_bf16", side_effect=fake) as mm,
):
    assert u._try_tuned_bf16_cublaslt(x, w, None, out=o) is o
    cases = [
        (x, w, torch.zeros(1, device="cuda"), None, None),
        (x, w, None, o, None),
        (x.float(), w, None, None, None),
        (x, w.float(), None, None, None),
        (x[:, ::2], w, None, None, None),
        (x, w, None, None, o.float()),
        (x, w, None, None, o.T),
        (x, w, None, None, o.cpu()),
    ]
    before = mm.call_count
    for a, b, bias, addend, out in cases:
        assert u._try_tuned_bf16_cublaslt(a, b, bias, addend, out) is None
    assert mm.call_count == before
    state.kernel.disable_flashinfer_autotune = True
    assert u._try_tuned_bf16_cublaslt(x, w, None) is None
    state.kernel.disable_flashinfer_autotune = False
    state.deterministic.enable_deterministic_inference = True
    assert u._try_tuned_bf16_cublaslt(x, w, None) is None
    state.deterministic.enable_deterministic_inference = False
    u._CUBLASLT_BF16_READY.clear()
    assert u._try_tuned_bf16_cublaslt(x, w, None) is None
for spec, draft in [(True, False), (False, True)]:
    mr = NS(
        device="cuda",
        dtype=torch.bfloat16,
        model_config=NS(quantization=None),
        is_draft_worker=draft,
        spec_algorithm=NS(is_speculative=lambda: spec),
    )
    assert f._bf16_cublaslt_weights(mr) == {}
with patch.object(u, "get_exec", return_value=NS(kernel=NS(bf16_gemm_backend="torch"))):
    u._CUBLASLT_BF16_READY.add((0, 19456, 2560))
    u.initialize_bf16_gemm_config()
    assert not u._CUBLASLT_BF16_READY
print(
    "PASS guards: identity bias addend dtype layout output-device disabled deterministic unready spec draft reinitialize"
)
