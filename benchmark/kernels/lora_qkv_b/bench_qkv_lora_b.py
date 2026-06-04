"""qkv_lora_b testbed: general Triton kernel vs single-adapter specializations.

Built on the PR #27292 ``marker.Kernel`` contract (correctness-aware, seeded, cold
by default, measured roofline). Four impls share one input set per combo:

  - ``torch``         : fp32 reference (ground truth for ``assert_close``), cast to bf16.
  - ``triton``        : ``qkv_lora_b_fwd`` with ``uniform_weight_index`` unset, so it
                        always takes the general Triton kernel.
  - ``cublas``        : ``_qkv_lora_b_cublas`` called DIRECTLY -- one cuBLAS ``addmm_``
                        per q/k/v slice for a single rank-``r`` adapter.
  - ``triton_single`` : ``qkv_lora_b_single_fwd`` -- a Triton kernel specialized for the
                        single-adapter case: no weight_indices/lora_ranks loads, no
                        min-rank clamp, no segment indirection, and a plain load-add-store
                        (single writer per output element).

We call ``_qkv_lora_b_cublas`` directly rather than through ``qkv_lora_b_fwd`` on
purpose: the production dispatch gates the cuBLAS path behind ``max_len >= 8`` and
``not use_cuda_graph``, which forces it OFF at decode. Decode-under-graph is exactly
the workload the proposed static single-LoRA fast flag would enable, so the testbed
measures these single-adapter paths there directly to see whether they actually win.

``x`` is the rank-packed LoRA-A output [s, n_slices*r] (slice i at columns
[i*r, (i+1)*r)); ``w`` is the single-adapter LoRA-B weight [1, n_q + 2*n_kv, r].
Shapes follow fused QKV at various TP: q = n_q, kv = n_kv (per TP shard), total
output = n_q + 2*n_kv.

  python3 bench_qkv_lora_b.py                 # cold bench + correctness (all combos)
  python3 bench_qkv_lora_b.py --mode check    # correctness only
  python3 bench_qkv_lora_b.py --mode bench    # timing only
  python3 bench_qkv_lora_b.py --warm          # L2-hot (reuse buffers)
"""

from __future__ import annotations

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.srt.lora.triton_ops.qkv_lora_b import (
    _qkv_lora_b_cublas,
    qkv_lora_b_fwd,
    qkv_lora_b_single_fwd,
)
from sglang.srt.lora.utils import LoRABatchInfo

N_SLICES = 3

# (n_q, n_kv) per TP shard for the fused QKV projection.
MODELS = {
    "qwen35_tp4": (1024, 256),  # 4 q-heads*256, 1 kv-head*256 (Qwen3.5 full-attn, TP4)
    "llama70b_tp8": (1024, 128),  # 64 q-heads*128/8, 8 kv-heads*128/8
    "qwen3_tp1": (4096, 1024),  # dense, no TP
}

# workload -> segment lengths. Decode = bs requests, 1 token each (max_len=1).
# Prefill = one long segment.
WORKLOADS = {
    "decode_bs1": [1],
    "decode_bs8": [1] * 8,
    "decode_bs32": [1] * 32,
    "decode_bs64": [1] * 64,
    "decode_bs128": [1] * 128,
    "prefill_512": [512],
    "prefill_2048": [2048],
}


def _make_batch_info(seg_lens, rank, scaling, device, uniform: bool) -> LoRABatchInfo:
    """Single-adapter (slot 0) batch info.

    ``uniform=False`` leaves ``uniform_weight_index`` None so ``qkv_lora_b_fwd``
    takes the Triton kernel. ``uniform=True`` sets the uniform fields consumed by
    ``_qkv_lora_b_cublas``. The segment tensors are read-only, so both variants can
    share them.
    """
    bs = len(seg_lens)
    seg_lens_t = torch.tensor(seg_lens, dtype=torch.int32, device=device)
    seg_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens_t, dim=0)
    bi = LoRABatchInfo(
        use_cuda_graph=False,
        bs=bs,
        num_segments=bs,
        seg_indptr=seg_indptr,
        weight_indices=torch.zeros(bs, dtype=torch.int32, device=device),
        lora_ranks=torch.tensor([rank], dtype=torch.int32, device=device),
        scalings=torch.tensor([scaling], dtype=torch.float32, device=device),
        max_len=int(max(seg_lens)),
        seg_lens=seg_lens_t,
        permutation=None,
    )
    if uniform:
        bi.uniform_weight_index = 0
        bi.uniform_rank = rank
        bi.uniform_scaling = scaling
    return bi


@marker.parametrize("model", list(MODELS), ci_vals=["qwen35_tp4"])
@marker.parametrize("rank", [16, 32, 64], ci_vals=[16])
@marker.parametrize(
    "workload",
    list(WORKLOADS),
    ci_vals=["decode_bs64", "prefill_512"],
)
@marker.kernel(
    "impl",
    ["triton", "cublas", "triton_single", "torch"],
    reference="torch",
    rtol=2e-2,
    atol=5e-2,
)
class QKVLoRAB:
    def inputs(self, model: str, rank: int, workload: str):
        device, dtype = "cuda", torch.bfloat16
        n_q, n_kv = MODELS[model]
        seg_lens = WORKLOADS[workload]
        s = sum(seg_lens)
        total_out = n_q + 2 * n_kv
        scaling = 2.0

        # Small magnitudes keep the bf16 accumulation well-conditioned.
        x = torch.randn(s, N_SLICES * rank, device=device, dtype=dtype) * 0.1
        w = torch.randn(1, total_out, rank, device=device, dtype=dtype) * 0.1

        offsets = [0, n_q, n_q + n_kv, n_q + 2 * n_kv]
        # Constant, tiny, read-only state the harness can't clone (dataclass / CPU
        # pinned tensor) -- stash on the instance; only the heavy reads x, w go
        # through marker.io so cold-mode L2 rotation rotates them.
        self._rank = rank
        self._scaling = scaling
        self._max_qkv_out_dim = max(n_q, n_kv)
        self._output_offset = torch.tensor(offsets, dtype=torch.int32, device=device)
        self._output_offset_cpu = torch.tensor(
            offsets, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._bi_triton = _make_batch_info(
            seg_lens, rank, scaling, device, uniform=False
        )
        self._bi_fast = _make_batch_info(seg_lens, rank, scaling, device, uniform=True)
        return marker.io(x, w)

    def _ref(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        r, scaling = self._rank, self._scaling
        wb = w[0].float()
        # CPU copy: a .tolist() on the GPU offsets would D2H-sync, which is illegal
        # while this impl is being captured into a CUDA graph for timing.
        offs = self._output_offset_cpu.tolist()
        out = torch.zeros(x.shape[0], w.shape[-2], device=x.device, dtype=torch.float32)
        for i in range(N_SLICES):
            lo, hi = offs[i], offs[i + 1]
            xi = x[:, i * r : (i + 1) * r].float()
            out[:, lo:hi] = scaling * (xi @ wb[lo:hi, :r].t())
        return out.to(x.dtype)

    def run(self, impl: str, x: torch.Tensor, w: torch.Tensor):
        if impl == "torch":
            return self._ref(x, w)
        if impl == "triton":
            return qkv_lora_b_fwd(
                x,
                w,
                self._bi_triton,
                self._output_offset,
                self._max_qkv_out_dim,
                base_output=None,
                n_slices=N_SLICES,
                output_offset_cpu=None,
            )
        if impl == "cublas":
            return _qkv_lora_b_cublas(
                x,
                w,
                self._bi_fast,
                self._output_offset_cpu,
                base_output=None,
                n_slices=N_SLICES,
            )
        if impl == "triton_single":
            return qkv_lora_b_single_fwd(
                x,
                w,
                self._output_offset,
                self._max_qkv_out_dim,
                self._rank,
                self._scaling,
                base_output=None,
                n_slices=N_SLICES,
            )
        raise ValueError(impl)


if __name__ == "__main__":
    marker.main(QKVLoRAB)
