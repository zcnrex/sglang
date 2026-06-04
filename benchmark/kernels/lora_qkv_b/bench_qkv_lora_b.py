"""qkv_lora_b testbed: general vs single-adapter kernels, atomic_add vs load-add-store.

Built on the PR #27292 ``marker.Kernel`` contract (correctness-aware, seeded, cold by
default, measured roofline). Five impls share one input set per combo:

  - ``general_atomic`` : ``qkv_lora_b_fwd``        (general kernel) with ``USE_ATOMIC``
  - ``general_ladd``   : ``qkv_lora_b_fwd``        (general kernel) load-add-store
  - ``single_atomic``  : ``qkv_lora_b_single_fwd`` (single-adapter kernel) with ``USE_ATOMIC``
  - ``single_ladd``    : ``qkv_lora_b_single_fwd`` (single-adapter kernel) load-add-store
  - ``cublas``         : ``_qkv_lora_b_cublas``    (one cuBLAS addmm_ per slice)

The accumulate mode is a kernel ``USE_ATOMIC`` constexpr flag (threaded through the
launchers as ``use_atomic=``), so atomic_add vs load-add-store is a one-arg switch.
``general_ladd`` is the correctness reference; all impls compute the same expand-add so
they are cross-checked against it (the fp32-gold check lives in the full 63-config run).

cuBLAS / single are dispatched directly to bypass the production ``max_len>=8`` gate, so
decode-under-graph is measured.

Models map to a list of per-slice output dims (length = n_slices). ``x`` is the
rank-packed LoRA-A output [s, n_slices*r]; ``w`` is the single-adapter LoRA-B weight
[1, sum(slice_dims), r].

  python3 bench_qkv_lora_b.py                 # cold bench + correctness
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

# model -> per-slice output dims (per TP shard). len(dims) == n_slices.
MODELS = {
    # Hot path: fused qkvz projection, q/k/v/z = 512/512/1024/1024.
    "qwen35_tp4": [512, 512, 1024, 1024],
    "llama70b_tp8": [1024, 128, 128],  # q, k, v (GQA, TP8)
    "qwen3_tp1": [4096, 1024, 1024],  # q, k, v (dense, no TP)
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
    """Single-adapter (slot 0) batch info. ``uniform`` sets the fields the cuBLAS path
    consumes; the general kernel ignores them. Segment tensors are read-only/shared."""
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


# Focused on the qkvz hot path; add other MODELS keys here to widen the sweep.
@marker.parametrize("model", ["qwen35_tp4"])
@marker.parametrize("rank", [16])
@marker.parametrize("workload", list(WORKLOADS), ci_vals=["decode_bs64", "prefill_512"])
@marker.kernel(
    "impl",
    ["general_atomic", "general_ladd", "single_atomic", "single_ladd", "cublas"],
    reference="general_ladd",
    rtol=2e-2,
    atol=5e-2,
)
class QKVLoRAB:
    def inputs(self, model: str, rank: int, workload: str):
        device, dtype = "cuda", torch.bfloat16
        dims = MODELS[model]
        n_slices = len(dims)
        seg_lens = WORKLOADS[workload]
        s = sum(seg_lens)
        total_out = sum(dims)
        scaling = 2.0

        # Small magnitudes keep the bf16 accumulation well-conditioned.
        x = torch.randn(s, n_slices * rank, device=device, dtype=dtype) * 0.1
        w = torch.randn(1, total_out, rank, device=device, dtype=dtype) * 0.1

        offsets = [0]
        for d in dims:
            offsets.append(offsets[-1] + d)
        # Constant, tiny, read-only state the harness can't clone (dataclass / CPU
        # pinned tensor) -- stash on the instance; only the heavy reads x, w go
        # through marker.io so cold-mode L2 rotation rotates them.
        self._rank = rank
        self._scaling = scaling
        self._n_slices = n_slices
        self._max_qkv_out_dim = max(dims)
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
        # CPU copy: a .tolist() on the GPU offsets would D2H-sync, illegal under the
        # CUDA-graph capture that also times this impl.
        offs = self._output_offset_cpu.tolist()
        out = torch.zeros(x.shape[0], w.shape[-2], device=x.device, dtype=torch.float32)
        for i in range(self._n_slices):
            lo, hi = offs[i], offs[i + 1]
            xi = x[:, i * r : (i + 1) * r].float()
            out[:, lo:hi] = scaling * (xi @ wb[lo:hi, :r].t())
        return out.to(x.dtype)

    def run(self, impl: str, x: torch.Tensor, w: torch.Tensor):
        if impl in ("general_atomic", "general_ladd"):
            return qkv_lora_b_fwd(
                x,
                w,
                self._bi_triton,
                self._output_offset,
                self._max_qkv_out_dim,
                base_output=None,
                n_slices=self._n_slices,
                output_offset_cpu=None,
                use_atomic=(impl == "general_atomic"),
            )
        if impl in ("single_atomic", "single_ladd"):
            return qkv_lora_b_single_fwd(
                x,
                w,
                self._output_offset,
                self._max_qkv_out_dim,
                self._rank,
                self._scaling,
                base_output=None,
                n_slices=self._n_slices,
                use_atomic=(impl == "single_atomic"),
            )
        if impl == "cublas":
            return _qkv_lora_b_cublas(
                x,
                w,
                self._bi_fast,
                self._output_offset_cpu,
                base_output=None,
                n_slices=self._n_slices,
            )
        raise ValueError(impl)


if __name__ == "__main__":
    marker.main(QKVLoRAB)
