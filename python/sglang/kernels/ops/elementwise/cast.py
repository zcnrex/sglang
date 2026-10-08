import torch
import triton
import triton.language as tl


@triton.jit
def _cast_bf16_to_fp32_kernel(x, out, numel: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < numel, other=0).to(tl.float32)
    tl.store(out + offsets, values, offsets < numel)


def cast_bf16_to_fp32(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Copy contiguous CUDA BF16 values into a same-shaped FP32 output buffer."""
    assert x.is_cuda and out.device == x.device
    assert x.dtype == torch.bfloat16 and out.dtype == torch.float32
    assert x.shape == out.shape and x.is_contiguous() and out.is_contiguous()
    assert not x.requires_grad and not out.requires_grad
    numel = x.numel()
    if numel:
        block, warps = (512, 8) if numel <= 1 << 24 else (1024, 4)
        _cast_bf16_to_fp32_kernel[(triton.cdiv(numel, block),)](
            x, out, numel, block, num_warps=warps, num_stages=1
        )
    return out
