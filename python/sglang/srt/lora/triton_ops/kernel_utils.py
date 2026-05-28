import os

import triton
import triton.language as tl


def _parse_env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() not in ("0", "false", "off", "no", "v1")


def lora_kernels_v2_enabled(default: bool = False) -> bool:
    """Shared LoRA Triton kernel v2 switch.

    `SGLANG_LORA_KERNELS_V2=1` enables the experimental local LoRA kernel
    changes.  The default path stays aligned with upstream/main.
    """
    value = os.environ.get("SGLANG_LORA_KERNELS_V2")
    return _parse_env_bool(value, default)


def lora_kernels_v2_block_m(default: int = 16) -> int:
    value = os.environ.get("SGLANG_LORA_KERNELS_V2_BLOCK_M")
    return int(value) if value else default


@triton.jit
def _resolve_token_positions(
    sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER: tl.constexpr
):
    """Map logical segment offsets to physical token positions.

    When SORTED_BY_ADAPTER is True, segments are grouped by adapter and
    sorted_token_ids provides the indirection to the original token rows.
    When False, tokens are already contiguous starting at seg_start.
    """
    if SORTED_BY_ADAPTER:
        return tl.load(
            sorted_token_ids + seg_start + s_offset, mask=s_offset < seg_len
        ).to(tl.int64)
    return (seg_start + s_offset).to(tl.int64)
