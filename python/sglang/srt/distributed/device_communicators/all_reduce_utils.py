MiB = 1024 * 1024

TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    9: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
    10: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
}


def get_torch_symm_mem_all_reduce_max_size(
    device_capability: int, world_size: int
) -> int:
    from sglang.srt.environ import envs

    max_size = TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES[device_capability][world_size]
    override = envs.SGLANG_TORCH_SYMM_MEM_ALLREDUCE_MAX_BYTES.get()
    if override is not None and override > 0:
        max_size = override
    return max_size
