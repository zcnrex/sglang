import os

if os.environ.get("QVL_PREFIX_OBSERVE"):
    import json
    import pathlib
    import time

    from sglang.kernels.ops.attention import dllm_kv_pack as mod

    orig = mod.pack_prefix_current
    count = 0

    def observed(*args, **kwargs):
        global count
        result = orig(*args, **kwargs)
        count += 1
        if count % 36 == 1:
            p = (
                pathlib.Path(os.environ["QVL_PREFIX_OBSERVE"])
                / f"pack-pid{os.getpid()}.jsonl"
            )
            with p.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "time_ns": time.time_ns(),
                            "pack_calls": count,
                            "query_rows": args[0].shape[0],
                            "packed_rows": args[7].shape[0],
                            "csr_ids": args[6].numel(),
                            "packed_bytes": args[7].numel()
                            * args[7].element_size()
                            * 2,
                        }
                    )
                    + "\n"
                )
        return result

    mod.pack_prefix_current = observed
