import json
import os
import pathlib
import runpy
import time

import flashinfer

ns = runpy.run_path("/root/qvl/experiments/fa4_prefix_serving_hook.py")
state = ns["state"]
original = flashinfer.prefill.trtllm_batch_context_with_kv_cache
p = pathlib.Path(os.environ["QVL_FA4_REPORT"])
trace = p.with_name("layer0-contexts-pid" + str(os.getpid()) + ".jsonl")


def observe(*a, **kw):
    cur = state["current"]
    tracked = cur is not None and cur[2].layer_id == 0
    if tracked:
        before = state["fa4_calls"]
        fallback = dict(state["fallback_reasons"])
        batch = cur[3]
        n = kw["batch_size"]
        prefix = (
            list(batch.extend_prefix_lens_cpu[:n])
            if batch.extend_prefix_lens_cpu is not None
            else None
        )
    result = original(*a, **kw)
    if tracked:
        reason = next(
            (k for k, v in state["fallback_reasons"].items() if v > fallback.get(k, 0)),
            None,
        )
        row = {
            "time_ns": time.time_ns(),
            "pid": os.getpid(),
            "used_fa4": state["fa4_calls"] > before,
            "query_tokens": int(kw["query"].shape[0]),
            "context_requests": n,
            "prefix_lens": prefix,
            "extend_lens": list(batch.extend_seq_lens_cpu[:n]),
            "fallback_reason": reason,
            "max_scratch_bytes": state["max_scratch_bytes"],
            "clc_cache_hits": state["clc_cache_hits"],
        }
        trace.parent.mkdir(exist_ok=True, parents=True)
        with trace.open("a") as f:
            f.write(json.dumps(row) + "\n")
    return result


flashinfer.prefill.trtllm_batch_context_with_kv_cache = observe
