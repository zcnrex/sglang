import json
import os
import pathlib
import time

import flashinfer

from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.managers.scheduler import Scheduler

root = pathlib.Path(os.environ["QVL_PREFIX_ROOT"])
root.mkdir(exist_ok=True, parents=True)
path = root / f"contexts-{os.getpid()}.jsonl"
current = None
epoch = 0
origextend = TRTLLMHAAttnBackend.forward_extend
origcontext = flashinfer.prefill.trtllm_batch_context_with_kv_cache
origflush = Scheduler.flush_cache


def write(d):
    d.update(pid=os.getpid(), time_ns=time.time_ns(), epoch=epoch)
    with path.open("a") as f:
        f.write(json.dumps(d) + "\n")


def flush(self, *a, **kw):
    global epoch
    result = origflush(self, *a, **kw)
    epoch += 1
    write({"kind": "flush_cache", "result": str(result)})
    return result


def extend(self, q, k, v, layer, batch, *a, **kw):
    global current
    prev = current
    current = (layer, batch)
    try:
        return origextend(self, q, k, v, layer, batch, *a, **kw)
    finally:
        current = prev


def context(*a, **kw):
    if current is not None:
        layer, b = current
        if layer.layer_id == 0:
            n = int(kw["batch_size"])
            prefix = list(b.extend_prefix_lens_cpu)
            length = list(b.extend_seq_lens_cpu)
            phase = (
                (root / "phase.txt").read_text().strip()
                if (root / "phase.txt").exists()
                else "startup"
            )
            write(
                {
                    "kind": "context",
                    "phase_marker": phase,
                    "mode": b.forward_mode.name,
                    "batch_size": b.batch_size,
                    "context_requests": n,
                    "context_tokens": kw["query"].shape[0],
                    "max_q_len": int(kw["max_q_len"]),
                    "prefix_lens": prefix,
                    "extend_lens": length,
                    "context_prefix_lens": prefix[:n],
                    "context_extend_lens": length[:n],
                    "suffix_requests": b.batch_size - n,
                    "suffix_prefix_lens": prefix[n:],
                    "suffix_extend_lens": length[n:],
                }
            )
    return origcontext(*a, **kw)


Scheduler.flush_cache = flush
TRTLLMHAAttnBackend.forward_extend = extend
flashinfer.prefill.trtllm_batch_context_with_kv_cache = context
