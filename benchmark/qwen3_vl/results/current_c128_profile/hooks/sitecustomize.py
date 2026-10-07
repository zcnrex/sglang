import os

if os.getenv("QVL_EVENT_PROFILE") == "1":
    import json
    import pathlib
    import queue
    import threading
    import time

    import torch

    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.model_runner import ModelRunner

    root = pathlib.Path(os.environ["QVL_PROFILE_DIR"])
    state = {"epoch": 0, "active": False, "seq": 0}
    pending = queue.Queue()
    original_forward = ModelRunner.forward
    original_flush = Scheduler.flush_cache

    def flush(self, *a, **kw):
        result = original_flush(self, *a, **kw)
        if result:
            state["epoch"] += 1
            state["active"] = True
            with (root / ("flush-" + str(os.getpid()) + ".jsonl")).open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": state["epoch"],
                            "wall": time.time(),
                            "monotonic": time.monotonic(),
                        }
                    )
                    + "\n"
                )
        return result

    def forward(self, batch, *a, **kw):
        if not state["active"]:
            return original_forward(self, batch, *a, **kw)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        item = {
            "epoch": state["epoch"],
            "seq": state["seq"],
            "mode": batch.forward_mode.name,
            "batch_size": batch.batch_size,
            "input_tokens": batch.input_ids.numel(),
            "extend_num_tokens": getattr(batch, "extend_num_tokens", None),
            "wall_start": time.time(),
        }
        state["seq"] += 1
        start.record()
        result = original_forward(self, batch, *a, **kw)
        end.record()
        item["host_end"] = time.time()
        pending.put((start, end, item))
        return result

    def writer():
        with (root / ("events-" + str(os.getpid()) + ".jsonl")).open(
            "a", buffering=1
        ) as f:
            while True:
                start, end, item = pending.get()
                while not end.query():
                    time.sleep(0.005)
                item["gpu_ms"] = start.elapsed_time(end)
                f.write(json.dumps(item) + "\n")

    threading.Thread(target=writer, daemon=True).start()
    ModelRunner.forward = forward
    Scheduler.flush_cache = flush
