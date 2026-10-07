import os

if os.environ.get("QVL_GRAPH_VARIANT") == "candidate":
    import json
    import pathlib

    from sglang.srt.configs import model_config

    model_config.multimodal_breakable_cuda_graph_supported_model_archs.append(
        "Qwen3VLForConditionalGeneration"
    )
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    orig = PrefillCudaGraphRunner.can_run_graph
    execute_orig = PrefillCudaGraphRunner.execute
    state = {"calls": 0, "eligible": 0, "mm_fallback": 0, "executed": 0, "shapes": {}}
    path = pathlib.Path(os.environ["QVL_GRAPH_REPORT"])

    def save():
        path.write_text(json.dumps(state, indent=2))

    def can(self, b):
        mm = b.contains_mm_inputs()
        eligible = not mm and orig(self, b)
        state["calls"] += 1
        state["eligible"] += int(eligible)
        state["mm_fallback"] += int(mm)
        key = str((str(b.forward_mode), len(b.input_ids), b.batch_size, eligible))
        state["shapes"][key] = state["shapes"].get(key, 0) + 1
        if mm or state["calls"] % 16 == 0:
            save()
        return eligible

    def execute(self, *a, **kw):
        state["executed"] += 1
        return execute_orig(self, *a, **kw)

    PrefillCudaGraphRunner.can_run_graph = can
    PrefillCudaGraphRunner.execute = execute
