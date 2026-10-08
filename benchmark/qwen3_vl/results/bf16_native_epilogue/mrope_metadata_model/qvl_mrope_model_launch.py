import hashlib
import json
import os
import pathlib
import subprocess

root = pathlib.Path("/root/qvl/experiments/mrope-metadata-model")
root.mkdir(exist_ok=True)
for gpu in [4, 5]:
    r = root / f"gpu{gpu}"
    r.mkdir(exist_ok=True)
    e = os.environ.copy()
    e.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PAIR_OFFSET=str(gpu - 4),
        PYTHONPATH="/root/qvl/sglang-prefix-production/python",
        HF_HOME="/root/qvl/hf",
        SGLANG_USE_HND_KVCACHE="1",
        MAX_JOBS="8",
        QVL_MODEL_OUT=str(r),
        SGLANG_CACHE_DIR=str(r / "sglang-cache"),
    )
    e.update(
        TRITON_CACHE_DIR="/root/.cache/sglang/triton",
        TORCHINDUCTOR_CACHE_DIR="/root/.cache/sglang/torchinductor",
        CUDA_CACHE_PATH="/root/.cache/sglang/cuda",
        FLASHINFER_WORKSPACE_BASE="/root/.cache/sglang/flashinfer",
        SGLANG_CUTE_AOT_CACHE_DIR="/root/.cache/sglang/cute_aot",
    )
    cmd = (
        [
            "/root/qvl/venv-sgl/bin/python",
            "-u",
            "/tmp/qvl_mrope_model.py",
            "--model-path",
            "/root/qvl/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "bfloat16",
            "--attention-backend",
            "trtllm_mha",
            "--page-size",
            "32",
            "--enable-mixed-chunk",
            "--chunked-prefill-size",
            "16384",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-max-bs-decode",
            "64",
            "--enforce-disable-flashinfer-allreduce-fusion",
            "--bf16-gemm-backend",
            "cutedsl",
            "--batch-size",
        ]
        + ["42"] * 8
        + [
            "--input-len",
            "8220",
            "--output-len",
            "4",
            "--result-filename",
            str(r / "results.jsonl"),
        ]
    )
    (r / "launch.json").write_text(
        json.dumps(
            {
                "cmd": cmd,
                "env": {
                    k: e[k]
                    for k in [
                        "CUDA_VISIBLE_DEVICES",
                        "PAIR_OFFSET",
                        "PYTHONPATH",
                        "HF_HOME",
                        "SGLANG_USE_HND_KVCACHE",
                        "MAX_JOBS",
                        "QVL_MODEL_OUT",
                        "SGLANG_CACHE_DIR",
                        "TRITON_CACHE_DIR",
                        "TORCHINDUCTOR_CACHE_DIR",
                        "CUDA_CACHE_PATH",
                        "FLASHINFER_WORKSPACE_BASE",
                        "SGLANG_CUTE_AOT_CACHE_DIR",
                    ]
                },
                "hook_sha256": hashlib.sha256(
                    pathlib.Path("/tmp/qvl_mrope_model.py").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    f = open(r / "run.log", "w")
    p = subprocess.Popen(
        cmd, env=e, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
    )
    print(gpu, p.pid, flush=True)
