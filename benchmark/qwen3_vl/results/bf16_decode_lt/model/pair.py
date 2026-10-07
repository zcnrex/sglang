import json
import os
import pathlib
import subprocess

root = pathlib.Path("/root/qvl/experiments/decode-lt")
base = [
    "/root/qvl/venv-sgl/bin/python",
    str(root / "model_verified.py"),
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
    "128",
    "--enforce-disable-flashinfer-allreduce-fusion",
    "--batch-size",
    *["128"] * 6,
    "--input-len",
    "128",
    "--output-len",
    "32",
]
for variant in ["control_gpu2", "candidate_lt"]:
    out = root / variant
    out.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="2",
        QVL_VARIANT=variant,
        QVL_TGV_DOWN="0",
        PYTHONPATH="/root/qvl/sglang-current-profile/python",
        HF_HOME="/root/qvl/hf",
        SGLANG_USE_HND_KVCACHE="1",
        MAX_JOBS="8",
    )
    cmd = base + ["--result-filename", str(out / "results.jsonl")]
    (out / "command.json").write_text(
        json.dumps(
            {
                "command": cmd,
                "environment": {
                    k: env[k]
                    for k in [
                        "CUDA_VISIBLE_DEVICES",
                        "QVL_VARIANT",
                        "QVL_TGV_DOWN",
                        "PYTHONPATH",
                        "HF_HOME",
                        "SGLANG_USE_HND_KVCACHE",
                        "MAX_JOBS",
                    ]
                },
            },
            indent=2,
        )
    )
    with (out / "run.log").open("w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    print(variant + " DONE", flush=True)
