import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--gpu", required=True)
p.add_argument("--variant", required=True)
p.add_argument("--extra", action="append", default=[])
a = p.parse_args()
r = pathlib.Path("/root/qvl/experiments/m1-gateup-full-accuracy") / a.variant
r.mkdir(parents=True, exist_ok=True)
model = "/root/qvl/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
port = 32100 + int(a.gpu)
url = f"http://127.0.0.1:{port}"
cache = r / "isolated-cache"
assert not cache.exists(), "Fresh per-run cache required"
cache.mkdir()
env = dict(
    os.environ,
    SGLANG_CACHE_DIR=str(cache),
    CUDA_VISIBLE_DEVICES=a.gpu,
    PYTHONPATH="/root/qvl/experiments/m1-gateup-accuracy/hooks:" + a.source + "/python",
    HF_HOME="/root/qvl/hf",
    SGLANG_USE_HND_KVCACHE="1",
    QVL_M1_CANDIDATE="1" if a.variant.startswith("candidate") else "0",
    TRITON_CACHE_DIR="/root/.cache/sglang/triton",
    TORCHINDUCTOR_CACHE_DIR="/root/.cache/sglang/inductor",
    CUDA_CACHE_PATH="/root/.cache/sglang/nv",
    FLASHINFER_WORKSPACE_BASE="/root/.cache/sglang",
    SGLANG_CUTE_AOT_CACHE_DIR="/root/.cache/sglang/cute_aot",
    QVL_M1_MARKER=str(r / "execution-marker.jsonl"),
    MAX_JOBS="8",
)
cmd = [
    "/root/qvl/venv-sgl/bin/python",
    "-m",
    "sglang.launch_server",
    "--model-path",
    model,
    "--bf16-gemm-backend",
    "cutedsl",
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
    "--enforce-disable-flashinfer-allreduce-fusion",
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
] + a.extra
(r / "command.json").write_text(
    json.dumps(
        {
            "server": cmd,
            "env": {
                k: env[k]
                for k in [
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                    "HF_HOME",
                    "SGLANG_USE_HND_KVCACHE",
                    "QVL_M1_MARKER",
                    "QVL_M1_CANDIDATE",
                    "SGLANG_CACHE_DIR",
                ]
            },
            "source": a.source,
            "cache_policy": "fresh isolated per worker; no prior tuning cache loaded",
        },
        indent=2,
    )
)
f = (r / "server.log").open("w")
server = subprocess.Popen(
    cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
)
try:
    for _ in range(600):
        if server.poll() is not None:
            raise RuntimeError("Server exited")
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=2):
                break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError("Readiness timeout")
    shutil.copytree(cache, r / "startup-cache")
    bench = [
        "/root/qvl/venv-sgl/bin/python",
        "/root/qvl/sglang-current-profile/benchmark/qwen3_vl/eval_gsm8k.py",
        "--base-url",
        url,
        "--output",
        str(r / "results"),
        "--model",
        model,
        "--data-path",
        "/root/qvl/experiments/m1-gateup-full-accuracy/shard"
        + a.variant[-1]
        + ".jsonl",
        "--num-threads",
        "1",
        "--max-tokens",
        "2048",
    ]
    (r / "eval-command.json").write_text(json.dumps(bench, indent=2))
    with (r / "eval.log").open("w") as log:
        subprocess.run(bench, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
finally:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    f.close()
