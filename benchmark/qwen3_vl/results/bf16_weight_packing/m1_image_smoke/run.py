import argparse
import base64
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int)
p.add_argument("--variant")
a = p.parse_args()
root = pathlib.Path("/root/qvl/experiments/m1-image")
out = root / a.variant
out.mkdir(parents=True, exist_ok=True)
control = pathlib.Path("/root/qvl/sglang-public-lt-clean")
candidate = pathlib.Path("/root/qvl/sglang-m1-production")
source = control if a.variant == "control" else candidate
files = lambda base: {
    str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (base / "python").rglob("*")
    if p.is_file()
    and "__pycache__" not in str(p)
    and p.suffix in [".py", ".cuh", ".cu", ".h", ".cpp"]
}
c, d = files(control), files(candidate)
diff = [k for k in c.keys() | d.keys() if c.get(k) != d.get(k)]
assert diff == ["python/sglang/srt/layers/quantization/unquant.py"], diff
assert d[diff[0]] == "0fa01b10b34bec46f4cca14b6dfb96fa67a07d94e68c1d08035b8c851740f431"
(out / "source-audit.json").write_text(
    json.dumps(
        {
            "count_control": len(c),
            "count_candidate": len(d),
            "differences": diff,
            "candidate_sha256": d[diff[0]],
            "control_sha256": c[diff[0]],
        },
        indent=2,
    )
    + "\n"
)
model = "/root/qvl/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
port = 32600 + a.gpu
url = f"http://127.0.0.1:{port}"
env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=str(a.gpu),
    PYTHONPATH=str(root / "hooks") + ":" + str(source / "python"),
    HF_HOME="/root/qvl/hf",
    SGLANG_USE_HND_KVCACHE="1",
    MAX_JOBS="4",
    SGLANG_CACHE_DIR=str(out / "cache"),
    QVL_M1_MARKER=str(out / "marker.jsonl"),
    TRITON_CACHE_DIR="/root/.cache/sglang/triton",
    TORCHINDUCTOR_CACHE_DIR="/root/.cache/sglang/inductor",
    CUDA_CACHE_PATH="/root/.cache/sglang/nv",
    FLASHINFER_WORKSPACE_BASE="/root/.cache/sglang",
    SGLANG_CUTE_AOT_CACHE_DIR="/root/.cache/sglang/cute_aot",
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
    "--mem-fraction-static",
    "0.4",
    "--cuda-graph-bs-decode",
    "1",
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
]
(out / "command.json").write_text(
    json.dumps(
        {
            "cmd": cmd,
            "env": {
                k: env[k]
                for k in env
                if k
                in [
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                    "HF_HOME",
                    "MAX_JOBS",
                    "TRITON_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "CUDA_CACHE_PATH",
                    "FLASHINFER_WORKSPACE_BASE",
                    "SGLANG_CUTE_AOT_CACHE_DIR",
                ]
                or k.startswith("SGLANG_")
                or k.startswith("QVL_")
            },
        },
        indent=2,
    )
    + "\n"
)
f = (out / "server.log").open("w")
server = subprocess.Popen(
    cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
)
(out / "server.pid").write_text(str(server.pid))
print("SERVER_PID", server.pid, flush=True)
try:
    for _ in range(600):
        if server.poll() is not None:
            raise RuntimeError("server exited")
        try:
            urllib.request.urlopen(url + "/v1/models", timeout=2).close()
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError("readiness timeout")
    image = pathlib.Path("/root/qvl/sglang-perf/examples/assets/example_image.png")
    data = image.read_bytes()
    assert (
        hashlib.sha256(data).hexdigest()
        == "e06917184a00b14abd70cd8ea0ff5dca9abfbbad29f7b25c02f97133d4cd060e"
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in one short sentence.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(data).decode()
                        },
                    },
                ],
            }
        ],
        "max_tokens": 32,
        "temperature": 0,
        "top_p": 1,
        "logprobs": True,
        "top_logprobs": 1,
    }
    (out / "request-metadata.json").write_text(
        json.dumps(
            {
                "image": str(image),
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "prompt": payload["messages"][0]["content"][0]["text"],
                "max_tokens": 32,
                "temperature": 0,
                "top_p": 1,
            },
            indent=2,
        )
        + "\n"
    )
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        response = json.load(r)
    (out / "response.json").write_text(json.dumps(response, indent=2) + "\n")
    assert response["usage"]["completion_tokens"] == 32, response["usage"]
    print("SUCCESS", response["usage"], flush=True)
finally:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    f.close()
    (out / "stopped").write_text(str(server.returncode))
