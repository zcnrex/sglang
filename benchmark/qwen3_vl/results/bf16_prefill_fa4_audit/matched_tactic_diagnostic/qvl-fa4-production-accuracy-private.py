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
r = (
    pathlib.Path(
        os.environ.get(
            "QVL_ACCURACY_ROOT", "/root/qvl/experiments/fa4-production-accuracy-private"
        )
    )
    / a.variant
)
r.mkdir(parents=True, exist_ok=True)
model = "/root/qvl/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
port = 32100 + int(a.gpu)
url = f"http://127.0.0.1:{port}"
cache = r / "isolated-cache"
assert not cache.exists(), "Fresh per-run cache required"
cache.mkdir()
seed = pathlib.Path(
    "/root/qvl/experiments/fa4-production-accuracy/control/saved-autotune.json"
)
seed_target = (
    cache
    / "flashinfer/autotune/0.7.0.post1/sm103/e601f2ea17165ebf/rank_tp0_pp0_dp0.json"
)
seed_target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(seed, seed_target)
shutil.copy2(seed, r / "seed-autotune.json")
import hashlib

(r / "seed.sha256").write_text(hashlib.sha256(seed.read_bytes()).hexdigest() + "\n")


def verify_tactics(label):
    actual = json.loads(seed_target.read_text())
    expected = json.loads(seed.read_text())
    assert {k: v for k, v in actual.items() if not k.startswith("_")} == {
        k: v for k, v in expected.items() if not k.startswith("_")
    }, "Tactic mismatch"
    (r / (label + "-autotune.json")).write_text(json.dumps(actual, indent=2))


env = dict(
    os.environ,
    SGLANG_CACHE_DIR=str(cache),
    SGLANG_FLASHINFER_AUTOTUNE_CACHE="1",
    CUDA_VISIBLE_DEVICES=a.gpu,
    PYTHONPATH="/root/qvl/experiments/prefix-production-private-observer:"
    + a.source
    + "/python",
    QVL_PREFIX_OBSERVE=str(r),
    HF_HOME="/root/qvl/hf",
    SGLANG_USE_HND_KVCACHE="1",
    QVL_FA4_EVIDENCE_DIR=str(r),
    QVL_FA4_REPORT=str(r / "fa4-coverage.json"),
    QVL_PACK_REPORT_DIR=str(r),
    TRITON_CACHE_DIR="/root/.cache/sglang/triton",
    TORCHINDUCTOR_CACHE_DIR="/root/.cache/sglang/inductor",
    CUDA_CACHE_PATH="/root/.cache/sglang/nv",
    FLASHINFER_WORKSPACE_BASE="/root/.cache/sglang",
    SGLANG_CUTE_AOT_CACHE_DIR="/root/.cache/sglang/cute_aot",
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
                    "QVL_FA4_EVIDENCE_DIR",
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
(r / "server.pid").write_text(str(server.pid))
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
    verify_tactics("startup")
    proofs = list(r.glob("dispatch-pid*.json"))
    assert len(proofs) == 1, proofs
    proof = json.loads(proofs[0].read_text())
    assert (
        len(proof["dispatch_shapes"]) == 2
        and len(proof["ready"]) == 2
        and len(proof["startup_shapes"]) >= 2
    ), proof
    assert proof["capture"], proof
    (r / "startup-proof-pass").write_text("pass")
    if os.environ.get("QVL_STARTUP_ONLY") == "1":
        raise SystemExit(0)
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
        "--num-examples",
        "1314",
        "--num-threads",
        "128",
        "--max-tokens",
        "2048",
    ]
    (r / "eval-command.json").write_text(json.dumps(bench, indent=2))
    with (r / "eval.log").open("w") as log:
        subprocess.run(bench, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify_tactics("terminal")
finally:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    f.close()
