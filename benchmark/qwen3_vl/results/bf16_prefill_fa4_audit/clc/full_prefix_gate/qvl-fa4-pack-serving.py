import difflib
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

root = pathlib.Path(os.environ["QVL_RUN_ROOT"])
source = pathlib.Path(os.environ["QVL_CONTROL_SOURCE"])
model = "/root/qvl/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
if len(sys.argv) == 1:
    root.mkdir(exist_ok=True, parents=True)
    workers = []
    with (root / "telemetry.csv").open("w") as f:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                "--format=csv",
                "-l",
                "1",
            ],
            stdout=f,
        )
        try:
            for phase in ["A", "B"]:
                workers = []
                for gpu in [
                    int(x)
                    for x in os.environ.get("QVL_GPUS", "0,1,2,3,4,5,6,7").split(",")
                ]:
                    variant = (
                        "candidate" if (gpu % 2 == 1) == (phase == "A") else "control"
                    )
                    out = root / phase / f"gpu{gpu}"
                    out.mkdir(parents=True, exist_ok=True)
                    log = (out / "worker.log").open("w")
                    p = subprocess.Popen(
                        [sys.executable, __file__, phase, str(gpu), variant],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    workers.append((p, out, log))
                while not all((out / "ready").exists() for _, out, _ in workers):
                    if any(p.poll() is not None for p, _, _ in workers):
                        raise RuntimeError("Worker failed before readiness")
                    time.sleep(2)
                (root / f"start_{phase}").touch()
                codes = [p.wait() for p, _, _ in workers]
                for _, _, log in workers:
                    log.close()
                if any(codes):
                    raise RuntimeError(f"Phase{phase} worker exits{codes}")
                print("PHASE " + phase + " DONE", flush=True)
        finally:
            for p, _, _ in workers:
                if p.poll() is None:
                    p.terminate()
            for p, _, _ in workers:
                if p.poll() is None:
                    p.wait(timeout=60)
            monitor.terminate()
            monitor.wait(timeout=10)
    raise SystemExit


def terminate_worker(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, terminate_worker)
phase, gpu, variant = sys.argv[1:]
source = pathlib.Path(
    os.environ["QVL_CANDIDATE_SOURCE"]
    if variant == "candidate"
    else os.environ["QVL_CONTROL_SOURCE"]
)
gpu = int(gpu)
out = root / phase / f"gpu{gpu}"
port = 32000 + gpu
url = f"http://127.0.0.1:{port}"
cache = out / "isolated-cache"
assert not cache.exists(), "Fresh per-run cache required"
cache.mkdir()
env = dict(
    os.environ,
    SGLANG_CACHE_DIR=str(cache),
    CUDA_VISIBLE_DEVICES=str(gpu),
    PYTHONPATH=(
        os.environ["QVL_CANDIDATE_HOOK_DIR"] + ":" if variant == "candidate" else ""
    )
    + str(source / "python"),
    QVL_FA4_EVIDENCE_DIR=str(out),
    QVL_FA4_REPORT=str(out / "fa4-coverage.json"),
    QVL_PACK_REPORT_DIR=str(out),
    HF_HOME="/root/qvl/hf",
    SGLANG_USE_HND_KVCACHE="1",
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
]
base = pathlib.Path(os.environ["QVL_CONTROL_SOURCE"])
files = [
    "python/sglang/srt/layers/quantization/unquant.py",
    "python/sglang/srt/model_executor/runner/flashinfer_autotune.py",
]
patch = "".join(
    "".join(
        difflib.unified_diff(
            (base / name).read_text().splitlines(True),
            (source / name).read_text().splitlines(True),
            fromfile="base/" + name,
            tofile="candidate/" + name,
        )
    )
    for name in files
)
hashes = {
    name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files
}
(out / "metadata.json").write_text(
    json.dumps(
        {
            "phase": phase,
            "gpu": gpu,
            "variant": variant,
            "source_commit": "5f60fb8b67-equivalent verified clean source",
            "cache_policy": "fresh isolated per worker; no prior tuning cache loaded",
            "source_file_sha256": hashes,
            "source_diff_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "server_command": cmd,
            "environment": {
                k: env[k]
                for k in [
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                    "SGLANG_USE_HND_KVCACHE",
                    "MAX_JOBS",
                    "SGLANG_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "CUDA_CACHE_PATH",
                    "FLASHINFER_WORKSPACE_BASE",
                    "SGLANG_CUTE_AOT_CACHE_DIR",
                ]
                if k in env
            },
        },
        indent=2,
    )
)
(out / "source.patch").write_text(patch)
log = (out / "server.log").open("w")
server = subprocess.Popen(
    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
)
(out / "server.pid").write_text(str(server.pid))
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
    with urllib.request.urlopen(url + "/server_info") as r:
        (out / "server_info.json").write_bytes(r.read())
    shutil.copytree(cache, out / "startup-cache")
    (out / "ready").touch()
    while not (root / f"start_{phase}").exists():
        time.sleep(1)
    probe = {
        "input_ids": [[1, 2, 3, 4] * 8] * 128,
        "sampling_params": {"temperature": 0, "max_new_tokens": 2, "ignore_eos": True},
        "return_logprob": True,
    }
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(probe).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        (out / "probe.json").write_bytes(r.read())
    benv = dict(env)
    benv["PYTHONPATH"] = str(source / "python")
    bench = [
        "/root/qvl/venv-bench/bin/sgl-bench",
        "serve",
        "--random-range-ratio",
        "1",
        "--backend",
        "sglang-oai-chat",
        "--base-url",
        url,
        "--model",
        model,
        "--dataset-name",
        "random",
        "--num-prompts",
        os.environ.get("QVL_REQUESTS", "640"),
        "--max-concurrency",
        os.environ.get("QVL_CONCURRENCY", "128"),
        "--warmup-requests",
        os.environ.get("QVL_WARMUP", "128"),
        "--flush-cache",
        "--random-input-len",
        "8192",
        "--random-output-len",
        "1024",
        "--output-file",
        str(out / "bench.jsonl"),
    ]
    (out / "bench-command.json").write_text(json.dumps(bench, indent=2))
    with (out / "bench.log").open("w") as f:
        subprocess.run(bench, env=benv, stdout=f, stderr=subprocess.STDOUT, check=True)
    result = json.loads((out / "bench.jsonl").read_text().splitlines()[-1])
    assert (
        result["completed"] == int(os.environ.get("QVL_REQUESTS", "640"))
        and result["total_input_tokens"]
        == 8192 * int(os.environ.get("QVL_REQUESTS", "640"))
        and result["total_output_tokens"]
        == 1024 * int(os.environ.get("QVL_REQUESTS", "640"))
    )
    (out / "summary.json").write_text(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "duration",
                    "completed",
                    "total_input_tokens",
                    "total_output_tokens",
                    "output_throughput",
                    "median_ttft_ms",
                    "median_tpot_ms",
                ]
            },
            indent=2,
        )
    )
finally:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    log.close()
