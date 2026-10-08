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
                    int(x) for x in os.environ.get("QVL_GPUS", "4,5").split(",")
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
seed = pathlib.Path(
    "/root/qvl/experiments/fa4-production-accuracy/control/saved-autotune.json"
)
seed_target = (
    cache
    / "flashinfer/autotune/0.7.0.post1/sm103/e601f2ea17165ebf/rank_tp0_pp0_dp0.json"
)
seed_target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(seed, seed_target)
shutil.copy2(seed, out / "seed-autotune.json")
(out / "seed.sha256").write_text(hashlib.sha256(seed.read_bytes()).hexdigest() + "\n")


def verify_tactics(label):
    actual = json.loads(seed_target.read_text())
    expected = json.loads(seed.read_text())
    assert {k: v for k, v in actual.items() if not k.startswith("_")} == {
        k: v for k, v in expected.items() if not k.startswith("_")
    }, "Tactic mismatch"
    (out / (label + "-autotune.json")).write_text(json.dumps(actual, indent=2))


expected = json.load(
    open("/root/qvl/experiments/prefix-production/candidate_expected.json")
)
actual = {
    str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (source / "python").rglob("*.py")
}
assert actual == expected
(out / "source-manifest.json").write_text(json.dumps(actual, sort_keys=True))
env = dict(
    os.environ,
    SGLANG_CACHE_DIR=str(cache),
    CUDA_VISIBLE_DEVICES=str(gpu),
    PYTHONPATH="/root/qvl/experiments/mrope-serving-hook:" + str(source / "python"),
    QVL_NATIVE_OUT=str(out),
    QVL_NATIVE_VARIANT=variant,
    SGLANG_FLASHINFER_AUTOTUNE_CACHE="1",
    TRITON_CACHE_DIR="/root/.cache/sglang/triton",
    TORCHINDUCTOR_CACHE_DIR="/root/.cache/sglang/inductor",
    CUDA_CACHE_PATH="/root/.cache/sglang/nv",
    FLASHINFER_WORKSPACE_BASE="/root/.cache/sglang",
    SGLANG_CUTE_AOT_CACHE_DIR="/root/.cache/sglang/cute_aot",
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
    "--max-total-tokens",
    "1400000",
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
]
base = pathlib.Path(os.environ["QVL_CONTROL_SOURCE"])
files = [
    "python/sglang/kernels/ops/attention/dllm_kv_pack.py",
    "python/sglang/kernels/ops/attention/flash_attn/cute/interface.py",
    "python/sglang/srt/layers/attention/trtllm_mha_backend.py",
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
            "source_commit": "same audited current packed-FA4 source; candidate external dynamic native MLP only",
            "cache_policy": "separate caches seeded with identical recorded gateup1/down1; scoped startup-only retuning suppression restored before requests",
            "source_file_sha256": hashes,
            "hook_sha256": hashlib.sha256(
                pathlib.Path("/root/qvl/experiments/mrope_serving_hook.py").read_bytes()
            ).hexdigest(),
            "verified_python_files": len(actual),
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
    verify_tactics("startup")
    proofs = list(out.glob("native-startup-pid*.json"))
    assert len(proofs) == 1, proofs
    proof = json.loads(proofs[0].read_text())
    assert proof["startup_policy_restored"] and len(proof["captured_lt_shapes"]) == 2
    assert proof["compile_count"] == 0 and proof["weight_transforms"] == 0
    info = json.loads((out / "server_info.json").read_text())
    assert (
        info["dtype"] == "bfloat16"
        and info["kv_cache_dtype"] == "bfloat16"
        and info["quantization"] is None
        and info["speculative_algorithm"] is None
    )
    internal = info["internal_states"][0]
    assert internal["memory_usage"]["token_capacity"] == 1400000, internal
    (out / "effective-kv.json").write_text(
        json.dumps(
            {
                "tokens": internal["memory_usage"]["token_capacity"],
                "memory_usage": internal.get("memory_usage"),
            },
            indent=2,
        )
    )
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
        os.environ.get("QVL_REQUESTS", "128"),
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
        result["completed"] == int(os.environ.get("QVL_REQUESTS", "128"))
        and result["total_input_tokens"]
        == 8192 * int(os.environ.get("QVL_REQUESTS", "128"))
        and result["total_output_tokens"]
        == 1024 * int(os.environ.get("QVL_REQUESTS", "128"))
    )
    verify_tactics("terminal")
    with urllib.request.urlopen(url + "/flush_cache") as response:
        assert response.status == 200
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
