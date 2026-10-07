import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request

root = pathlib.Path("/root/qvl/experiments/decode-lt-public-serving")
source = pathlib.Path("/root/qvl/sglang-current-profile")
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
                for gpu in range(8):
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
gpu = int(gpu)
out = root / phase / f"gpu{gpu}"
port = 32000 + gpu
url = f"http://127.0.0.1:{port}"
env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=str(gpu),
    PYTHONPATH=str(root / "hooks") + ":" + str(source / "python"),
    HF_HOME="/root/qvl/hf",
    SGLANG_USE_HND_KVCACHE="1",
    QVL_LT_PATCH="1" if variant == "candidate" else "0",
    QVL_EVIDENCE_DIR=str(out),
    MAX_JOBS="8",
)
cmd = [
    "/root/qvl/venv-sgl/bin/python",
    "-m",
    "sglang.launch_server",
    "--model-path",
    model,
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
(out / "metadata.json").write_text(
    json.dumps(
        {
            "phase": phase,
            "gpu": gpu,
            "variant": variant,
            "source_commit": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "server_command": cmd,
            "environment": {
                k: env[k]
                for k in [
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                    "SGLANG_USE_HND_KVCACHE",
                    "QVL_LT_PATCH",
                    "MAX_JOBS",
                ]
            },
        },
        indent=2,
    )
)
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
    benv.pop("QVL_LT_PATCH")
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
        "640",
        "--max-concurrency",
        "128",
        "--warmup-requests",
        "128",
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
        result["completed"] == 640
        and result["total_input_tokens"] == 5242880
        and result["total_output_tokens"] == 655360
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
