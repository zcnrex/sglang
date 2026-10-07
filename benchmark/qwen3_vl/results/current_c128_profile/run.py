import json
import os
import pathlib
import signal
import subprocess
import time
import urllib.request

R = pathlib.Path("/root/qvl")
out = R / "experiments/current-c128-profile"
source = R / "sglang-current-profile"
model = str(
    R
    / "hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
url = "http://127.0.0.1:32000"
env = dict(
    os.environ,
    HF_HOME=str(R / "hf"),
    PYTHONPATH=str(out / "hooks") + ":" + str(source / "python"),
    CUDA_VISIBLE_DEVICES="0",
    SGLANG_USE_HND_KVCACHE="1",
    QVL_EVENT_PROFILE="1",
    QVL_PROFILE_DIR=str(out),
    MAX_JOBS="8",
)
cmd = [
    str(R / "venv-sgl/bin/python"),
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
    "--chunked-prefill-size",
    "16384",
    "--enable-mixed-chunk",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--enforce-disable-flashinfer-allreduce-fusion",
    "--host",
    "127.0.0.1",
    "--port",
    "32000",
]
(out / "metadata.json").write_text(
    json.dumps(
        {
            "command": cmd,
            "source_commit": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "env": {
                k: env[k]
                for k in [
                    "CUDA_VISIBLE_DEVICES",
                    "SGLANG_USE_HND_KVCACHE",
                    "PYTHONPATH",
                ]
            },
            "scope": "ModelRunner.forward CUDA intervals exclude sampling and scheduling outside forward; profiled diagnostic not throughput claim",
        },
        indent=2,
    )
)
log = (out / "server.log").open("w")
p = subprocess.Popen(
    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
)
(out / "server.pid").write_text(str(p.pid))
try:
    for i in range(600):
        if p.poll() is not None:
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
    benv = dict(env)
    benv.pop("QVL_EVENT_PROFILE")
    benv["PYTHONPATH"] = str(source / "python")

    def bench(name, n, w):
        args = [
            str(R / "venv-bench/bin/sgl-bench"),
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
            str(n),
            "--max-concurrency",
            "128",
            "--warmup-requests",
            str(w),
            "--flush-cache",
            "--random-input-len",
            "8192",
            "--random-output-len",
            "1024",
            "--output-file",
            str(out / (name + ".jsonl")),
        ]
        (out / (name + "-command.json")).write_text(json.dumps(args))
        with (out / (name + ".log")).open("w") as f:
            subprocess.run(
                args, env=benv, stdout=f, stderr=subprocess.STDOUT, check=True
            )
        d = json.loads((out / (name + ".jsonl")).read_text().splitlines()[-1])
        assert (
            d["completed"] == n
            and d["total_input_tokens"] == 8192 * n
            and d["total_output_tokens"] == 1024 * n
        )

    bench("events", 256, 128)
    time.sleep(2)
    (out / "events_complete").touch()
    request = urllib.request.Request(
        url + "/start_profile",
        data=json.dumps(
            {
                "output_dir": str(out / "traces"),
                "num_steps": 5,
                "activities": ["CPU", "GPU"],
                "profile_by_stage": True,
                "with_stack": False,
                "record_shapes": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as r:
        (out / "profile-response.txt").write_bytes(r.read())
    bench("trace", 128, 0)
    time.sleep(5)
finally:
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
