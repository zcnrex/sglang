#!/usr/bin/env python3
"""Reproduce Qwen3-VL B300 serving sweeps with isolated output directories."""

import argparse
import json
import os
import pathlib
import socket
import subprocess
import time
import urllib.request

BASELINE = {1: 334, 4: 1125, 8: 1774, 16: 2499, 32: 3084, 64: 3494, 128: 3815}
SWEEP = [(1, 30), (4, 40), (8, 80), (16, 80), (32, 160), (64, 320), (128, 640)]
MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--python", default="python")
    parser.add_argument("--bench", default="sgl-bench")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is already in use")
    args.output.mkdir(parents=True, exist_ok=False)
    flags = args.server_args
    if flags[:1] == ["--"]:
        flags = flags[1:]
    command = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        *flags,
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    metadata = {
        "command": command,
        "gpu": args.gpu,
        "screen": args.screen,
        "baseline_source": "qwen3vl4b-b300-handoff.md, 2026-10-07",
        "vllm_output_tokens_per_second": BASELINE,
        "pythonpath": env.get("PYTHONPATH"),
        "hf_home": env.get("HF_HOME"),
    }
    for name, cmd in {
        "gpu_before": ["nvidia-smi"],
        "git_head": ["git", "rev-parse", "HEAD"],
        "git_diff": ["git", "diff"],
        "packages": [
            args.python,
            "-c",
            "import importlib.metadata as m; print(chr(10).join(sorted(d.metadata['Name'] + '==' + d.version for d in m.distributions())))",
        ],
    }.items():
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        (args.output / f"{name}.txt").write_text(result.stdout + result.stderr)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    server_log = (args.output / "server.log").open("w")
    pmon_log = (args.output / "pmon.log").open("w")
    server = subprocess.Popen(
        command,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (args.output / "server.pid").write_text(str(server.pid))
    pmon = subprocess.Popen(
        ["nvidia-smi", "pmon", "-i", str(args.gpu), "-s", "u", "-d", "1"],
        stdout=pmon_log,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{args.port}"
    results = []
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError("Server exited; inspect server.log")
            try:
                with urllib.request.urlopen(url + "/v1/models", timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Server readiness timeout")
                time.sleep(2)
        points = [(1, 3), (128, 256)] if args.screen else SWEEP
        for concurrency, count in points:
            warmup = (
                (1 if concurrency == 1 else 128)
                if args.screen
                else max(64, concurrency)
            )
            output = args.output / f"bench_c{concurrency}.jsonl"
            bench = [
                args.bench,
                "serve",
                "--random-range-ratio",
                "1",
                "--backend",
                "sglang-oai-chat",
                "--base-url",
                url,
                "--model",
                MODEL,
                "--dataset-name",
                "random",
                "--num-prompts",
                str(count),
                "--max-concurrency",
                str(concurrency),
                "--warmup-requests",
                str(warmup),
                "--flush-cache",
                "--random-input-len",
                "8192",
                "--random-output-len",
                "1024",
                "--output-file",
                str(output),
            ]
            with (args.output / f"bench_c{concurrency}.log").open("w") as log:
                subprocess.run(bench, stdout=log, stderr=subprocess.STDOUT, check=True)
            data = json.loads(output.read_text().splitlines()[-1])
            if data["completed"] != count:
                raise RuntimeError(
                    f"c={concurrency}: completed {data['completed']} of {count}"
                )
            for key, expected in (
                ("total_input_tokens", count * 8192),
                ("total_output_tokens", count * 1024),
            ):
                if data[key] != expected:
                    raise RuntimeError(
                        f"c={concurrency}: {key}={data[key]}, expected {expected}"
                    )
            result = {
                k: data[k]
                for k in (
                    "completed",
                    "total_input_tokens",
                    "total_output_tokens",
                    "output_throughput",
                    "median_ttft_ms",
                    "median_tpot_ms",
                )
            }
            result.update(
                concurrency=concurrency,
                versus_vllm=data["output_throughput"] / BASELINE[concurrency] - 1,
            )
            results.append(result)
            (args.output / "summary.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
    finally:
        pmon.terminate()
        pmon.wait(timeout=10)
        if not args.keep_server and server.poll() is None:
            import signal

            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        server_log.close()
        pmon_log.close()
        subprocess.run(
            ["nvidia-smi"],
            stdout=(args.output / "gpu_after.txt").open("w"),
            check=False,
        )


if __name__ == "__main__":
    main()
