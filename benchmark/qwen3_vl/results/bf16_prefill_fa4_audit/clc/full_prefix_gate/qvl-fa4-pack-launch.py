import hashlib
import json
import os
import pathlib
import subprocess

hook = pathlib.Path("/root/qvl/experiments/fa4_prefix_serving_hook.py")
assert (
    hashlib.sha256(hook.read_bytes()).hexdigest()
    == "4fe16403fdf4c050455fbefb78a8ceaa90abe994852d42541ea1fc56ab8677ef"
)
hooks = pathlib.Path("/root/qvl/experiments/fa4-pack-hooks")
hooks.mkdir(exist_ok=True)
(hooks / "sitecustomize.py").write_text(
    pathlib.Path("/tmp/qvl-fa4-pack-sitecustomize.py").read_text()
)
source = "/root/qvl/sglang-public-lt-clean"
assert (
    hashlib.sha256(
        pathlib.Path(
            source + "/python/sglang/srt/layers/quantization/unquant.py"
        ).read_bytes()
    ).hexdigest()
    == "9db197749dd28b84f8af16c70187556d035fdaeab148675e043858c565be47f3"
)
r = pathlib.Path("/root/qvl/experiments/fa4-pack-accuracy")
r.mkdir(exist_ok=True)
for gpu, v in [(6, "control"), (7, "candidate")]:
    f = (r / f"{v}-driver.log").open("w")
    env = dict(os.environ, QVL_CANDIDATE_HOOK_DIR=str(hooks))
    p = subprocess.Popen(
        [
            "/root/qvl/venv-sgl/bin/python",
            "/tmp/qvl-fa4-pack-accuracy.py",
            "--source",
            source,
            "--gpu",
            str(gpu),
            "--variant",
            v,
        ],
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(v, p.pid, flush=True)
s = pathlib.Path("/root/qvl/experiments/fa4-pack-serving")
s.mkdir(exist_ok=True)
env = dict(
    os.environ,
    QVL_RUN_ROOT=str(s),
    QVL_CONTROL_SOURCE=source,
    QVL_CANDIDATE_SOURCE=source,
    QVL_CANDIDATE_HOOK_DIR=str(hooks),
    QVL_GPUS="0,1,2,3",
    QVL_REQUESTS="128",
    QVL_CONCURRENCY="128",
    QVL_WARMUP="128",
    TRITON_CACHE_DIR="/root/.cache/sglang/triton",
    TORCHINDUCTOR_CACHE_DIR="/root/.cache/sglang/inductor",
    CUDA_CACHE_PATH="/root/.cache/sglang/nv",
    FLASHINFER_WORKSPACE_BASE="/root/.cache/sglang",
    SGLANG_CUTE_AOT_CACHE_DIR="/root/.cache/sglang/cute_aot",
)
f = (s / "run.log").open("w")
p = subprocess.Popen(
    ["/root/qvl/venv-sgl/bin/python", "/tmp/qvl-fa4-pack-serving.py"],
    env=env,
    stdout=f,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("serving", p.pid, flush=True)
for root in [r, s]:
    (root / "hook-provenance.json").write_text(
        json.dumps(
            {
                "source": source,
                "source_equivalent_commit": "5f60fb8b67",
                "m1_excluded": True,
                "hook_sha256": hashlib.sha256(hook.read_bytes()).hexdigest(),
                "hook_path": str(hook),
                "accuracy_only_disable_radix": False,
                "telemetry_wrapper_sha256": hashlib.sha256(
                    (hooks / "sitecustomize.py").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
