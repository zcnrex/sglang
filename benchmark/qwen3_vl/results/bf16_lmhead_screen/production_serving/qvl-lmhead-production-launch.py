import os
import pathlib
import subprocess

r = pathlib.Path("/root/qvl/experiments/lmhead-production")
base = "/root/qvl/sglang-prefix-production"
candidate = "/root/qvl/sglang-lmhead-production"
env = os.environ.copy()
env.update(
    TRITON_CACHE_DIR="/root/.cache/sglang/triton",
    TORCHINDUCTOR_CACHE_DIR="/root/.cache/sglang/inductor",
    CUDA_CACHE_PATH="/root/.cache/sglang/nv",
    FLASHINFER_WORKSPACE_BASE="/root/.cache/sglang",
    SGLANG_CUTE_AOT_CACHE_DIR="/root/.cache/sglang/cute_aot",
    QVL_RUN_ROOT=str(r / "serving"),
    QVL_CONTROL_SOURCE=base,
    QVL_CANDIDATE_SOURCE=candidate,
    QVL_CONTROL_MANIFEST=str(r / "control_expected.json"),
    QVL_CANDIDATE_MANIFEST=str(r / "candidate_expected.json"),
    QVL_GPUS="4,5",
    QVL_REQUESTS="40",
    QVL_WARMUP="64",
    QVL_CONCURRENCY="4",
)
cmd = ["/root/qvl/venv-sgl/bin/python", "-u", "/tmp/qvl-lmhead-production-serving.py"]
f = open(r / "serving-driver.log", "w")
p = subprocess.Popen(
    cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
)
print("serving", p.pid, flush=True)
for gpu, variant, source in [(6, "control", base), (7, "candidate", candidate)]:
    e = os.environ.copy()
    e["QVL_EXPECTED_MANIFEST"] = str(r / (variant + "_expected.json"))
    cmd = [
        "/root/qvl/venv-sgl/bin/python",
        "-u",
        "/tmp/qvl-lmhead-production-accuracy.py",
        "--gpu",
        str(gpu),
        "--variant",
        variant,
        "--source",
        source,
    ]
    f = open(r / (variant + "-gsm-driver.log"), "w")
    p = subprocess.Popen(
        cmd, env=e, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
    )
    print(variant + "-gsm", p.pid, flush=True)
