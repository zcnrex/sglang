import os
import pathlib
import subprocess

s = pathlib.Path("/root/qvl/experiments/m1-production-regression-low")
s.mkdir(exist_ok=True)
env = dict(
    os.environ,
    QVL_RUN_ROOT=str(s),
    QVL_CONTROL_SOURCE="/root/qvl/sglang-public-lt-clean",
    QVL_CANDIDATE_SOURCE="/root/qvl/sglang-m1-production",
    QVL_GPUS="0,1,2,3",
    QVL_CASES='{"0":[4,40],"1":[8,80],"2":[16,80],"3":[32,160]}',
    QVL_REQUESTS="30",
    QVL_CONCURRENCY="1",
    QVL_WARMUP="64",
    TRITON_CACHE_DIR="/root/.cache/sglang/triton",
    TORCHINDUCTOR_CACHE_DIR="/root/.cache/sglang/inductor",
    CUDA_CACHE_PATH="/root/.cache/sglang/nv",
    FLASHINFER_WORKSPACE_BASE="/root/.cache/sglang",
    SGLANG_CUTE_AOT_CACHE_DIR="/root/.cache/sglang/cute_aot",
)
log = (s / "run.log").open("w")
job = subprocess.Popen(
    ["/root/qvl/venv-sgl/bin/python", "/tmp/qvl-m1-production-regression.py"],
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("serving", job.pid, flush=True)
