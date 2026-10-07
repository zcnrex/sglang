import hashlib
import json
import os
import pathlib
import subprocess

r = pathlib.Path("/root/qvl/experiments/m1-production-full-accuracy")
r.mkdir(exist_ok=True)
p = pathlib.Path("/tmp/test.jsonl")
data = p.read_bytes()
assert (
    hashlib.sha256(data).hexdigest()
    == "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
)
rows = data.splitlines(keepends=True)
assert len(rows) == 1319
mapping = {}
for i in range(2):
    ids = list(range(5 + i * 657, 5 + (i + 1) * 657))
    q = r / f"shard{i}.jsonl"
    q.write_bytes(b"".join(rows[:5] + [rows[j] for j in ids]))
    mapping[str(i)] = {
        "original_rows": ids,
        "local_rows": list(range(5, 662)),
        "sha256": hashlib.sha256(q.read_bytes()).hexdigest(),
    }
(r / "shards.json").write_text(
    json.dumps(
        {
            "original_sha256": hashlib.sha256(data).hexdigest(),
            "fewshot_original_rows": list(range(5)),
            "shards": mapping,
        },
        indent=2,
    )
)
assert set(mapping["0"]["original_rows"]) | set(mapping["1"]["original_rows"]) == set(
    range(5, 1319)
)
for gpu, variant in [
    (4, "candidate0"),
    (5, "control0"),
    (6, "candidate1"),
    (7, "control1"),
]:
    log = (r / f"{variant}-driver.log").open("w")
    cmd = [
        "/root/qvl/venv-sgl/bin/python",
        "/tmp/qvl-m1-production-gsm.py",
        "--source",
        (
            "/root/qvl/sglang-m1-production"
            if variant.startswith("candidate")
            else "/root/qvl/sglang-public-lt-clean"
        ),
        "--gpu",
        str(gpu),
        "--variant",
        variant,
    ]
    job = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    print(variant, job.pid, flush=True)
s = pathlib.Path("/root/qvl/experiments/m1-production-serving")
s.mkdir(exist_ok=True)
env = dict(
    os.environ,
    QVL_RUN_ROOT=str(s),
    QVL_CONTROL_SOURCE="/root/qvl/sglang-public-lt-clean",
    QVL_CANDIDATE_SOURCE="/root/qvl/sglang-m1-production",
    QVL_GPUS="0,1,2,3",
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
    ["/root/qvl/venv-sgl/bin/python", "/tmp/qvl-m1-production-serving.py"],
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("serving", job.pid, flush=True)
