import hashlib
import json
import os
import pathlib
import subprocess

r = pathlib.Path("/root/qvl/experiments/prefix-observer")
r.mkdir(exist_ok=True)
h = r / "hooks"
h.mkdir(exist_ok=True)
(h / "sitecustomize.py").write_text(
    "import runpy\nrunpy.run_path('/tmp/qvl-prefix-observer.py')\n"
)
source = "/root/qvl/sglang-public-lt-clean"
expected = "9db197749dd28b84f8af16c70187556d035fdaeab148675e043858c565be47f3"
assert (
    hashlib.sha256(
        pathlib.Path(
            source + "/python/sglang/srt/layers/quantization/unquant.py"
        ).read_bytes()
    ).hexdigest()
    == expected
)
(r / "provenance.json").write_text(
    json.dumps(
        {
            "source": source,
            "equivalent_commit": "5f60fb8b67",
            "unquant_sha256": expected,
            "observer_sha256": hashlib.sha256(
                pathlib.Path("/tmp/qvl-prefix-observer.py").read_bytes()
            ).hexdigest(),
            "attention_replacement": False,
            "measurement_not_performance_claim": True,
        },
        indent=2,
    )
)
env = dict(
    os.environ,
    QVL_RUN_ROOT=str(r),
    QVL_CONTROL_SOURCE=source,
    QVL_CANDIDATE_SOURCE=source,
    QVL_GPUS="0",
    QVL_REQUESTS="128",
    QVL_CONCURRENCY="128",
    QVL_WARMUP="128",
    TRITON_CACHE_DIR="/root/.cache/sglang/triton",
    TORCHINDUCTOR_CACHE_DIR="/root/.cache/sglang/inductor",
    CUDA_CACHE_PATH="/root/.cache/sglang/nv",
    FLASHINFER_WORKSPACE_BASE="/root/.cache/sglang",
    SGLANG_CUTE_AOT_CACHE_DIR="/root/.cache/sglang/cute_aot",
)
f = (r / "run.log").open("w")
p = subprocess.Popen(
    ["/root/qvl/venv-sgl/bin/python", "/tmp/qvl-prefix-run.py"],
    env=env,
    stdout=f,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(p.pid)
