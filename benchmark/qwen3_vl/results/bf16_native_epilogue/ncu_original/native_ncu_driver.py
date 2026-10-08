import json
import os
import pathlib
import subprocess

r = pathlib.Path("/root/qvl/experiments/native-epilogue-ncu")
for kind in ["production", "candidate"]:
    e = os.environ.copy()
    e.update(
        CUDA_VISIBLE_DEVICES="4",
        PROFILE_KIND=kind,
        PYTHONPATH="/root/qvl/sglang-prefix-production/python",
        MAX_JOBS="8",
    )
    cmd = [
        "/usr/local/cuda/bin/ncu",
        "--profile-from-start",
        "off",
        "--launch-count",
        "1",
        "--set",
        "detailed",
        "--clock-control",
        "none",
        "--force-overwrite",
        "-o",
        str(r / kind),
        "/root/qvl/venv-sgl/bin/python",
        "-u",
        "/tmp/native_tail_ncu.py",
    ]
    (r / (kind + "-command.json")).write_text(json.dumps(cmd))
    with open(r / (kind + ".log"), "w") as f:
        p = subprocess.run(cmd, env=e, stdout=f, stderr=subprocess.STDOUT)
    print(kind, p.returncode, flush=True)
