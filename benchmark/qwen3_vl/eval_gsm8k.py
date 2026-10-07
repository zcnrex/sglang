"""Reproduce the Qwen3-VL accuracy gate against an existing server.

Run with the SGLang environment's Python (including its evaluation dependencies:
openai, httpx, numpy, pandas, jinja2, requests, and tqdm), not the isolated
sgl-bench client environment. Point PYTHONPATH at this checkout's python/.

Example:
    PYTHONPATH=python python benchmark/qwen3_vl/eval_gsm8k.py \
        --base-url http://127.0.0.1:32002 --output /tmp/qwen3-vl-accuracy

The scorer and chat sampler match sglang.test.run_eval. Reports are written
straight to the requested directory, avoiding that module's shared /tmp names.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.num_shots < 0 or args.num_threads < 1 or args.max_tokens < 1:
        parser.error(
            "shots must be nonnegative; threads and max-tokens must be positive"
        )
    if args.num_examples is not None and args.num_examples < 1:
        parser.error("num-examples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"
    # Reserve the result path before requests so parallel runs cannot collide.
    with metrics_path.open("x") as output:
        output.write('{"status": "running"}\n')

    from sglang.test.run_eval import run_eval_once
    from sglang.test.simple_eval_common import make_report
    from sglang.test.simple_eval_mixed_prefix_gsm8k import (
        GSM8K_URL,
        GSM8KEval,
        get_answer_value,
    )
    from sglang.utils import download_and_cache_file, read_jsonl

    data_path = args.data_path or Path(download_and_cache_file(GSM8K_URL))
    rows = list(read_jsonl(str(data_path)))
    selected = rows[args.num_shots :]
    if args.num_examples is not None:
        selected = selected[: args.num_examples]
    if not selected:
        raise ValueError(
            "No held-out examples remain after selecting few-shot examples"
        )
    evaluator = GSM8KEval(
        num_examples=args.num_examples,
        num_threads=args.num_threads,
        num_shots=args.num_shots,
        data_path=str(data_path),
    )
    args.api = "chat"
    args.temperature = 0.0
    args.top_p = 1.0
    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    result, latency, sampler = run_eval_once(args, base_url, evaluator)
    outcomes = [
        {
            "dataset_row": args.num_shots + index,
            "correct": get_answer_value(conversation[-1]["content"])
            == get_answer_value(row["answer"]),
        }
        for index, (row, conversation) in enumerate(zip(selected, result.convos))
    ]
    if len(outcomes) != len(selected):
        raise RuntimeError("Evaluation returned an unexpected number of conversations")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    metrics = {
        "status": "complete",
        "model": sampler.model,
        "source_revision": revision,
        "scorer": "sglang.test.simple_eval_mixed_prefix_gsm8k.GSM8KEval",
        "dataset_url": GSM8K_URL,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "selection": {
            "seed": None,
            "method": "dataset order; first num_shots rows excluded from evaluation",
            "num_shots": args.num_shots,
            "first_evaluation_row": args.num_shots,
            "last_evaluation_row": args.num_shots + len(selected) - 1,
            "row_indexing": "zero-based",
        },
        "sampling": {
            "api": "chat",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "concurrency": args.num_threads,
        },
        "total": len(outcomes),
        "correct": sum(item["correct"] for item in outcomes),
        "score": float(result.score),
        "latency_seconds": latency,
        "completion_tokens": sum(sampler._completion_tokens),
        "outcomes": outcomes,
    }
    (args.output / "report.html").write_text(make_report(result))
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
