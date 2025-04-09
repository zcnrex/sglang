"""
Bench the sglang-hosted vLM with benchmark MMMU

Usage:
    Host the VLM: python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template qwen2-vl --port 30000

    Benchmark: python benchmark/mmmu/bench_sglang.py --port 30000 --batch-size 4

The eval output will be logged
"""

import argparse
import time
from typing import List

import openai
from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    prepare_samples,
    process_result,
)
from tqdm import tqdm

from sglang.test.test_utils import add_common_sglang_args_and_parse


def process_batch(
    client, batch_samples: List[dict], sampling_params: dict
) -> List[str]:
    """Process a batch of samples and return their responses."""
    messages = []
    for sample in batch_samples:
        prompt = sample["final_input_prompt"]
        prefix = prompt.split("<")[0]
        suffix = prompt.split(">")[1]
        image_path = sample["image_path"]

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prefix,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_path},
                    },
                    {
                        "type": "text",
                        "text": suffix,
                    },
                ],
            }
        )

    response = client.chat.completions.create(
        model="default",
        messages=messages,
        temperature=0,
        max_completion_tokens=sampling_params["max_new_tokens"],
        max_tokens=sampling_params["max_new_tokens"],
    )

    return [choice.message.content for choice in response.choices]


def eval_mmmu(args):
    eval_args = EvalArgs.from_cli_args(args)

    out_samples = dict()
    sampling_params = get_sampling_params(eval_args)
    samples = prepare_samples(eval_args)
    answer_dict = {}

    client = openai.Client(api_key="sk", base_url=f"http://127.0.0.1:{args.port}/v1")

    start = time.time()
    batch_size = args.batch_size

    for i in tqdm(range(0, len(samples), batch_size)):
        batch_samples = samples[i : i + batch_size]
        responses = process_batch(client, batch_samples, sampling_params)

        for sample, response in zip(batch_samples, responses):
            process_result(response, sample, answer_dict, out_samples)

    print(f"Benchmark time: {time.time() - start}")

    args.output_path = f"./val_sglang.json"
    save_json(args.output_path, out_samples)
    eval_result(model_answer_path=args.output_path, answer_dict=answer_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    EvalArgs.add_cli_args(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples to process in each request",
    )
    args = add_common_sglang_args_and_parse(parser)
    args = parser.parse_args()

    eval_mmmu(args)
