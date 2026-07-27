"""Race K3 router top-16-of-896: native CUDA radix vs Triton vs tokenspeed packed-key.

`packed_key_sigmoid_topk` ports tokenspeed's trick (`triton/kimi3_sigmoid_topk.py`):
pack an order-preserving bit-transformed fp32 score and an *inverted* expert id into
one uint64 so a SINGLE `tl.topk` reproduces "descending score, lower-id tiebreak",
instead of 16 sequential argmax rounds.

NOTE `moe_fused_gate` short-circuits to `route_radix` for any covered K3 input, so the
`sglang_triton` line monkeypatches `moe_route_radix.covered` (restored in `finally`) to
reach a real Triton kernel. That means it measures a path production never takes for K3.

Measured GB300 (SM103), GPU 2, triton 3.6.0, torch 2.11.0+cu130, median us.
`sglang_radix` here is `sorted=True` (apples-to-apples ordering); production uses
`sorted=False`, which is faster still:

   num_tokens |   sglang_radix(us)  sglang_triton(us)  packed_key_topk(us) |   sglang_radix(GB/s)  sglang_triton(GB/s)  packed_key_topk(GB/s)
0           1 |             3.3702             7.3840               3.8614 |                 2.02                 0.92                   1.76
1           8 |             3.3702             7.2614               3.9642 |                 9.20                 4.27                   7.82
2          32 |             3.4723             7.3635               4.3942 |                32.82                15.48                  25.93
3         128 |             3.6096             7.4867               4.5373 |               123.52                59.55                  98.26
4         512 |             4.1690             9.6570               9.9850 |               425.37               183.64                 177.60
5        2048 |             9.8538            26.1508              28.8415 |               718.85               270.87                 245.60

Reading it:
  - The packed-key trick NEVER beats the native radix router. It beats Triton 1.6-1.9x
    at 1-128 tokens (its design regime) but loses to radix everywhere, and crosses over
    to losing to *Triton* at >=512: one program per token with a 1024-element uint64
    bitonic top-k is register-hungry, so once there are enough tokens to fill the GPU,
    occupancy dominates. `num_warps` swept {2,4,8,16} -- no config wins.
  - radix's "3.1-3.5x vs Triton" docstring claim measures 2.2-2.9x here.
  - GB/s columns are near-meaningless at small M -- rows 0-3 are launch-bound (~3.4 us
    floor). Only the 2048 row is a real throughput comparison.

Tiebreak semantics: all three agree (lowest expert id on equal biased score).
One real divergence found and fixed in the port: tokenspeed's ordered-key transform maps
NaN ABOVE +inf, so a NaN logit would be force-selected; SGLang's Triton router maps NaN
to -1e30. The port now matches SGLang.

Bug in the sibling bench: `bench_moe_route_radix.py`'s `_triton` provider claims fp32
fails radix `covered()` -- false (it accepts fp32, and stride(0)=896 % 4 == 0), so both
its lines dispatch into `route_radix`. That bench measures radix-vs-radix.

Relevance caveat: on a real TP8 bs=1 decode trace the whole routing family
(`route_radix` + trtllm-gen `routingIndicesClusterKernel` + `pack_topk`) is ~1111 us/step
summed but only ~138 us *exposed*, so router work is ~1% of the critical path.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.moe import moe_route_radix
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.kernels.ops.moe.moe_route_radix import route_radix
from sglang.srt.layers.moe.topk_packed_key import packed_key_sigmoid_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NUM_EXPERTS = 896
TOPK = 16
SCALE = 2.5


def _sglang_radix(logits, bias):
    return route_radix(logits, bias, TOPK, True, SCALE, True, sorted=True)


def _sglang_triton(logits, bias):
    # moe_fused_gate short-circuits to route_radix for covered K3 inputs; the
    # guard has to be neutralized to reach the Triton kernel it claims to beat.
    covered = moe_route_radix.covered
    moe_route_radix.covered = lambda *args, **kwargs: False
    try:
        return moe_fused_gate(
            logits,
            bias,
            topk=TOPK,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=SCALE,
            apply_routed_scaling_factor_on_output=True,
        )
    finally:
        moe_route_radix.covered = covered


def _packed_key(logits, bias):
    return packed_key_sigmoid_topk(
        logits,
        bias,
        topk=TOPK,
        renormalize=True,
        routed_scaling_factor=SCALE,
        apply_routed_scaling_factor_on_output=True,
    )


FN_MAP = {
    "sglang_radix": _sglang_radix,
    "sglang_triton": _sglang_triton,
    "packed_key_topk": _packed_key,
}


@marker.parametrize("num_tokens", [1, 8, 32, 128, 512, 2048], [32])
@marker.benchmark("provider", ["sglang_radix", "sglang_triton", "packed_key_topk"])
def benchmark(num_tokens: int, provider: str):
    torch.manual_seed(42)
    logits = create_random(num_tokens, NUM_EXPERTS, dtype=torch.float32)
    bias = create_random(NUM_EXPERTS, dtype=torch.float32)
    return marker.do_bench(FN_MAP[provider], input_args=(logits, bias))


def check() -> None:
    torch.manual_seed(0)
    for num_tokens in (1, 3, 32, 512):
        logits = create_random(num_tokens, NUM_EXPERTS, dtype=torch.float32)
        bias = create_random(NUM_EXPERTS, dtype=torch.float32)
        out = {name: fn(logits, bias) for name, fn in FN_MAP.items()}
        ref_w, ref_i = out["sglang_triton"]
        ref_set = ref_i.sort(dim=-1).values
        for name, (w, i) in out.items():
            assert torch.equal(i.sort(dim=-1).values, ref_set), f"{name} expert set"
            gathered = torch.gather(w, 1, i.sort(dim=-1).indices) - torch.gather(
                ref_w, 1, ref_i.sort(dim=-1).indices
            )
            assert gathered.abs().max() < 2e-5, f"{name} weights {gathered.abs().max()}"
    print("router top-k impls agree on expert set and combine weights")


if __name__ == "__main__":
    check()
    benchmark.run()
