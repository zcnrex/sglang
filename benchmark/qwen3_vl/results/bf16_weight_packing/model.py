import atexit
import copy
import json
import os
import pathlib

import numpy as np
import torch
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm import mm_bf16
from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
    default_tactic,
    run_direct_dense,
)

import sglang.benchmark.one_batch as b
import sglang.srt.layers.quantization.unquant as uq

M = int(os.environ["QVL_M"])
out = pathlib.Path("/root/qvl/experiments/weight-pack/model-m%d" % M)
out.mkdir(exist_ok=True)
active = False
counts = {"capture": 0, "eager": 0}
orig = uq._bf16_gemm_dispatch_impl


def dispatch(x, w, bias, addend=None):
    if (
        active
        and x.numel() == M * 2560
        and w.shape == (19456, 2560)
        and bias is None
        and addend is None
    ):
        y = torch.empty(M, 19456, device=x.device, dtype=x.dtype)
        if M == 1:
            run_direct_dense(
                x.reshape(M, 2560), w.t(), y, False, default_tactic(M, 19456, 2560)
            )
        else:
            mm_bf16(x.reshape(M, 2560), w.t(), out=y, backend="cute-dsl")
        counts["capture" if torch.cuda.is_current_stream_capturing() else "eager"] += 1
        return y.view(*x.shape[:-1], 19456)
    return orig(x, w, bias, addend)


uq._bf16_gemm_dispatch_impl = dispatch
report = {
    "m": M,
    "source": "5f60fb8b67 equivalent /root/qvl/sglang-public-lt-clean",
    "pairs": [],
}
saved = {}
origload = b.load_model


def choose(wrapper, mode):
    global active
    active = mode == "candidate"
    wrapper.torch_runner.decode_cuda_graph_runner.backend = saved[mode]


@torch.no_grad()
def load(*args, **kw):
    global active
    wrapper, tok = origload(*args, **kw)
    mr = wrapper.torch_runner
    dr = mr.decode_cuda_graph_runner
    assert dr.backend.__class__.__name__ == "FullCudaGraphBackend"
    w = next(
        mod.weight
        for mod in mr.model.modules()
        if hasattr(mod, "weight") and mod.weight.shape == (19456, 2560)
    )
    x = torch.randn(M, 2560, device=w.device, dtype=torch.bfloat16)
    if M == 4:
        with autotune(tuning_buckets=(M,), round_up=False):
            mm_bf16(x, w.t(), backend="cute-dsl")
        AutoTuner.get().save_configs(str(out / "cache.json"))
    else:
        run_direct_dense(
            x,
            w.t(),
            torch.empty(M, 19456, device=w.device, dtype=w.dtype),
            False,
            default_tactic(M, 19456, 2560),
        )
    saved["baseline"] = dr.backend
    candidate = copy.copy(dr.backend)
    candidate._graphs = {}
    candidate._outputs = {}
    candidate._output_buffer = None
    dr.backend = candidate
    active = True
    dr.capture()
    saved["candidate"] = candidate
    assert counts["capture"] >= 36, counts
    report["capture_dispatches"] = dict(counts)
    report["graph_keys"] = {k: [str(s) for s in v._graphs] for k, v in saved.items()}
    inputs = np.random.default_rng(123).integers(0, 10000, (M, 8192), dtype=np.int32)
    captures = {}
    seqs = {}
    for mode in ["baseline", "candidate"]:
        choose(wrapper, mode)
        wrapper.clear()
        reqs = b.prepare_synthetic_inputs_for_latency_test(M, 8192, inputs.tolist())
        ids, logits, batch = wrapper.extend(reqs)
        seq = [ids.clone()]
        vals = []
        for j in range(31):
            ids, logits = wrapper.decode(ids, batch)
            seq.append(ids.clone())
            if j in [0, 30]:
                vals.append(logits.clone())
        torch.cuda.synchronize()
        seqs[mode] = torch.stack(seq).cpu()
        captures[mode] = [v.cpu() for v in vals]
    report["greedy_equal"] = torch.equal(seqs["baseline"], seqs["candidate"])
    report["greedy_differences"] = (seqs["baseline"] != seqs["candidate"]).sum().item()
    report["logit_checks"] = [
        {
            "bitwise": torch.equal(a, z),
            "max_abs": (a - z).abs().max().item(),
            "nrms": ((a - z).square().mean() / a.square().mean()).sqrt().item(),
        }
        for a, z in zip(captures["baseline"], captures["candidate"])
    ]
    torch.save({"logits": captures, "tokens": seqs}, out / "numerics.pt")
    choose(wrapper, "baseline")
    return wrapper, tok


b.load_model = load
origonce = b.latency_test_run_once
calls = 0


def once(*args, **kw):
    global calls
    calls += 1
    wrapper = args[1]
    if calls == 1:
        result = None
        for mode in ["baseline", "candidate"]:
            choose(wrapper, mode)
            np.random.seed(123)
            a = list(args)
            a[3] = b.prepare_synthetic_inputs_for_latency_test(M, 8192)
            result = origonce(*a, **kw)
        return result
    result = None
    for r in range(8):
        order = ["baseline", "candidate"] if r % 2 == 0 else ["candidate", "baseline"]
        row = {}
        for mode in order:
            choose(wrapper, mode)
            np.random.seed(123)
            a = list(args)
            a[0] = mode
            a[3] = b.prepare_synthetic_inputs_for_latency_test(M, 8192)
            result = origonce(*a, **kw)
            row[mode] = result
        report["pairs"].append(row)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return result


b.latency_test_run_once = once
atexit.register(
    lambda: (out / "state.json").write_text(
        json.dumps({"counts": counts, "report": report}, indent=2) + "\n"
    )
)
np.random.seed(123)
torch.manual_seed(123)
b.cli_main()
