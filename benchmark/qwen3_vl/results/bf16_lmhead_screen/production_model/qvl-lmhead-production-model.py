import hashlib
import json
import os
import pathlib

import numpy as np
import torch

import sglang.benchmark.one_batch as bench

root = pathlib.Path(os.environ["QVL_MODEL_OUT"])
root.mkdir(parents=True, exist_ok=True)
source = pathlib.Path(os.environ["QVL_SOURCE"])
expected = json.load(open(os.environ["QVL_EXPECTED_MANIFEST"]))
actual = {
    str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (source / "python").rglob("*.py")
}
assert actual == expected
saved = {}
report = {"source_files_verified": len(actual), "source": str(source), "seed": 123}
original = bench.load_model


@torch.no_grad()
def load(*args, **kwargs):
    runner, tok = original(*args, **kwargs)
    data = np.random.default_rng(123).integers(0, 10000, (4, 8192), dtype=np.int32)
    reqs = bench.prepare_synthetic_inputs_for_latency_test(4, 8192, data.tolist())
    ids, logits, batch = runner.extend(reqs)
    saved["prefill"] = logits.detach().cpu()
    seq = [ids.cpu()]
    for step in range(15):
        ids, logits = runner.decode(ids, batch)
        seq.append(ids.cpu())
        if step in [0, 14]:
            saved[f"decode{step}"] = logits.detach().cpu()
    saved["greedy"] = torch.stack(seq)
    torch.save(saved, root / "outputs.pt")
    report["greedy"] = saved["greedy"].tolist()
    report["graph_backend"] = type(
        runner.torch_runner.decode_cuda_graph_runner.backend
    ).__name__
    (root / "report.json").write_text(json.dumps(report, indent=2))
    return runner, tok


bench.load_model = load
# The load hook is the test; avoid an additional timing experiment.
bench.latency_test_run_once = lambda *args, **kwargs: {"latency": 0}
np.random.seed(123)
torch.manual_seed(123)
bench.cli_main()
