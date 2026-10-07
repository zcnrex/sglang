import flashinfer
import torch

from sglang.srt.utils.cuda_vmm_utils import *

torch.cuda.set_device(0)
torch.manual_seed(20261007)
B, L, H, D, P = 128, 8192, 8, 128, 32
shape = (B * L // P, H, P, D)
nbytes = B * L * H * D * 2
props = []
caches = {}
for pad in [0]:
    stride = (H * P * D + pad, P * D, D, 1)
    caches[str(pad)] = tuple(
        torch.empty_strided(shape, stride, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    props.append(
        dict(padding_elements=pad, stride=stride, capacity_overhead=pad / (H * P * D))
    )
q = torch.randn(B, 32, D, device="cuda", dtype=torch.bfloat16)
ws = torch.empty(512 * 1024**2, device="cuda", dtype=torch.uint8)
table = torch.arange(B * L // P, device="cuda", dtype=torch.int32).reshape(B, L // P)
lens = torch.full((B,), L, device="cuda", dtype=torch.int32)
outs = {k: torch.empty_like(q) for k in caches}


def call(k):
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=caches[k],
        workspace_buffer=ws,
        block_tables=table,
        seq_lens=lens,
        max_seq_len=L,
        bmm1_scale=D**-0.5,
        bmm2_scale=1.0,
        out=outs[k],
    )


for t in caches["0"]:
    t.normal_()
for _ in range(10):
    call("0")
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(10):
        call("0")
for _ in range(5):
    g.replay()
a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
a.record()
for _ in range(10):
    g.replay()
z.record()
z.synchronize()
print("unprofiled_control_us", a.elapsed_time(z) * 10, flush=True)
torch.cuda.nvtx.range_push("steady_decode")
call("0")
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
