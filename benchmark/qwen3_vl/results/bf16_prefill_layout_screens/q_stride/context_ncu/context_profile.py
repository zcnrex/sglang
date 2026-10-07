import flashinfer
import torch

torch.manual_seed(20261007)
rows = []
# Trace preserves M/B, not individual lengths. These are synthetic compatible geometries.
for qlens, klens in [([8200, 8007], [8200, 8007])]:
    m = sum(qlens)
    tail = 124
    total = m + tail
    backing = torch.randn(total, 6144, device="cuda", dtype=torch.bfloat16)
    qs = backing[:, :4096].view(total, 32, 128)
    qc = qs.contiguous()
    lens = torch.tensor(klens, device="cuda", dtype=torch.int32)
    maxpages = (max(klens) + 31) // 32
    k = torch.randn(2 * maxpages, 8, 32, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.arange(2 * maxpages, device="cuda", dtype=torch.int32).reshape(
        2, maxpages
    )
    cuq = torch.tensor([0, qlens[0], m], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, klens[0], sum(klens)], device="cuda", dtype=torch.int32)
    ws = torch.empty(512 * 1024**2, device="cuda", dtype=torch.uint8)
    out = torch.empty(m, 32, 128, device="cuda", dtype=torch.bfloat16)

    def call(q):
        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q[:m],
            kv_cache=(k, v),
            workspace_buffer=ws,
            block_tables=table,
            seq_lens=lens,
            max_q_len=max(qlens),
            max_kv_len=262144,
            bmm1_scale=128**-0.5,
            bmm2_scale=1.0,
            batch_size=2,
            cum_seq_lens_q=cuq,
            cum_seq_lens_kv=cuk,
            out=out,
        )

    for _ in range(10):
        call(qs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(5):
            call(qs)
    for _ in range(5):
        g.replay()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):
        g.replay()
    b.record()
    b.synchronize()
    print("control_us", a.elapsed_time(b) * 20, flush=True)
    torch.cuda.nvtx.range_push("steady_context")
    call(qs)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
