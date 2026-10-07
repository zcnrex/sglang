import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root/qvl/experiments/lossless-kv")
import pair_decode
from lossless import pack

sys.path.insert(0, "/root/qvl/experiments/lossless-attention")
import decode as old

b = 1
n = 8192
s = 8
bn = 128
torch.manual_seed(1)
q = torch.randn(b, 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(b * n // 32, 8, 32, 128, device="cuda", dtype=torch.bfloat16)
p, meta = pack(k)
a = torch.empty(b * 8 * s * 4 * 128, device="cuda")
m = torch.empty(b * 8 * s * 4, device="cuda")
l = torch.empty_like(m)
o = torch.empty_like(q)
rows = []
for name, mod, packed in [
    ("raw", old, False),
    ("old", old, True),
    ("pair", pair_decode, True),
]:
    c = mod._decode[(b * 8, s)](
        q,
        p if packed else k,
        p if packed else k,
        meta,
        meta,
        a,
        m,
        l,
        o,
        n,
        8,
        4,
        32,
        s,
        bn,
        packed,
        num_warps=8,
        num_stages=2,
    )
    ptx = c.asm["ptx"]
    Path(f"/root/qvl/experiments/pair-unpack/{name}.ptx").write_text(ptx)
    row = dict(
        name=name,
        regs=c.n_regs,
        spills=c.n_spills,
        shared=c.metadata.shared,
        instructions={
            x: len(re.findall(re.escape(x), ptx))
            for x in [
                "ld.global.b8",
                "ld.global.b16",
                "prmt.b32",
                "ld.shared",
                "st.shared",
            ]
        },
    )
    rows.append(row)
    print(json.dumps(row), flush=True)
Path("/root/qvl/experiments/pair-unpack/resources.json").write_text(
    json.dumps(rows, indent=2)
)
