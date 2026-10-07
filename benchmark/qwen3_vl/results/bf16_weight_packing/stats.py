import glob
import json

import torch
from safetensors import safe_open

root = "/root/qvl/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
r = []
for path in glob.glob(root + "/*.safetensors"):
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            if not any(
                ".layers." + str(l) + "." in name for l in [0, 17, 35]
            ) or not any(
                name.endswith(s + ".weight")
                for s in ["q_proj", "o_proj", "gate_proj", "down_proj"]
            ):
                continue
            x = f.get_tensor(name).cuda()
            bits = x.view(torch.int16).to(torch.int32) & 65535
            e = (bits >> 7) & 255
            entry = {"name": name, "shape": list(x.shape), "groups": {}}
            for size in [8, 32, 64, 128]:
                eg = e.reshape(-1, size)
                lo = eg.min(1).values
                hi = eg.max(1).values
                valid = hi < 255
                d = hi - lo
                entry["groups"][size] = {
                    str(width): ((d < (1 << width)) & valid).float().mean().item()
                    for width in [2, 3, 4]
                }
            r.append(entry)
            print(entry, flush=True)
open("/root/qvl/experiments/weight-pack/stats.json", "w").write(
    json.dumps(r, indent=2) + "\n"
)
