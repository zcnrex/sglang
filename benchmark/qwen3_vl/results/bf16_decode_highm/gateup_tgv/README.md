# High-M gate-up TGV screen

PhysicalGPU0, pinned devbox environment, seed20261007. BF16 M64/128,
N19456,K2560. Four distinct rotating weights total398458880bytes, exceeding
L2. Six alternating CUDA-graph timing rounds. All tested outputs match Torch
bitwise. PDLfalse for the isolated CuTeTGV call. No production edits.

Installed direct BF16 and SM100 split-K kernels explicitly reject M>32;
unsupported tactics were not forced. Twelve supported TGV cases used IDs
1,9,10,14,15,28 at each M. Wider M tiles and one/two-CTA configurations were
represented; this is a bounded subset, not an exhaustive optimum claim.

| M | Torch us | Best tested TGV us | Tactic |
|---:|---:|---:|---:|
|64|18.841|25.754|9|
|128|22.298|31.322|28|

No improvement. No serving test justified for these candidates.
