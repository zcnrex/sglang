# B4 QKV fusion standalone screen

The bounded screen is positive on two GPUs, with no production changes or model/serving runs. It extends the proven B8 epilogue to four tokens by keeping the 128×8 shared redistribution and CTA barrier uniform, then masking token <4 before norm/shuffle and all global positions, slots, output and cache accesses. BF16 rounding and the GEMM tactic are unchanged. Positions have shape (3,4), stride (128,1).

Production maps (4,6144,2560) to (128,8,2,6), identical to B8. Installed direct preference requires K8192 and is false here. The reference calls the actual `_bf16_splitk_gemm_out` with its installed implementation helpers bound; assertions verify the map and direct predicate. No previous B4 QKV fusion result was found. The rejected B16 variant used a second token tile and does not predict this one-tile result.

| GPU | Reference sequence µs/layer | Fused sequence µs/layer | Latency reduction |
| --- | ---: | ---: | ---: |
| 0 | 34.0291 | 31.5464 | 7.296% |
| 1 | 33.9732 | 31.4786 | 7.343% |

All eight counterbalanced pairs on each GPU favored fusion. Each graph includes 36 real layer weights, QKV projection, norm/RoPE/cache preparation and dependent TRT decode attention. Weights exceed L2; each timing uses 100 graph replays. These are standalone CUDA-event timings, not serving throughput claims. GPU1 reverses the first timing order. Graphs and all inputs/callables remain alive throughout measurement.

On each GPU, all 36 layers pass bitwise QKV, complete K/V cache and attention equality, both initially and after changed inputs and changed positions with graph replay. Distinct rotary axes and a negative padding slot are covered. Caller output identity is asserted. Equality is against the actual production same-GEMM path, unlike the rejected B16 tactic's different reduction sequence.

Remote root: `/root/qvl/experiments/qkv-m4-screen`; terminal workers 670432 (GPU0) and 670676 (GPU1). Python `/root/qvl/venv-sgl/bin/python`, dependency source `/root/qvl/sglang-qkv-integrated/python`, `MAX_JOBS=4`, `CUDA_VISIBLE_DEVICES=0/1`, and `SCREEN_GPU=1` for confirmation. The exact pinned model path is in the script. Restore `.py.txt` files to `.py` in a fresh output directory to reproduce. Remote dependency hashes are recorded separately from local HEAD metadata: this screen does not claim the remote directory equals current HEAD.
