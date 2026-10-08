# External B1/B2 QKV fusion screen

Both valid-slot standalone screens are positive. No production files changed. One fixed split-K tactic (128,8,2,6) matches actual production B1/B2 selection. The external epilogue uses one iteration and masks inactive whole warps after uniform shared redistribution and the CTA barrier. All BF16 rounding stages are preserved.

| Batch / GPU | Reference µs/layer | Fused µs/layer | Latency reduction |
| --- | ---: | ---: | ---: |
| B1 / GPU0 | 21.7798 | 19.7993 | 9.093% |
| B2 / GPU1 | 26.6902 | 24.0904 | 9.740% |

All eight alternating pairs are positive for each batch. The timing includes QKV, preparation/cache stores and dependent TRT attention across 36 rotating actual layer weights. Each timing uses 100 retained graph replays. This does not establish model or serving gains.

Each final run passes 108 bitwise layer checks for QKV, full K/V caches and attention: initial state, changed inputs/stride128 positions with a negative slot, and all-valid slots restored immediately before timing. Final timing slots are [8200] for B1 and [8200,16456] for B2. The original sentinel-state runs are retained under sentinel_diagnostic; they are not the admission result because B1's negative slot skipped all cache writes. The report field negative_slot describes correctness coverage, not the final timing state; timing_slots.json and valid_sequence.py.txt explicitly record the latter.

Remote root /root/qvl/experiments/qvl-qkv-small; final workers679039/679040 are terminal. Use valid_sequence.py.txt restored as valid_sequence.py beside prototype.py, interpreter /root/qvl/venv-sgl/bin/python, PYTHONPATH=/root/qvl/sglang-qkv-m4-production/python, MAX_JOBS=4, PYTHONDONTWRITEBYTECODE=1, CUDA_VISIBLE_DEVICES=0 or1. B2 uses SCREEN_GPU=1. The script pins the exact model snapshot. The unchanged dependency source is the previously audited PR3016 three-file M4 overlay; the external candidate bypasses only its unsupported B1/B2 wrapper eligibility. Separate model-gate artifacts will be recorded independently.
