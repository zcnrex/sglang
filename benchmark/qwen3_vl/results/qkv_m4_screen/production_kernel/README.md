# Integrated M4/M8 kernel gate

The formatted three-file production extension passes the standalone gate. M4 uses one compile-time epilogue iteration; M8 uses two with the existing body unchanged. Shared redistribution/barriers remain uniform. Both tests use graph-buffer positions stride (128,1), distinct axes, negative cache slots and caller-owned output buffers.

| GPU / batch | Reference | Candidate | Change |
| --- | ---: | ---: | ---: |
| GPU0 M4 | 34.0172 µs/layer | 31.3524 µs/layer | 7.834% lower latency |
| GPU1 M8 | 51.9451 µs/layer | 51.9686 µs/layer | 0.0453% higher latency |

Times cover QKV projection, norm/RoPE/cache writing and dependent TRT decode attention over 36 cold rotating actual layer weights, eight counterbalanced CUDA-event pairs and 100 retained graph replays per measurement. M4 improves in all pairs. M8 is slightly slower in all pairs by about 0.024 µs/layer; this is a small measured difference, not proof of binary identity or exactly zero overhead.

M4 passes 108 layer checks: initial and changed-input/changed-position graphs versus actual production unfused preparation, plus the frozen masked external prototype versus the same reference. Every check is bitwise for QKV, full K/V caches and downstream attention. M8 passes 72 equivalent checks directly against the committed M8 wrapper, including changed-input graphs. Existing namespace tests pass: 67 tests, with one existing unknown-pytest-config warning.

Immutable candidate `/root/qvl/sglang-qkv-m4-production` is exact PR3016 source plus the three intended files. The 10,245-entry manifest verifies no missing or extra files. Hashes were frozen after successful pre-commit and before launch. Workers 671473 (M4/GPU0) and 671474 (M8/GPU1) are terminal. Scripts load byte-identical candidate module copies outside the source tree for kernel isolation; actual runtime/model dispatch is tested separately.

Run m4/m4.py.txt or m8/m8.py.txt restored to .py beside their module files, with `/root/qvl/venv-sgl/bin/python`, `PYTHONPATH=/root/qvl/sglang-qkv-m4-production/python`, `PYTHONDONTWRITEBYTECODE=1`, `MAX_JOBS=4` and the corresponding visible GPU. M8 sets `SCREEN_GPU=1` for reversed initial order. Namespace command: `PYTHONPATH=/root/qvl/test-deps:/root/qvl/sglang-qkv-m4-production/python python -m pytest -q -p no:cacheprovider test/registered/unit/kernels/test_kernels_namespace.py`.
