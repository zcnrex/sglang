# Integrated B1/B2 QKV fusion kernel gate

The formatted three-file extension passes integrated standalone correctness, guard and namespace checks. M1/M2 use one iteration with a whole-warp token guard after uniform shared redistribution/barrier. M4/M8 retain one/two iterations; a compile-time true left operand short-circuits the token-bound condition. All original BF16 arithmetic and safety guards remain.

| Batch / GPU | Reference µs/layer | Candidate µs/layer | Reference |
| --- | ---: | ---: | --- |
| B1 / GPU0 | 21.7746 | 19.8081 | Production GEMM plus preparation |
| B2 / GPU1 | 26.6869 | 24.0738 | Production GEMM plus preparation |
| B4 / GPU6 | 31.3605 | 31.3465 | Committed M4 wrapper |
| B8 / GPU7 | 51.9206 | 51.9362 | Committed M8 wrapper |

All timings include dependent TRT attention over 36 real rotating layer weights. Eight counterbalanced pairs use 100 retained graph replays each, with all cache slots valid before timing. B1/B2 reductions are 9.031%/9.792%; B4/B8 differences are -0.0445%/+0.0300%. The tiny regression differences do not establish binary identity or exact zero overhead.

B1/B2 each pass 144 layer checks: initial reference equality, frozen masked-prototype reference equality, changed-input/stride128-position graphs with sentinel slots, and restored-all-valid-slot graphs before timing. B4/B8 each pass 108 checks against committed wrappers. QKV, full caches and dependent attention are bitwise equal in every check. Twenty-seven actual-wrapper guards cover all four exact shapes, unsupported3/5/6/7, stride/alias/current-device/grad/compile constraints, warm capture reuse and cold-layout fallback. Existing namespace tests:67 passed, one unknown-config warning.

Installed DSL source excerpts/hashes in constexpr_source.json show visit_BoolOp generates a Python short-circuit expression: a true Python bool on the OR left side returns without evaluating the right side. const_expr(self.num_tokens>=4) is a compile-time Python bool. This supports removal of the token comparison for M4/M8 at DSL evaluation; no generated-binary equality claim is made. The successful four specializations and timings provide independent runtime coverage.

Immutable source /root/qvl/sglang-qkv-small-production differs from the frozen M4 source in exactly three intended files across10,245 manifest entries. Initial admission caught inherited .pyc files in the new copy; only those copied bytecode files were removed before any job launch, with paths retained. The original M4 source was untouched. Final workers686961/686962/686963/686964 are terminal. Source hashes were frozen after formatting and before launch.

Remote evidence root /root/qvl/experiments/qvl-qkv-small-integrated. Scripts run with /root/qvl/venv-sgl/bin/python, PYTHONPATH=/root/qvl/sglang-qkv-small-production/python, PYTHONDONTWRITEBYTECODE=1, MAX_JOBS=4 and the indicated visible GPU. original.tar.gz preserves unnormalized remote evidence; readable copies may have whitespace normalized. Full model/serving validation is separate.
