# Production pack and per-call CLC standalone gate

The maintained pack_prefix_current HND extension passed bitwise K/V and cumulative-length checks against the exact external full-prefix pack. The original token-major default passed the same comparisons. Cases cover zero prefixes, prefixes 31/64 across page boundaries, and measured prefix [7968,0,0] with extend [232,8200,7872]. Fresh input stride is (6144,128,1), cache is HND page32, and physical pages are shuffled. GPU4 worker PID355779.

Six alternating CUDA-graph timing rounds give representative medians 34.856 us for the CSR production pack and 33.705 us for the direct-page external pack. These are warm repeated replays: a 384 MiB flush occurs once before each 100-replay block, not before every replay. CSR metadata creation is outside these kernel timings and must be included in the subsequent model gate. No full attention/serving performance claim follows from this test.

Per-call FA4 use_clc_scheduler None/False/True/None passes BF16 numerical checks on 512 tokens, Q32/KV8/D128; outputs happened to be bitwise equal, caller output identity was preserved, the global default remained false, and the compiled cache contained distinct false/true CLC fields. FakeTensorMode None/False/True paths also passed. Final GPU5 worker PID356405. The existing FlashAttnVarlenFunc has no backward implementation; this change does not add training support. The initial external cache-inspection harness mistakenly iterated the cache object and failed after its first successful invocation; the corrected archived harness uses cache.cache. This was a harness failure, not a kernel failure.

Production files frozen after pre-commit passed:
- dllm_kv_pack.py SHA256 237a1ea2a36e5fcb6b727225efa4a60d86dbe953edde10f8d6c9eb24aa9e73b4
- flash_attn/cute/interface.py SHA256 9d842bca4c3182562246f290f42479ae644605f4b5b54b0f402030488346b964

Remote run directory: /root/qvl/experiments/pack-production. Python: /root/qvl/venv-sgl/bin/python. The isolated interface loaded unchanged dependencies from /root/qvl/sglang-public-lt-clean/python (audited 5f60fb8b67). No running serving source was modified. Scripts are preserved as .py.txt; rename to execute together. Pre-commit passes for the two production files.

Original pre-format artifact bytes are retained in `original-artifacts.tar.gz`. Recorded raw hashes apply to files extracted from that archive; readable copies may have whitespace normalized by repository hooks.
