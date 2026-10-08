# Integrated QKV kernel validation

The production shared-engine hook passes the 36-layer standalone dependent-attention gate. Initial and changed-input retained-graph outputs are bitwise equal for complete QKV output, full K/V cache buffers, and downstream TRT attention. All eight timing pairs favor the integrated kernel: median full sequence time is 52.644755 → 51.911223 µs per layer, a 1.393% reduction. This is a standalone sequence measurement, not serving throughput.

The same source passes all 77 existing hc_mix and kernel namespace tests. An external probe passes 27 wrapper/model-guard checks. The model-guard portion extracts the actual method AST into controlled fake runtime state; it proves the individual guard branches, not complete model integration. Independent actual-model validation is archived separately.

## Scope and source

Candidate source: `/root/qvl/sglang-qkv-integrated`, copied from frozen `/root/qvl/sglang-m16-down-production` and overlaid with exactly four owned files. `source_manifest.json` records the sequence-test hashes. After that worker was terminal, the public wrapper gained one current-device mismatch fallback check; `final_source_manifest.json` records this final snapshot, used by the guard probes and independent model validation. Frozen source copies are included as `.py.txt` files. No live source was changed.

macOS tar also transferred four AppleDouble `._*` metadata files alongside source. They are non-runtime artifacts; the independent model source audit records them separately. They are not source changes or imported modules.

## Sequence gate

- GPU 4 was verified empty before launch. Worker PID 626652 is terminal.
- All 36 actual model QKV/norm weights from pinned revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`; 1,132,462,080 bytes of distinct projection weights.
- B8, Q32/KV8/D128, BF16 input/weights/output/cache, HND page32, exact tactic (128,8,2,6), PDL enabled.
- Reference uses installed FlashInfer GEMM plus the existing norm/MRoPE/cache writer and TRT decode. Candidate uses the integrated shared-engine epilogue plus the same TRT decode.
- Three distinct position axes, one negative slot, caller-owned output pointer identity, random finite cache contents, and whole-cache comparisons. Changed inputs and positions are replayed through retained graphs.
- Eight counterbalanced timing rounds; 100 graph replays per arm per round, each graph covering all 36 layers. Startup compilation is excluded. Metadata and all tensors remain alive. Distinct weights/caches rotate across layers; no per-call explicit L2 flush is claimed.
- `sequence.json`, `sequence.log`, and `sequence.py.txt` preserve measurements and protocol.

## Existing tests

Worker PID 626936 completed:

```sh
CUDA_VISIBLE_DEVICES=4 MAX_JOBS=4 PYTHONPATH=/root/qvl/test-deps:/root/qvl/sglang-qkv-integrated/python /root/qvl/venv-sgl/bin/python -m pytest -q test/registered/kernels/ops/gemm/test_hc_mix.py test/registered/unit/kernels/test_kernels_namespace.py
```

Result: **77 passed in 40.89 seconds**. The existing hc_mix cases exercise BF16/FP16, row tails, SiLU and gate behavior; namespace tests cover registration and import boundaries. No test files were added to the production patch.

## Fallback probes

`guards.py.txt` and `guards.json` preserve 27 checks:

- Actual wrapper rejection of unsupported rows, FP32 input/output, noncontiguous input, cache/output overlap, wrong axis dtype, mismatched current device, gradient-enabled execution, outer compilation, and a cold graph-capture cache miss.
- Actual nondefault-stream execution preserves caller-owned output identity.
- AST-isolated model checks cover supported zero-copy parameter detach and disabled split-K/backend/autotune, deterministic mode, speculative configuration/batch, LoRA, CP/DCP, bias, prefill, unsupported model flag and RL target.

The wrapper's rejection during outer compilation was tested by a mocked compile-state predicate; this is not an end-to-end torch.compile model test. The dispatch falls back before the custom kernel. Independent model/serving gates remain separate evidence.

## Generated-code comparison limitation

A bounded optional comparison of original versus hook-disabled generated SASS/ABI did not obtain retained compiler artifacts. The first attempt stopped before kernel compilation because automatic SASS dumping requires nvdisasm ≥13.4, while the pinned CUDA toolkit is older. Two subsequent attempts requested only PTX/cubin, including one with `CUTE_DSL_NO_CACHE=1`; the compiled object still exposed no artifacts, so the audit stopped at an explicit assertion. Logs and extraction harness are retained. These are diagnostic tooling failures, not CUDA faults or numerical failures. No binary-identical or generated-ABI-identical claim is made. Existing public signatures, source review, successful default plain-kernel prerequisite, and the 77 passing tests provide the available compatibility evidence.

`raw_reports.tar.gz` preserves the unnormalized remote JSON/log originals; visible text copies may have trailing whitespace or final newlines normalized by repository hooks.
