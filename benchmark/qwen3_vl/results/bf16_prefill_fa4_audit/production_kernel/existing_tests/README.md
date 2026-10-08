# Existing targeted regression tests

All selected tests passed against frozen /root/qvl/sglang-prefix-production:

- TRT page-layout tests and namespace checks: 5 passed, 4 subtests passed, 64 deselected in 19.14 seconds. This covers packed/flat slot strides, HND pass-through, lazy imports, registered target existence and reclassified entry-point inventory.
- Four existing SM103 FA4 cases: 4 passed in 48.92 seconds on GPU4, worker PID376605. Cases cover BF16 GQA D128 causal/noncausal 256-token attention, causal 4096-token prefill, and chunked prefill. Each existing test compares row-max implementations and reference numerical tolerance.

Existing test files were copied unchanged from the repository into the external run directory because the frozen production copy omits the test tree. Their hashes and exact commands are adjacent. No production files or repository test files were changed. Dependency deprecation/DSL compilation warnings were emitted; no tests failed or skipped.

These existing cases exercise default FA4 behavior, not explicit CLC True, the new pack path, or all backend eligibility exclusions. The adjacent standalone production_kernel evidence covers explicit CLC and bitwise token-major/HND packing separately. No dedicated existing pack numerical test was found. This was a targeted selection, not the complete namespace/attention suite or a training/backward validation.

Verified frozen production SHA256:
- dllm_kv_pack.py: 237a1ea2a36e5fcb6b727225efa4a60d86dbe953edde10f8d6c9eb24aa9e73b4
- flash_attn/cute/interface.py: 9d842bca4c3182562246f290f42479ae644605f4b5b54b0f402030488346b964
- trtllm_mha_backend.py: 4896aa04b862ef0d926b9f0fcb489fcce095aeccb20a2a579be6fa8f85d8476f

`original-artifacts.tar.gz` preserves the archived files before repository formatting. Recorded artifact hashes refer to these extracted originals; readable copies may have whitespace normalized.
