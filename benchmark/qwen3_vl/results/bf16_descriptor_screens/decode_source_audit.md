# Decode counter, output and launch audit

Read-only audit of current runtime and pinned devbox FlashInfer. No GPU run or
production modification. This does not replace the earlier raw NCU evidence.

## Counter initialization is already amortized

`TRTLLMHAAttnBackend.__init__` allocates a persistent
`_multi_ctas_kv_counter_buffer` with
`make_persistent_multi_ctas_kv_counter_buffer`. `_run_fixed_q_len_decode`
passes it to FlashInfer, as do the other decode entry points. Installed
FlashInfer `decode.py` documents that a provided buffer must initially be zero
and that the kernel resets counters at each launch's end. Its
`utils._resolve_trtllm_gen_multi_ctas_kv_counter_buffer` validates a supplied
buffer and returns it without zeroing. Removing initialization would violate
the contract; removing per-layer zeroing offers nothing because it is already
absent on this production path.

The earlier standalone NCU script omitted the optional persistent counter.
Its extra `FillFunctor<unsigned char>` kernel therefore reflects the default
FlashInfer allocation path, not redundant production workspace clearing.
Attention-kernel ID1 counters remain valid: the fill kernel is separately
reported as ID0. The standalone call latency includes that additional small
operation and should not be treated as an exact production-layer latency.

## Output and PDL

For BF16 query/cache the decode call requests BF16 output. FlashInfer allocates
`torch.empty_like(query, dtype=out_dtype)` when no output is provided; there is
no zero initialization or precision conversion. For the non-dense QKV-split
query view, ordinary empty-like allocation produces a dense output suitable
for the following output projection. Graph replay reuses captured allocations;
supplying a preallocated output alone would primarily affect eager allocation,
not remove a replay GPU kernel.

The runtime leaves `enable_pdl` unspecified. Installed FlashInfer resolves this
to `device_support_pdl(query.device)` and passes it into TRT runner parameters.
Thus PDL is not accidentally disabled. Prior on/off experiments already cover
this public knob. Source inspection alone does not demonstrate removable
inter-kernel launch gaps, and this audit makes no such claim.

No new safe, untested BF16 optimization emerged. Interleaving layer cache
allocations also has no identified mechanism to reduce mandatory per-layer
reads or improve the already measured94.15% sustained DRAM utilization.
