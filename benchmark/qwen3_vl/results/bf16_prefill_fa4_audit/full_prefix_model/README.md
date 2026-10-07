# Full-prefix packing mixed-model gate

Both GPU4/5 runs verified all 4229 Python files against audited 5f60fb8b67.
Each comparison rebuilt independent cache state: context prefixes [7968,0,0]
and query lengths [232,8200,7872], followed by 17 genuine decode requests
with 8192 cached tokens. Original HND page32 writes and TRT decode remain.
Four warmups precede ABBA measured variants. Forward wall time includes
packing and Python bookkeeping; scratch is reused after warmup.

GPU4 control/candidate means: 133.7446/130.4438 ms (2.47% lower).
GPU5: 130.9587/128.8054 ms (1.64% lower). These are bounded model screens,
not serving throughput measurements.

Layer0 cache writes matched bitwise for all four warmup variants; packed
24272 K/V rows matched bitwise in both candidate checks. Actual CLC cache-key
selection was recorded, with 144 FA4 calls per run. All 20 requests across
four greedy positions matched. Logits were not bitwise: maximum differences
were 0.125/0.125/0.1875/1.3125; normalized RMS errors were
0.00257/0.00185/0.00192/0.02817. The last decode discrepancy requires the
separate serving accuracy gate; matching these greedy tokens is insufficient.

The deployable hook adds conservative guards and bounded reusable storage:
text-only BF16, Q32/KV8/D128, HND page32, fresh stride6144, at least4096 query
rows, at most32768 total KV rows and16384 per request. Scratch grows to at
most128 MiB per device, rather than retaining one allocation per shape.
Missing K/V, empty/short contexts, and over-bound requests were checked to
fall back without entering FA4. Telemetry appends to per-PID files and skips
empty states, including selected CLC keys, prefix histogram, packed/query
counts, fallback reasons, and maximum scratch bytes.

Raw scripts are preserved unchanged as `.txt`; hash metadata covers their
bytes. The serving hook is a guarded derivative of the validated model core;
serving validation is separate. No production changes were made.

Original artifact bytes are preserved in `original-artifacts.tar.gz`; `hashes.json` verifies the extracted files. Uncompressed copies may have whitespace normalized by repository hooks.
