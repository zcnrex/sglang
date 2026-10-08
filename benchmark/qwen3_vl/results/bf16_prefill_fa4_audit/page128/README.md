# Direct paged FA4 page128 gate: rejected before model

Both GPUs2/3 verified all4229 Python hashes against the frozen production
candidate. BF16 throughout, Q32/KV8/D128; native HND caches on both paths.
Logical K/V conversion matched bitwise. Page128 direct FA4 retained TMA and
CLC (constructor proof in reports), and produced bitwise-identical outputs
to page32 packing plus FA4 across three measured heterogeneous prefix batches.
Both include the maintained cache writer; writer cost was effectively equal.
Cold median context savings were27.6–61.5 us, with substantial within-run
clock drift. Raw counterbalanced rounds are retained.

Current TRT decode used production persistent counters, B128, lengths8192/9216,
and two rotating native HND caches per layout (8–9 GiB total, far beyond L2).
Six alternating rounds showed page128 consistently slower by2.48–3.77 us,
with bitwise output equality. This revisits the previous page128 test's
NHD-backed permuted layout only to assess the new direct-paged prefill path.

Using historical131 mixed steps and1985 decode steps,36 layers, mean saved
context time is approximately0.20 s while decode adds approximately0.22 s.
This already lacks a demonstrated weighted benefit and excludes losing the
current page32-only fused norm/RoPE/cache writer, plus page allocation/radix
reuse changes. No model or serving run was justified. The counts are an
illustrative bound, not a current end-to-end performance prediction.

Exact scripts are `.py.txt`; source manifest hash and FlashInfer version are
in reports. No production changes. Both GPU workers completed.

`original-artifacts.tar.gz` preserves raw files before repository formatting; recorded hashes refer to extracted originals.
