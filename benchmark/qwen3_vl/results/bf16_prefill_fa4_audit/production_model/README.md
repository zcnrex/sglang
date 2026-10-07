# Production prefix-packing model gate

The frozen control matched all4229 Python files at058959cd. The isolated
candidate differed in exactly the three files recorded in source.json.
Both runs asserted the complete candidate manifest before CUDA initialization.
The maintained CSR-index and packing operators replace the external prototype;
FA4 receives per-call CLC selection without mutating the process global.

GPUs6/7 each ran four common warmups followed by ABBA measurement, rebuilding
cache state independently for each variant. Context prefix/query tuples were
[7968,0,0]/[232,8200,7872], followed by17 genuine cached8192-token decode tails.
Baseline disables the new metadata route; candidate enables it. Timings include
per-batch CSR construction, dispatch checks, packing and workspace lookup.
Scratch allocation is amortized after warmup, bounded to128 MiB per backend.

Final runtime-guard revision results:

| GPU | Control ms | Candidate ms | Latency reduction |
| --- | ---: | ---: | ---: |
| 6 | 134.3187 | 132.3588 | 1.459% |
| 7 | 133.6967 | 131.8711 | 1.365% |

Each run executed144 candidate packing/FA4 calls with actual CLC cache-key
proof. Packed24272 K/V rows and original layer0 cache writes passed bitwise
checks. All20 requests across4 greedy positions matched. Logits remain
non-bitwise, including last-decode maximum difference1.3125 and NRMS0.028171;
these bounded results do not establish task-accuracy noninferiority.

CPU guard checks execute the unchanged production metadata method and dispatch
expression with mocks: disabled route, alternate stream, short/empty/oversized
contexts, multimodal and speculative batches, absent optional dependencies,
and actual dispatch capture/stream changes all fall back. Missing imports
disable further attempts; execution errors are not swallowed. Both overlap
modes are statically excluded. These checks do not substitute for GPU validation
of unsupported modes; they establish that those modes do not enter this route.

The scripts are raw `.txt` archives; JSON reports only receive final newlines.
No serving throughput claim is made here. Production serving and full accuracy
validation are separate gates.

GPU4 additionally exercised the unchanged extracted production context callable
with real CUDA streams and capture: eligible metadata pinned to streamA fell
back when invoked on streamB, and capture on streamA also fell back. The TRT
fallback was a tiny GPU-add stub; no packed operator or scratch was accessed.
This tests dispatch safety directly, not full-model capture correctness.

Original pre-format artifact bytes are retained in `original-artifacts.tar.gz`. Recorded raw hashes apply to files extracted from that archive; readable copies may have whitespace normalized by repository hooks.
