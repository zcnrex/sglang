# Dynamic-M model correctness gate

One warmed exact B42/M16331 mixed-forward pair validated the startup-compiled symbolic-M callable before serving. The source audit checked 4229 files against current packed-prefix production. The geometry and synthetic seed 42 input protocol are recorded in the script/report; context query rows 16292 plus 39 genuine decode tails equal 16331.

All 36 MLP layers dispatched the dynamic fused kernel, with 36 packed FA4 calls. The prefill and three following decode logit comparisons were bitwise equal, with matching greedy outputs. Startup compiled one callable and transformed 36 weights; measured forwards performed zero compilation, input DLPack conversion or weight transforms. The single pair was 137.674 ms control versus 137.894 ms candidate and is a correctness gate, not a performance estimate or accuracy-equivalence claim.

Kernel identity/dependencies and full counters are preserved in gpu4/report.json and scripts. This gate proceeded to the separately archived dynamic_serving screen, which showed no end-to-end gain. No production code changed.
