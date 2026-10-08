# BF16 normalization exact-order refinement

Both bounded variants retain the installed CuTe residual normalization arithmetic, layouts, FP32 reduction helper, BF16 rounding and PDL dependency waits. Neither changes an installed module or shared compiler cache. They compile independent local objects for hidden size 2560 and epsilon 1e-6.

Disabling async input staging preserves bitwise output and residual values in five numerical regimes and two changed-input graphs containing 36 disjoint layer buffers at each of B8/B16/B128. B8/B16 timing is flat near 2.108 us; B128 changes from approximately 2.222 to 2.190 us. This small standalone difference does not justify integration.

Relocating the dependency launch signal after input/residual loading and addition preserves default staging and arithmetic. It also passes the same bitwise checks, plus ten changed-seed graphs per batch with 36 producer/PDL-waiting-consumer pairs. Timing regresses: approximately 2.108 to 2.221 us at B16 and 2.222 to 2.297 us at B128. This variant is rejected.

Timing uses eight counterbalanced CUDA-event rounds, each replaying a retained 36-buffer graph twenty times. Numerical checks precede timing. These are standalone measurements, not serving results. The earlier independent reduction had larger standalone gains but failed strict full-model numerical validation; neither exact-order follow-up recovers that gain. No normalization production promotion is recommended.

The public normalization API exposes no staging option. The prototypes use the installed CuTe class and exact compiler signature; the early-signal producer is an external source copy with a single signal relocation. Production use of such an implementation would need a maintained API or explicitly guarded integration, but these negative results make that unnecessary.
