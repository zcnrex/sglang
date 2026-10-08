# Down projection numerical triage

No production changes or serving run. Worker PID 465494 completed on GPU 0. Sixteen real five-shot GSM prompts use examples 0–4 and questions 5–20 from the existing cached test dataset (SHA256 recorded in report.json). Actual tokenizer lengths are 821–897, not 8K. Both graph variants start from fresh identical prefill; at all 16 decode steps the candidate receives the baseline-generated token, so logit comparisons have matching token histories. This differs from the prior free-running synthetic 8K probe.

255/256 top1 choices agree. One differs at step 5. Per-step normalized RMS logit error ranges 0.00832–0.01772; largest absolute difference is 0.796875. Outputs are not bitwise equal. Agreement on this small teacher-forced sample does not establish model accuracy or numerical equivalence. The earlier 24/256 synthetic free-running token differences remain valid evidence and are not erased by this check. Large raw teacher logits remain at `/root/qvl/experiments/m16-down-teacher/down/teacher_logits.pt`.

## Arithmetic and layout audit

Installed `flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py` SHA256 is `a9c050cc7fa56f508da2bd969653f323a68421a49b38f8a8ac620ff371f55090`. Header and lines 763–810 show FP32 per-slice accumulators, FP32 DSMEM partial exchange/reduction, then one BF16 output conversion. There is no BF16 rounding of partial sums. This is the same implementation used by accepted M1/2/4/8 down projection tactics; the new M16 tactic changes kernel-N width 8→16 and stages 6→5 while retaining split-K 4. Different FP32 summation ordering versus Torch can change BF16 rounding. Source structure supports that explanation, but is not proof that all observed differences are benign.

The external call uses x[M,9728], weight.T[9728,2560], BF16 out[M,2560], no bias and PDL enabled, matching the existing production split-K API. Real rotating-weight standalone checks showed small errors (NRMS 0.0001312, max abs 0.03125) and changed-input replay passed. There is no evidence of wrong addressing/layout from those checks. No same-activation measurement of the accepted M8 route was performed; comparison to accepted paths here is an implementation/precision comparison, not an empirical error-bound equivalence. Wider accuracy evaluation is a separate decision.

## Exact harness hashes

- `qvl-m16-down-teacher.py`: `7bc869848439677a652b65c0346f5c9861689a0934a52bc3b5818137474e8486`
- `qvl-m16-down-teacher-launch.py`: `cf98fbd23d4a7bd28f4343eb6dc831437ca2ccc88a8c750adfb5cb1ab330fced`
