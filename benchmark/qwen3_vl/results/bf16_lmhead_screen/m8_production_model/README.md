# Production M8 LM-head model validation

Committed overlay5babd15f4f passes B8 numerical checks against frozen production M4 control. For two distinct input seeds123/124, each with8K prefill and16 generated tokens per request, prefill plus first/fifteenth decode logits and every generated token match bitwise. This is bounded deterministic model evidence, not full accuracy equivalence.

Candidate `/root/qvl/sglang-lmhead-m8-production` has exactly two changed Python files versus control `/root/qvl/sglang-lmhead-production`: logits_processor.py and runner/flashinfer_autotune.py. All4229 candidate source hashes match local committed production. The observer records model identity, tied BF16 unquantized embedding, startup READY containing M8, actual optimized M8 capture and graph8 replay. It replaces no algorithms or tuning policy. Public startup tuning and cache records are retained.

GPU2 control worker495623 and GPU3 candidate495624 completed successfully. Original model tensors remain remote under `/root/qvl/experiments/lmhead-m8-production/model/{control,candidate}/outputs.pt`; comparison.json records exact checks. Raw archive preserves scripts, manifests, logs and cache JSON. No serving result is included here.
