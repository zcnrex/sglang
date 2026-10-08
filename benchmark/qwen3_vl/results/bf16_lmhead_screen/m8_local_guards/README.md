# M8 local admission probes

AST-isolated metadata mocks exercise the actual committed dispatch/startup functions. Twelve logits cases cover M4/M8 admission, M1/M16 fallback and existing precision/compile/LoRA/quantization/READY exclusions. Eleven startup exclusions reject both M4 and M8; tuning-bucket and deferred-READY checks retain M128 and verify failed-verifier fallback. These are local branch/metadata probes, not real Torch, CUDA, numerical, or serving validation. Remote production validation is separate.
