# Production M32 vocabulary model proof

Candidate d0d528db4f passes B32/input128 numerical checks against frozen M8+M16 production. Two distinct input seeds123/124 each generate16 tokens per request. Prefill, first/fifteenth decode logits and every generated token are bitwise equal. This bounded check avoids an unnecessary giant prefill; it does not establish full accuracy equivalence.

Control is `/root/qvl/sglang-m16-down-production`; candidate `/root/qvl/sglang-lmhead-m32-production` changes only logits_processor.py and runner/flashinfer_autotune.py. All4229 local Python source hashes match the candidate. Remote baseline additionally contains two non-runtime `.claude/skills` Python scripts; audit.json records their exact paths/hashes and verifies identical bytes in the overlay. Complete remote manifests cover4231 files. No source was deleted to satisfy the comparison.

The initial candidate stopped before model loading because its expected manifest omitted those two inherited artifacts; its logs remain under candidate-initial-manifest-failure. Live control was preserved. Only the terminal candidate was retried with the explicit complete manifest. Control worker535293 and successful candidate536893 completed. Initial failed candidate535294 also remains documented.

No algorithm hook is used. Observer records the actual tied BF16 unquantized model, normal public startup READY including M32, optimized M32 capture and graph32 replay. Large outputs.pt tensors remain remote; exact comparison results, scripts, logs, manifests and caches are archived. No serving or accuracy result is part of this artifact.
