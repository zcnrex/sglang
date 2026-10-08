# Integrated B4 QKV model gate

Passed against exact PR head 3016f3b5917650199c12b47195b9dc1be4c26e4e. Candidate `/root/qvl/sglang-qkv-m4-production` differs in exactly three files; complete 10,245-file manifests and the runtime-source manifests are retained. Public production dispatch was used, with observation only and no replacement algorithm hook.

Two seeds (123/124), B4, 8192 input tokens and 16 output tokens: full [4,151936] prefill, first-decode and fifteenth-decode logits were bitwise identical, as were all 64 generated tokens per seed. Both arms executed 30 graph4 decodes. Candidate observation confirms successful B4 eager and capture dispatch at all 36 layers. All 12 shared graph input buffers were asserted unchanged after extra replay.

Both arms used BF16 model/query/KV/output, TRT HND page32, mixed chunk16384, pool1,600,000, server seed0, and identical explicit graph buckets capped at128. Startup used normal public tuning. This is a short numerical/capture gate, not a serving or broad accuracy claim.

Observer limitation: the generic public-op layout observer deduplicates by capture/stride/result without including batch size. Its retained layout records show B8 calls with positions stride(128,1), rather than directly observing B4 stride. The B4 helper dispatch observer is correctly shape-qualified. Subsequent serving admission separately asserts successful B4 eager and capture calls for all36 layers with stride(128,1); no model rerun was made solely to add logging.

Raw archive preserves original scripts/logs/manifests. Full numerical outputs remain remote under `/root/qvl/experiments/qkv-m4-production/model`; SHA256 values are in `model/numerics_hashes.json`. Control PID672339 on GPU2 and candidate PID672404 on GPU3 completed normally. Readable scripts are renamed `.py.txt` only in this extracted copy.
