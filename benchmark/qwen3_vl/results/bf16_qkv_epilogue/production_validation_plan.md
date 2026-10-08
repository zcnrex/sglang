# Integrated production validation plan

The production integration reuses a different in-tree GEMM engine; external prototype results do not establish its correctness or speed.

1. After integrated standalone QKV/cache/dependent-attention checks pass, freeze the exact four-file overlay and full source manifest against frozen M16 production. Run the public integrated model on GPU3 and unchanged control on GPU2, with an observer that records successful helper dispatch without replacing computation. Use B8, input8192/output16, seeds123/124, BF16, graph cap8 and KV capacity1,600,000. Compare complete prefill/first/fifteenth decode logits and all generated tokens. Assert all36 layers captured, exact graph8 and unchanged input buffers through replay. Preserve failures.
2. Only after that gate passes and parent authorizes serving, run a two-phase C8/N80 production crossover on GPUs2/3: warm64, flush, nominal8192/1024, cap8, capacity1,600,000, mixed chunk16384 and normal public startup tuning. Include `--output-details` so per-request TTFT/order can be inspected. Record exact counts, source/cache hashes, successful fused capture, actual M128 participation/tactics and TTFT/TPOT mean/median/p95/p99. No external algorithm replacement.
3. Report production serving results before full GSM accuracy or PR promotion. Run the full paired C8 accuracy gate only after authorization. Keep integrated evidence separate from external prototype archives and preserve normal startup results if a separately authorized matched-startup diagnostic is needed.

No repeated external serving is planned. The external best1928.099 tokens/s is still below1951.4 target; neither it nor retained-graph timings establish completion of the overall goal.
