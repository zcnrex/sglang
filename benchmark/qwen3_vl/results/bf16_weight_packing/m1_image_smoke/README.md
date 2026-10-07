# Production M1 direct image smoke

Pass for this single fixture: control and candidate return identical text and
32 greedy token/byte entries. Maximum absolute chosen-token logprob difference
is 0.015081878751516342. Both process 234 prompt tokens, including 216 image
tokens. This is a basic multimodal smoke, not broad image/video accuracy.

The pinned repository image has SHA256
`e06917184a00b14abd70cd8ea0ff5dca9abfbbad29f7b25c02f97133d4cd060e`.
Prompt: “Describe this image in one short sentence.” Temperature zero, top-p one,
32 maximum output tokens, chosen-token logprobs enabled. Model snapshot is
`ebb281ec70b05090aa6165b016eac8ec08e71b17`.

Source audit compares 4661 Python/CUDA/C++ source files per copy. Only intended
`unquant.py` differs: candidate SHA256
`0fa01b10b34bec46f4cca14b6dfb96fa67a07d94e68c1d08035b8c851740f431`.
Control is `/root/qvl/sglang-public-lt-clean`; candidate is
`/root/qvl/sglang-m1-production`. All weights, KV and queries remain BF16.

Observation-only instrumentation calls the existing production helper unchanged.
Candidate reports 36 distinct weight buffers, 36 direct calls during graph
capture, and batch-one graph replay; control reports zero direct calls. The
marker records the first graph-one replay, not a per-request trace.

GPU6 control server PID308481 and GPU7 candidate PID307985 completed the request
and were terminated by the harness's ownership-scoped cleanup. Later shutdown
crash-diagnostic messages in server logs follow that explicit termination;
both launchers exited successfully after recording their responses. Final GPU
inventory showed no jobs on GPUs6/7; unrelated GPU0–3 workers were preserved.

`run.py` contains the full request and launch/cleanup protocol. Fresh per-run
tuner caches coexist with shared compiled caches; exact commands/environment,
source audits, response JSON, logs and markers are archived. No production
source was changed by this experiment.
