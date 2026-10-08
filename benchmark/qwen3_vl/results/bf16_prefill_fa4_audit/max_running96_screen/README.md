# max-running-requests96 bounded screen

Rejected for throughput. Same-source GPU-swapped pairs compare the current committed prefix candidate with default max-running requests versus explicit96. No prior equivalent max96 experiment was found. The 4,229-file candidate manifest matched before launch; GPU inventory was empty. No production source changes or algorithm hooks were installed.

C128 clients, N128 measured requests,128 warmup requests, random8192 input/1024 output, BF16 weights/query/KV, mixed16k,HNDpage32; fresh tuner perworker/sharedcompiledcaches. Same unchanged established serving harness except candidate command adds --max-running-requests96. ParentPID399526; actual A servers399532/399531, B402337/402338. Both phases completed and owned servers were cleaned up.

|GPU|Default tok/s|Max96 tok/s|Change|TTFT default→96 ms|TPOT default→96 ms|
|---|---:|---:|---:|---:|---:|
|0|3820.044|3514.444|−8.000%|4494.990→4503.097|28.8292→23.4531|
|1|3816.217|3502.432|−8.223%|4560.509→4559.470|28.7952→23.6400|

All four workers completed exactly128requests,1,048,576inputtokens and131,072outputtokens. Limiting concurrency improves TPOT but leaves TTFT approximately unchanged and reduces throughput by about8.1%. This N128 short screen is a rejection gate, not a full steady-state sweep. Startup M128 tactics were allowed to autotune independently and are preserved in tactics.json and original archive; no matched-tactic numerical claim is made.

Raw archive preserves original logs, JSONL, source hashes/commands, probes, telemetry, startup tuner JSON and PIDs without normalization. No compiled artifacts or bytecode included. No wider sweep follows this negative result.
