# Real-QKV numerical reference diagnostic

Production GSM evaluation reported1218/1314 control versus1207/1314 candidate.
Its down-projection autotuner selected different M128 tactics (control1,
candidate4), so that difference is not isolated to attention. This diagnostic
uses identical real BF16 Q/K/V to compare the attention kernels themselves.

Dataset row13 was control-correct/candidate-incorrect. The exact first-five-shot
prompt and HF user chat-template token IDs are archived in prompt.json. A
single fresh858-token control forward captured layers0,17,35 after norm/RoPE,
with the existing TRT output. All4229 control Python files were checked against
058959cd-equivalent production. This does not reproduce the original128-way
batching, shared-prefix cache history, or autoregressive continuation.

GPU5 compared FA4 CLC with the captured TRT output on85 query positions spanning
the sequence. Reference QK, softmax and PV are FP32 with TF32 disabled; original
model weights, Q, K and V remain BF16. FP64 checks on the first4 query positions
agree with FP32 to relative L2 error1.4e-7–2.7e-7.

| Layer | TRT relative L2 | FA4 relative L2 |
| --- | ---: | ---: |
| 0 | 0.00156007 | 0.00156434 |
| 17 | 0.00169468 | 0.00171418 |
| 35 | 0.00165777 | 0.00163137 |

Both paths have mean cosine similarity approximately0.9999985. Against rounded
BF16 reference, median ULP difference is0, p90 is1, and p99 is5–6. Maximum ULP
counts are large around near-zero/sign-changing outputs; the report retains
them together with absolute error rather than interpreting them as large
relative tensor error.

There is no systematic meaningful FA4 error increase across these sampled
layers. Accordingly no rescale-threshold modification or new full evaluation
was performed. This limited result does not prove equal model accuracy.

Captured tensors remain remote at /root/qvl/experiments/fa4-numerics/layer*.pt;
their SHA256 values are in reference-report.json. Large tensor binaries are
excluded. Scripts are exact raw `.py.txt` archives; JSON only receives final
newlines. Capture used GPU4, reference GPU5; both processes completed.

`original-artifacts.tar.gz` preserves the archived files before repository formatting. Recorded artifact hashes refer to these extracted originals; readable copies may have whitespace normalized.
