# Short LM-head model gate

External hook only; frozen `/root/qvl/sglang-prefix-production` source was used. No production change. Same loaded model retains its original projection tactics and both full decode graph backends. Public Lt tuning uses the actual tied embedding weight once before candidate graph capture. Exact LM-head shape and BF16/default unquantized path are guarded; downstream logits handling is unchanged. Baseline/candidate get fresh identical 8K prompts and matching decode lengths. Four alternating pairs, 16 output tokens, opposite initial order for B16. Timings are the existing one-batch host-inclusive median decode latencies, not CUDA-event kernel measurements.

| Batch | Paired decode speedups (%) | Median (%) |
|---|---|---|
| 4 | -1.587, -5.547, +6.295, -4.223 | -2.905 |
| 8 | +4.533, -0.130, +0.090, -0.961 | -0.020 |
| 16 | -2.984, -0.599, -2.888, -0.239 | -1.743 |

All three batches recorded one actual candidate LM-head graph-capture dispatch. First and fifteenth decode logits were bitwise equal (max absolute error and NRMS zero); all 16 greedy outputs per request matched after independent fresh prefills. These checks use real model activations, not just the standalone random inputs.

The short host-inclusive gate does not establish a model speedup; no serving or GSM run has been run. Standalone savings are only roughly 12–14 us once per decode step and can be obscured by host variation or graph layout effects. The observed regressions are retained, not attributed causally without further timing evidence. These weak host-inclusive results triggered a separate bounded CUDA-event diagnostic; they are not a definitive backend rejection. Worker PIDs 425852, 425853 and 426579 all exited. Numerical tensors remain remotely in each original directory; compact JSON records preserve exact error checks. Compilation caches are excluded.

## Hook hashes

- `qvl-lmhead-model.py`: `5e0edfc69fa79bb4720f4cd27d1c927a823ebc316a04f32efe951c152851c80c`
- `qvl-lmhead-model-launch.py`: `74ebb777b905f24d313c075f6342d2c6e518e27f85bef4dbc2e86de667345697`
- `qvl-lmhead-model-launch8.py`: `c0ae30fe1475925a1c64a31b730fc9391df60b2474d9f532a5806c8f3146235d`
