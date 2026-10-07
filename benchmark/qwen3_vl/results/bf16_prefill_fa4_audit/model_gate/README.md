# External FA4 CLC model gates

All successful scripts assert that all4,229 imported Python source files match candidate5f60fb8b67 before CUDA allocation. Model weights, Q/K/V and outputs remain BF16; native HND page32 KV cache and TRT decode remain unchanged. Prefill graphs are disabled. Executed external scripts are preserved as `.py.txt` to avoid formatter changes. No production integration is included.

## Pure prefill

`pure_model.py.txt` alternates baseline/candidate in one loaded model with common warmups, fresh request/cache state, and eight measured ABBA iterations. B1 has8200 input tokens; B2 has8200/8203. Both generate four output tokens. The hook replaces only the TRT context call, after original cache writes, using fresh unpaged K/V with actual token stride6144.

- B1 median prefill65.021→61.994ms (-4.66%).
- B2 median128.613→125.143ms (-2.70%).
- Each run recorded216 actual FA4 calls and zero fallback calls.
- All four paired greedy positions match. Logits are not bitwise: B1 maximum absolute difference0.15625, B2 maximum0.3125. NRMS reaches0.01314 in B2's final decode step.

The initial prototype returned FA4's `(output,lse)` tuple where TRT expected a tensor, failing before timing. Its failure logs are retained under `fa4-clc-model-b1/b2`. Successful corrected results are under `fa4-clc-model-v2-b1/b2`. Empty constructor records in those successful runs reflect persistent JIT-cache hits; actual selected-key proof is supplied by the mixed gate below.

## Genuine cached decode tails

`mixed_model.py.txt` rebuilds124 actual8192-token caches in groups of eight before each variant, then evaluates two fresh requests8200/8007 plus124 one-token cached tails:16,331 query rows. The benchmark uses an equivalent EXTEND batch; the production attention router recognizes its one-token suffix. Three further TRT decode steps follow. Setup and cache verification are outside measured iterations.

The context hook receives exactly16,207 rows for two requests with prefix lengths[0,0]; it leaves the124 decode rows to the existing TRT path. Four measured ABBA iterations give baseline153.972ms versus candidate150.334ms (-2.36%). This is a bounded gate with limited timing samples, not an end-to-end throughput claim.

-144 FA4 calls, zero fallback calls.
- The persistent cache proxy verifies actual selected compile-key field39 is `True` for CLC; selected-key SHA256 is `d193fda1cde8751bc3ab51cd2011057d9475c3414fe128ad21c9797560934adf`.
- Layer0 K/V writes at all16,331 destination positions match incoming BF16 K/V bitwise in four warmup checks, including cached tails.
- All126 requests have matching greedy tokens at all four compared positions.
- Logits differ: maximum0.90625 at the first decode step, NRMS0.006925; subsequent steps have maximum differences0.1875 and0.125. Accuracy evaluation remains required.

GPU5's forced-fallback case first caches64 tokens of the first context request. It records zero FA4 calls and144 cached-prefix fallbacks; all four compared logit steps are bitwise, and layer0 KV checks pass. This checks that cached-prefix requests retain the original attention path.

## Delegated serving hook

`serving_hook.py.txt` is the external hook handed to the validation agent. It retains the validated slice checks and adds conservative multimodal/feature/dtype fallbacks and output-buffer pointer verification. It changes no production file. Candidate-only `sitecustomize` can load it via `runpy.run_path`; `QVL_FA4_REPORT` specifies coverage output. Reports include calls, processed context rows, fallback reasons, actual CLC cache hits and first-context layout proof. The serving and accuracy gates are separate work; these model results do not establish accuracy equivalence.
