# Current-source CLC kernel screen

`bench.py.txt` preserves the exact executed standalone harness bytes. Both runs verify all4,229 Python files in the imported candidate against commit5f60fb8b67. GPU4 and5 compare TRT nativeHND page32 with FA4 default and CLC, using BF16 Q/K/V with token stride6144. Inputs are synthetic, B1[8200] and B2[8200,8203]. Counterbalanced warm and256MiB-flushed timings are reported separately. Constructor records confirm actual CLC scheduling mode4 versus static mode2; both use512 threads,128×128 tiles,q_stage2 and TMA KV.

CLC and default FA4 outputs are bitwise equal. Against TRT maximum absolute difference is0.00390625, NRMS approximately0.0022, within asserted atol/rtol0.02. B2 warm medians on GPU4 are TRT899.46/default837.64/CLC758.12us; GPU5 are888.94/822.33/751.97us. Cold medians are895.99/850.45/762.93 and897.10/826.36/772.15us respectively. There is substantial within-run drift, but CLC wins each recorded paired B2 comparison. This supports a model gate, not an end-to-end claim.

`initial-invalid-model.py.txt` is a superseded, formatter-modified prototype. It incorrectly returned FA4's tuple where TRT expected a tensor and failed before model timing. It is not an executed-success reproduction script or source for performance claims. Corrected pure-prefill and mixed-model validation will be archived separately. File hashes are in hashes.json.
