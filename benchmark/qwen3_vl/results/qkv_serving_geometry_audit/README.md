# QKV accuracy: serving geometry audit

Read-only inspection of the retained matched GSM8K and fixed-input artifacts
finds a concrete execution-geometry mismatch. It does not identify the numerical
effect of that mismatch or establish scheduling causality.

The fixed-input diagnostic performs one EXTEND with 6,972 rows, eight selected
prompts and zero cached prefixes, followed by 128 fixed-batch decode steps.
It bypasses ordinary request completion/removal and supplies the candidate with
the control token sequence. Its bitwise full-logit result remains valid for
those identical inputs.

The full-serving control records 1,294 MIXED and two EXTEND forwards; the
candidate records 1,290 MIXED and two EXTEND forwards. Every recorded MIXED
forward has cached prefixes. The arms have 124 and 128 distinct prefill row
counts, respectively. Observer ordinal 2 has the same 6,074 total rows but
different extension ordering:

- Control: `[861, 868, 870, 865, 843, 905, 861, 1]`
- Candidate: `[870, 843, 868, 865, 905, 861, 861, 1]`

Early server logs also show a new request with 90 new and 768 cached tokens.
The observer lacks request IDs, so these records cannot reliably be assigned
to individual disagreement questions. Dataset/report alignment does not imply
identical runtime batch packing.

Both matched-serving arms genuinely validate the four targeted common tactics
and restore ordinary tuning before evaluation. Each records six M128 forwards,
all MIXED with batch size eight. Equal call counts do not imply equal inputs.
Prompt HTML hashes align across the full-serving arms. The fixed diagnostic
verifies token IDs between its own arms; the full-serving artifacts do not
retain per-request token IDs for an independent cross-protocol comparison.

Seven selected disagreement responses first differ within 109–275 characters;
row 425 first differs at character 979. These are character offsets, not token
positions, and do not establish the first divergent decode step.

The next useful accuracy evidence would associate request IDs with mixed-batch
and cached-prefix geometry, retain exact token IDs, and capture first-divergent
tokens/logits. No additional benchmark was run for this audit. It does not
erase the matched full-serving score of 1,216 versus 1,213 out of 1,314.

Sources: sibling `qkv_integrated_matched_accuracy/` and
`qkv_integrated_fixed_gsm/`, especially their `lmhead-proof-*.jsonl`, server
logs, prompt/row reports, and fixed diagnostic `model.py.txt`.
