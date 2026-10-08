# Lossless trace archives

The two large prefill traces are split into 1MiB parts to satisfy the repository file-size hook. Full originals remain available locally and remotely. Concatenate each trace’s parts in filename order; `trace_archives.json` records each part and reconstructed SHA256. Smaller traces are saved directly as gzip files.

```bash
cat analysis_inputs/sglang-b128/prefill-TP0.trace.json.gz.part* > analysis_inputs/sglang-b128/prefill-TP0.trace.json.gz
cat vllm/b128-v2/traces/prefill_rank0.1791491168898869904.pt.trace.json.gz.part* > vllm/b128-v2/traces/prefill_rank0.1791491168898869904.pt.trace.json.gz
```

Run these commands from this evidence directory. Verify the reconstructed hashes against the manifest before analysis. The shared input fixture has its own archive manifest under `shared/`, and SGLang metadata has a separate archive under `sglang_capture_metadata/`.
