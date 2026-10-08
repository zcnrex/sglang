# Scoped PDL consumer audit

The exact recorded B8 BF16 TRT decode consumer contains a programmatic dependency wait before its Q/K/V TMA data loads. This supports retaining the original early GEMM launch trigger while adding epilogue writes. It does not replace the separate changed-input dependent-attention correctness test.

The audited consumer is `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP32MultiCtasKvCgaVarSeqQ8Kv256StaticGroupedSwapsAbForGen`, for Q32/KV8/D128, HND page32. No GPU kernels were launched for this audit. Only source reading, CPU compilation, and disassembly were performed.

## Runtime and instruction evidence

`trtllm_mha_backend.py::_run_fixed_q_len_decode` calls the FlashInfer decode API without an `enable_pdl` override. Installed `flashinfer/decode.py` resolves the default through `device_support_pdl`, which returns true for CUDA major architecture at least 9; backend auto selects trtllm-gen for major 10. The launcher forwards `enable_pdl` to runner parameters. `fmhaKernels.cuh::buildLaunchConfig` sets `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` from that parameter.

CPU compilation of the two minimal PTX wrappers with CUDA 13.0.88 and `-arch=sm_103a` establishes the instruction mapping:

- `griddepcontrol.wait` becomes `ACQBULK` (opcode `0x782e`).
- `griddepcontrol.launch_dependents` becomes `PREEXIT` (opcode `0x782d`).

The exact consumer contains `ACQBULK` at 0x8e70 before its first `UTMALDG.4D` instructions at 0x8f30 and 0x8f40. Additional waits occur at 0x1ce0, 0x1d00, 0xc1f0, 0xc2a0, and 0xc2e0. The earlier `DEPBAR.LE` instructions are not the evidence for a PDL wait.

## Earlier loads and pointer layout

The installed `kernelParams.h` layout, interpreted at the cubin parameter base 0x380, identifies the earlier pointers as metadata. Seven 128-byte TMA descriptors occupy 0x380 bytes, followed by three int32 grid dimensions and alignment to the pointer fields:

| Constant offset | Struct offset | Field | Observed reads |
| --- | --- | --- | --- |
| 0x728 | 0x3a8 | ptrCumSeqLensQ | LDG at 0x09c0 and 0x09e0 |
| 0x778 | 0x3f8 | ptrPageIdxKv | LDGSTS at 0x1270 and 0x13c0 |
| 0x7c8 | 0x448 | ptrSeqLensKv | LDG at 0x0990 |

The disassembly loads the page-index base at 0x1070 and computes the address consumed at 0x1270. These pre-wait reads are therefore metadata, not the fused Q/K/V values. This layout identification is static source/disassembly analysis, not a runtime pointer trace.

## Reproduction and limits

`excerpts.txt` records exact cubin and installed-source SHA256 hashes, compact instruction excerpts, and the compiled instruction mapping. `extract.py.txt` reproduces that extraction. The mapping source is retained as `instruction_map.cu.txt`.

Commands on the pinned host:

```sh
/usr/local/cuda/bin/nvcc -arch=sm_103a -cubin /tmp/qvl-pdl-map.cu -o /tmp/qvl-pdl-audit/map.cubin
/usr/local/cuda/bin/cuobjdump --dump-sass /tmp/qvl-pdl-audit/map.cubin > /tmp/qvl-pdl-audit/map.sass
/usr/local/cuda/bin/cuobjdump --dump-sass <exact-cubin-path-in-excerpts.txt> > /tmp/qvl-pdl-audit/consumer.sass
python3 /tmp/qvl-pdl-extract.py
```

The [CUDA PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) distinguishes an early launch permission from completion and memory visibility. The consumer dependency wait supplies the latter. This audit does not claim the early trigger itself flushes epilogue writes.

This is not an exhaustive control-flow proof of every consumer branch or a guarantee for other TRT kernels. It does not verify graph dependency edges dynamically. The proposed fused producer must still pass changed-input dependent-attention graph replay checks with the actual consumer and launch configuration. No production files were modified.
