from pathlib import Path

p = Path(
    "/root/qvl/venv-sgl/lib/python3.12/site-packages/flashinfer/gemm/kernels/dense_bf16_gemm_direct.py"
)
s = p.read_text()
s = (
    s.replace(
        "        stream: _cuda.CUstream,",
        "        gMeta: cute.Tensor,\n        stream: _cuda.CUstream,",
        1,
    )
    .replace(
        "self.kernel(gA, gB, gC, copy_a, copy_b)",
        "self.kernel(gA, gB, gC, gMeta, copy_a, copy_b)",
        1,
    )
    .replace(
        "        copy_a: cute.CopyAtom,",
        "        gMeta: cute.Tensor,\n        copy_a: cute.CopyAtom,",
        1,
    )
)
s = s.replace(
    "                cute.copy(copy_b, tB[None, k_tile], b_regs[ni, k_tile, None])",
    """                kk = (k_tile * block_size + tidx) * vector_width
                meta = gMeta[n_base + ni, kk // 128].to(cutlass.Uint32)
                if meta & 256:
                    packed = cute.recast_tensor(gB, cutlass.Uint16)
                    dst = cute.recast_tensor(b_regs, cutlass.Uint16)
                    base = (kk // 128) * 128
                    for vi in cutlass.range_constexpr(vector_width):
                        d = kk % 128 + vi
                        sm = packed[n_base + ni, base + d // 2].to(cutlass.Uint32)
                        sm = (sm >> ((d % 2) * 8)) & 255
                        ex = packed[n_base + ni, base + 64 + d // 4].to(cutlass.Uint32)
                        ex = (ex >> ((d % 4) * 4)) & 15
                        bits = ((sm & 128) << 8) | (sm & 127) | (((meta & 255) + ex) << 7)
                        dst[ni, k_tile, vi] = bits.to(cutlass.Uint16)
                else:
                    cute.copy(copy_b, tB[None, k_tile], b_regs[ni, k_tile, None])""",
)
Path("/root/qvl/experiments/weight-pack/packed_direct.py").write_text(s)
