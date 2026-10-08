import sys

sys.path.insert(
    0,
    "/root/qvl/venv-sgl/lib/python3.12/site-packages/flashinfer/data/cutlass/examples/python/CuTeDSL/blackwell",
)
import cutlass
import cutlass.utils as utils
import dense_gemm_persistent as d
import native_epi_map

utils.gemm.sm100.epilogue = native_epi_map.epilogue
print("starting", flush=True)
d.run(
    (128, 256, 256, 1),
    cutlass.BFloat16,
    cutlass.BFloat16,
    cutlass.Float32,
    "k",
    "k",
    "n",
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    use_2cta_instrs=False,
    use_tma_store=False,
    iterations=1,
    skip_ref_check=False,
)
