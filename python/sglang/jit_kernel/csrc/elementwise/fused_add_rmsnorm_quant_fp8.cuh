// Fused (residual-add + RMSNorm + static per-tensor FP8 quantization) kernel.
//
// This is a copy of fused_add_rmsnorm.cuh whose writeback, instead of storing
// the bf16/fp16 normalized value back to `input`, quantizes it to FP8-E4M3
// (qval = clip(norm * (1/scale))) and writes it to a separate `quant_out`
// buffer. The residual writeback (input+residual -> residual) is unchanged.
//
// Purpose: fold the activation FP8 quant that a downstream W8A8-FP8 linear
// (e.g. NemotronH mamba in_proj) would otherwise do in a separate
// static_quant_fp8 launch into the norm kernel, removing that launch and the
// bf16 norm-output HBM round-trip. Gated by SGLANG_OPT_NEMOTRON_FUSE_NORM_QUANT.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cooperative_groups/reduce.h>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <type_traits>

namespace {

template <typename T, int VEC_SIZE_IN_BYTE>
struct QuantVecTypeTrait;

template <>
struct QuantVecTypeTrait<bf16_t, 16> {
  using packed_t = packed_t<bf16_t>;
  using vec_t = device::AlignedVector<packed_t, 4>;
};

template <>
struct QuantVecTypeTrait<fp16_t, 16> {
  using packed_t = packed_t<fp16_t>;
  using vec_t = device::AlignedVector<packed_t, 4>;
};

template <>
struct QuantVecTypeTrait<bf16_t, 32> {
  using packed_t = packed_t<bf16_t>;
  using vec_t = device::AlignedVector<packed_t, 8>;
};

template <>
struct QuantVecTypeTrait<fp16_t, 32> {
  using packed_t = packed_t<fp16_t>;
  using vec_t = device::AlignedVector<packed_t, 8>;
};

template <bool kCastXBeforeOutMul, typename packed_t>
SGL_DEVICE packed_t q_rms(float2 valf, packed_t& weight, float rsqrt_square_sum) {
  float2 weightf = device::cast<fp32x2_t, packed_t>(weight);
  if constexpr (kCastXBeforeOutMul) {
    auto rounded = device::cast<packed_t, fp32x2_t>(make_float2(valf.x * rsqrt_square_sum, valf.y * rsqrt_square_sum));
    valf = device::cast<fp32x2_t, packed_t>(rounded);
    return device::cast<packed_t, fp32x2_t>(make_float2(valf.x * weightf.x, valf.y * weightf.y));
  }
  return device::cast<packed_t, fp32x2_t>(
      make_float2(valf.x * weightf.x * rsqrt_square_sum, valf.y * weightf.y * rsqrt_square_sum));
}

SGL_DEVICE float fp8_e4m3_clip(float val) {
  namespace math = device::math;
  return math::max(math::min(val, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX);
}

template <bool kCastXBeforeOutMul, typename T, int VEC_SIZE_IN_BYTE>
__global__ void fused_add_rmsnorm_quant_fp8_reg_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    fp8_e4m3_t* __restrict__ quant_out,
    const float* __restrict__ scale,
    int vec_hidden_size,
    float eps) {
  constexpr int inner_loop = VEC_SIZE_IN_BYTE == 16 ? 4 : 8;
  // Elements (scalars) each thread owns across the vector load.
  constexpr int elems_per_thread = inner_loop * 2;

  __shared__ float shared_memory[32];  // Used for CTA reduce

  using vec_t = typename QuantVecTypeTrait<T, VEC_SIZE_IN_BYTE>::vec_t;
  using packed_t = typename QuantVecTypeTrait<T, VEC_SIZE_IN_BYTE>::packed_t;
  using fp8_vec_t = device::AlignedVector<fp8_e4m3_t, elems_per_thread>;
  vec_t v;         // Save input
  vec_t v_res;     // Save residual
  vec_t v_weight;  // Save weight
  vec_t v_out;     // Save normalized output (pre-quant)
  float2 inp_res_cache[inner_loop];

  const float scale_val = 1.0f / (*scale);

  auto token_id = blockIdx.x;
  float2 acc_square = make_float2(0.0f, 0.0f);

  if (threadIdx.x < vec_hidden_size) {
    vec_t* p = reinterpret_cast<vec_t*>(input) + token_id * vec_hidden_size;
    vec_t* p_res = reinterpret_cast<vec_t*>(residual) + token_id * vec_hidden_size;
    const vec_t* p_weight = reinterpret_cast<const vec_t*>(weight);

    v = p[threadIdx.x];
    v_res = p_res[threadIdx.x];
    v_weight = p_weight[threadIdx.x];

    for (int i = 0; i < inner_loop; i++) {
      float2 val = device::cast<fp32x2_t, packed_t>(v[i]);
      float2 res = device::cast<fp32x2_t, packed_t>(v_res[i]);
      float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
      acc_square.x += inp_res.x * inp_res.x;
      acc_square.y += inp_res.y * inp_res.y;
      v[i] = device::cast<packed_t, fp32x2_t>(inp_res);
      if constexpr (kCastXBeforeOutMul) {
        inp_res_cache[i] = inp_res;
      }
    }

    // Store inp+res to residual (unchanged from the non-quant kernel).
    p_res[threadIdx.x] = v;
  }

  // CTA Reduce
  auto cg_warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  float warp_sum = cooperative_groups::reduce(cg_warp, acc_square.x + acc_square.y, cooperative_groups::plus<float>());

  float* buffer = shared_memory;
  if (threadIdx.x % 32 == 0) {
    buffer[threadIdx.x / 32] = warp_sum;
  }

  __syncthreads();
  if (threadIdx.x < 32) {
    float cta_sum = cooperative_groups::reduce(
        cg_warp, (threadIdx.x < blockDim.x / 32) ? buffer[threadIdx.x] : 0.0f, cooperative_groups::plus<float>());
    buffer[threadIdx.x] =
        rsqrtf(eps + cta_sum * (1.0f / static_cast<float>(vec_hidden_size * (VEC_SIZE_IN_BYTE / sizeof(T)))));
  }
  __syncthreads();

  // Compute RMSNorm, then quantize to FP8 (qval = clip(norm * 1/scale)).
  if (threadIdx.x < vec_hidden_size) {
    float rsqrt_square_sum = buffer[threadIdx.x / 32];
    fp8_vec_t q_out;
    for (int i = 0; i < inner_loop; i++) {
      float2 valf;
      if constexpr (kCastXBeforeOutMul) {
        valf = inp_res_cache[i];
      } else {
        valf = device::cast<fp32x2_t, packed_t>(v[i]);
      }
      v_out[i] = q_rms<kCastXBeforeOutMul>(valf, v_weight[i], rsqrt_square_sum);
      float2 normed = device::cast<fp32x2_t, packed_t>(v_out[i]);
      q_out[2 * i] = static_cast<fp8_e4m3_t>(fp8_e4m3_clip(normed.x * scale_val));
      q_out[2 * i + 1] = static_cast<fp8_e4m3_t>(fp8_e4m3_clip(normed.y * scale_val));
    }
    // Write only the FP8 quantized output; the bf16 norm output is intentionally
    // not written back to `input` (the downstream FP8 linear consumes quant_out).
    fp8_vec_t* p_quant = reinterpret_cast<fp8_vec_t*>(quant_out) + token_id * vec_hidden_size;
    p_quant[threadIdx.x] = q_out;
  }
}

template <bool kCastXBeforeOutMul, typename DType>
struct FusedAddRMSNormQuantFp8Kernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView quant_out,
      const tvm::ffi::TensorView scale,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})  // input
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({D})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({N, D})  // residual
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual);
    TensorMatcher({N, D})  // quant_out (FP8-E4M3)
        .with_strides({D, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(quant_out);
    TensorMatcher({1})  // scale (per-tensor activation scale)
        .with_dtype<float>()
        .with_device(device)
        .verify(scale);

    int hidden_size = static_cast<int>(D.unwrap());
    if (hidden_size <= (device::kMaxVecBytes == 32 ? 12288 : 8192)) {
      int elements_in_vec = device::kMaxVecBytes / sizeof(DType);
      int vec_hidden_size = hidden_size / elements_in_vec;
      uint threads = (vec_hidden_size + 31) / 32 * 32;

      host::RuntimeCheck(
          hidden_size % elements_in_vec == 0,
          "hidden_size",
          hidden_size,
          " can not align to elements_in_vec ",
          elements_in_vec);

      auto kernel = fused_add_rmsnorm_quant_fp8_reg_kernel<kCastXBeforeOutMul, DType, device::kMaxVecBytes>;
      LaunchKernel(static_cast<uint>(N.unwrap()), threads, device.unwrap())
          .enable_pdl(false)(
              kernel,
              reinterpret_cast<DType*>(input.data_ptr()),
              reinterpret_cast<DType*>(residual.data_ptr()),
              reinterpret_cast<DType*>(weight.data_ptr()),
              reinterpret_cast<fp8_e4m3_t*>(quant_out.data_ptr()),
              reinterpret_cast<const float*>(scale.data_ptr()),
              vec_hidden_size,
              eps);
    } else {
      host::RuntimeCheck(false, "Large hidden_sizes are not supported for now.");
    }
  }
};

}  // namespace
