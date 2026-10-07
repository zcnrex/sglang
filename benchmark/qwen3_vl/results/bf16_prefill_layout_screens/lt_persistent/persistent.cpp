#include <array>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <stdexcept>
#include <torch/extension.h>

namespace sglang {
void check(cublasStatus_t s) {
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cuBLASLt status ", int(s));
}
struct Plan {
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, d{};
  Plan(int64_t m, int64_t n, int64_t k) {
    check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    auto ta = CUBLAS_OP_T;
    auto tb = CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                         sizeof(ta)));
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                         sizeof(tb)));
    check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k));
    check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k));
    check(cublasLtMatrixLayoutCreate(&d, CUDA_R_16BF, n, m, n));
  }
  ~Plan() {
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(d);
    cublasLtMatmulDescDestroy(op);
  }
};
using Key = std::array<int64_t, 5>;
thread_local std::map<Key, std::unique_ptr<Plan>> plans;
void run(torch::Tensor x, torch::Tensor w, torch::Tensor y,
         torch::Tensor workspace, torch::Tensor algorithms, int64_t index,
         int64_t handle, int64_t stream, bool cached) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && y.is_cuda() && workspace.is_cuda(),
              "CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                  w.scalar_type() == torch::kBFloat16 &&
                  y.scalar_type() == torch::kBFloat16,
              "BF16 only");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y.dim() == 2 &&
                  x.is_contiguous() && w.is_contiguous() && y.is_contiguous(),
              "contiguous row-major matrices required");
  TORCH_CHECK(x.device() == w.device() && x.device() == y.device() &&
                  x.device() == workspace.device(),
              "same device required");
  TORCH_CHECK(workspace.scalar_type() == torch::kUInt8 &&
                  workspace.is_contiguous(),
              "byte workspace required");
  TORCH_CHECK(!algorithms.is_cuda() &&
                  algorithms.scalar_type() == torch::kUInt8 &&
                  algorithms.is_contiguous(),
              "CPU algorithm bytes required");
  auto m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(w.size(1) == k && y.size(0) == m && y.size(1) == n,
              "shape mismatch");
  TORCH_CHECK(index >= 0 && (index + 1) * 64 <= algorithms.numel(),
              "invalid algorithm index");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0,
              "aligned tensors required");
  c10::cuda::CUDAGuard guard(x.device());
  std::unique_ptr<Plan> temporary;
  Plan *plan;
  if (cached) {
    Key key = {x.get_device(), m, n, k, int64_t(x.scalar_type())};
    auto &ptr = plans[key];
    if (!ptr)
      ptr = std::make_unique<Plan>(m, n, k);
    plan = ptr.get();
  } else {
    temporary = std::make_unique<Plan>(m, n, k);
    plan = temporary.get();
  }
  static_assert(sizeof(cublasLtMatmulAlgo_t) == 64);
  cublasLtMatmulAlgo_t algo;
  std::memcpy(&algo, algorithms.data_ptr<uint8_t>() + index * 64, 64);
  const float alpha = 1.f, beta = 0.f;
  check(cublasLtMatmul(reinterpret_cast<cublasLtHandle_t>(handle), plan->op,
                       &alpha, w.data_ptr(), plan->a, x.data_ptr(), plan->b,
                       &beta, nullptr, plan->d, y.data_ptr(), plan->d, &algo,
                       workspace.data_ptr(), workspace.numel(),
                       reinterpret_cast<cudaStream_t>(stream)));
}
} // namespace sglang
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &sglang::run); }
