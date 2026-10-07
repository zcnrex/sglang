#include <cuda_runtime.h>
#include <stdint.h>
template <bool Packed, bool Check>
__global__ void producer(const unsigned char *src, const uint16_t *meta,
                         uint32_t *out, int rows) {
  __shared__ uint4 tile[32 * 16];
  int t = threadIdx.x;
#pragma unroll
  for (int z = 0; z < 2; z++) {
    int grain = t + z * 256;
    int r = grain / 16;
    int c = grain % 16;
    int row = blockIdx.x * 32 + r;
    uint4 *dst = tile + r * 16 + (c ^ (r & 7));
    if constexpr (!Packed) {
      unsigned a = (unsigned)__cvta_generic_to_shared(dst);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a),
                   "l"(src + (size_t)row * 256 + c * 16));
    } else {
      unsigned h = meta[row];
      if (!(h & 256)) {
        unsigned a = (unsigned)__cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a),
                     "l"(src + (size_t)row * 256 + c * 16));
      } else {
        const unsigned char *p = src + (size_t)row * 256;
        uint2 sm = *(const uint2 *)(p + c * 8);
        unsigned ex = *(const unsigned *)(p + 128 + c * 4);
        uint4 v;
        unsigned *q = (unsigned *)&v;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          unsigned s = (j < 2 ? sm.x : sm.y) >> (16 * (j & 1));
          unsigned spread;
          asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(spread) : "r"(s));
          unsigned nib = (ex >> (j * 8)) & 255;
          unsigned e = (h & 255) * 65537 + (nib & 15) + ((nib >> 4) << 16);
          q[j] =
              (spread & 0x007f007f) | ((spread & 0x00800080) << 8) | (e << 7);
        }
        *dst = v;
      }
    }
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  unsigned sum = 0;
#pragma unroll
  for (int z = 0; z < 2; z++) {
    int grain = t + z * 256;
    int r = grain / 16;
    int c = grain % 16;
    uint4 v = tile[r * 16 + (c ^ (r & 7))];
    if constexpr (Check)
      ((uint4 *)out)[(size_t)blockIdx.x * 512 + grain] = v;
    else
      sum ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if constexpr (!Check)
    out[blockIdx.x * 256 + t] = sum;
}
extern "C" void run(const void *p, const void *m, void *o, int rows, int packed,
                    int check, void *stream) {
  if (packed) {
    if (check)
      producer<true, true><<<rows / 32, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned char *)p, (const uint16_t *)m, (uint32_t *)o, rows);
    else
      producer<true, false><<<rows / 32, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned char *)p, (const uint16_t *)m, (uint32_t *)o, rows);
  } else {
    if (check)
      producer<false, true><<<rows / 32, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned char *)p, (const uint16_t *)m, (uint32_t *)o, rows);
    else
      producer<false, false><<<rows / 32, 256, 0, (cudaStream_t)stream>>>(
          (const unsigned char *)p, (const uint16_t *)m, (uint32_t *)o, rows);
  }
}
