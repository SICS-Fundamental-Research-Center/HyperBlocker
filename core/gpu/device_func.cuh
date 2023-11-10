#ifndef HYPERBLOCKER_CORE_DEVICE_FUNC_SIM_CUH_
#define HYPERBLOCKER_CORE_DEVICE_FUNC_SIM_CUH_

#include "core/gpu/kernel_data_structures/kernel_bitmap.cuh"

namespace sics {
namespace hyperblocker {
namespace core {
namespace gpu {

using sics::hyperblocker::core::gpu::KernelBitmap;

__forceinline__ __device__ bool test(size_t i, size_t j) { return true; }
__device__ void test() { return; }

__host__ bool equalities(const char *term_l, const char *term_r, size_t len_l,
                         size_t len_r) {
  std::cout << term_l << std::endl;
  std::cout << term_r << std::endl;
  size_t len = len_l < len_r ? len_l : len_r;
  for (size_t i = 0; i < len - 1; i++) {
    if (*(term_l + i) != *(term_r + i))
      return false;
  }
  return true;
}

__host__ float Jaccard(const char *term_l, const char *term_r, size_t len_l,
                       size_t len_r) {

  size_t len = 256;
  // len_l < len_r ? len_r : len_l;
  uint8_t bm_l[len] = {0};
  uint8_t bm_r[len] = {0};

  int i;
  for (i = 0; term_l[i]; i++)
    bm_l[term_l[i]] = 1;
  for (i = 0; term_r[i]; i++)
    bm_r[term_r[i]] = 1;

  int in = 0;
  int un = 0;

  for (i = 0; i < len; i++) {
    in += bm_l[i] && bm_r[i];
    un += bm_l[i] || bm_r[i];
  }

  return (float)in / (float)un;
}

__host__ float jaro(const char *term_l, const char *term_r, size_t len_l,
                    size_t len_r) {
  std::cout << term_l << std::endl;
  std::cout << term_r << std::endl;
  size_t i, j, halflen, trans, match, to;
  float md;
  if (len_l == len_r && len_l == 0)
    return 1;
  if (len_l == len_r && len_l > 0)
    return 0;

  if (len_l > len_r) {
    const char *b;
    b = term_l;
    term_l = term_r;
    term_r = b;
    i = len_l;
    len_l = len_r;
    len_r = i;
  }

  halflen = (len_l + 1) / 2;
  size_t *idx = (size_t *)malloc(sizeof(size_t) * len_l);
  memset(idx, 0, sizeof(size_t) * len_l);
  float result = 0;
  if (!idx)
    result = -1.0;

  match = 0;

  for (i = 0; i < halflen; i++) {
    for (j = 0; j <= i + halflen; j++) {
      if (term_l[j] == term_r[i] && !idx[j]) {
        match++;
        idx[j] = match;
        break;
      }
    }
  }
  to = len_l + halflen < len_r ? len_l + halflen : len_r;

  for (i = halflen; i < to; i++) {
    for (j = i - halflen; j < len_l; j++) {
      if (term_l[j] == term_r[i] && !idx[j]) {
        match++;
        idx[j] = match;
        break;
      }
    }
  }

  if (!match) {
    result = 0.0;
  } else {
    i = 0;
    trans = 0;
    for (j = 0; j < len_l; j++) {
      if (idx[j]) {
        i++;
        if (idx[j] != i)
          trans++;
      }
    }
    md = (float)match + 1;
    result = (md / len_l + md / len_r + 1.0) / 3.0;
  }
  free(idx);
  return result;
}

__device__ bool equalities_kernel(const char *term_l, const char *term_r,
                                  size_t len_l, size_t len_r) {
  size_t len = len_l < len_r ? len_l : len_r;
  for (size_t i = 0; i < len - 1; i++) {
    if (*(term_l + i) != *(term_r + i))
      return false;
  }
  return true;
}

__device__ float lev_jaro_ratio(size_t len_l, const char *term_l, size_t len_r,
                                const char *term_r) {
  size_t i, j, halflen, trans, match, to;
  float md;
  if (len_l == len_r && len_l == 0)
    return 1;
  if (len_l == len_r && len_l > 0)
    return 0;

  if (len_l > len_r) {
    const char *b;
    b = term_l;
    term_l = term_r;
    term_r = b;
    i = len_l;
    len_l = len_r;
    len_r = i;
  }

  halflen = (len_l + 1) / 2;
  size_t *idx = (size_t *)malloc(sizeof(size_t) * len_l);
  memset(idx, 0, sizeof(size_t) * len_l);
  float result = 0;
  if (!idx)
    result = -1.0;

  match = 0;

  for (i = 0; i < halflen; i++) {
    for (j = 0; j <= i + halflen; j++) {
      if (term_l[j] == term_r[i] && !idx[j]) {
        match++;
        idx[j] = match;
        break;
      }
    }
  }
  to = len_l + halflen < len_r ? len_l + halflen : len_r;

  for (i = halflen; i < to; i++) {
    for (j = i - halflen; j < len_l; j++) {
      if (term_l[j] == term_r[i] && !idx[j]) {
        match++;
        idx[j] = match;
        break;
      }
    }
  }

  if (!match) {
    result = 0.0;
  } else {
    i = 0;
    trans = 0;
    for (j = 0; j < len_l; j++) {
      if (idx[j]) {
        i++;
        if (idx[j] != i)
          trans++;
      }
    }
    md = (float)match + 1;
    result = (md / len_l + md / len_r + 1.0) / 3.0;
  }
  free(idx);
  return result;
}

__device__ float jaccard_kernel(const char *term_l, const char *term_r) {

  // len_l < len_r ? len_r : len_l;
  uint8_t bm_l[256] = {0};
  uint8_t bm_r[256] = {0};

  KernelBitmap kbm_l(256);
  KernelBitmap kbm_r(256);

  int i;
  for (i = 0; term_l[i]; i++)
    kbm_l.SetBit(term_l[i]);
  for (i = 0; term_r[i]; i++)
    kbm_r.SetBit(term_r[i]);

  return (float)kbm_l.Count() / (float)kbm_r.Count();
}

} // namespace gpu
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_DEVICE_FUNC_SIM_CUH_
