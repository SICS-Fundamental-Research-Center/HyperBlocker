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

__host__ float jaro(const char *term_l, const char *term_r) {
  std::cout << term_l << std::endl;
  std::cout << term_r << std::endl;
  size_t i, j, halflen, trans, match, to;
  int len_l = 0, len_r = 0;
  for (int i = 0; term_l[i]; i++)
    len_l++;
  for (int i = 0; term_r[i]; i++)
    len_r++;
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

__device__ float lev_jaro_ratio(const char *term_l, const char *term_r) {
  size_t i, j, halflen, trans, match, to;
  size_t len_l = 0, len_r = 0;
  for (int i = 0; term_l[i]; i++)
    len_l++;
  for (int i = 0; term_r[i]; i++)
    len_r++;

  float md;
  if (len_r == 0 || len_l == 0)
  {
    if (len_l == 0 && len_r == 0)
      return 1.0;
    return 0.0;
  }


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

  char c_diff = 'a' - 'A';

  bool bm_l[256];
  bool bm_r[256];
  for (int i = 0; term_l[i]; i++) {
    int val = term_l[i];
    if (term_l[i] - 'Z' <= 0)
      val += c_diff;
    bm_l[val] = true;
  }
  for (int i = 0; term_r[i]; i++) {
    int val = term_r[i];
    if (term_r[i] - 'Z' <= 0)
      val += c_diff;
    bm_r[val] = true;
  }
  int count_l = 0, count_r = 0;
  for (int i = 0; i < 256; i++) {
    count_l += bm_l[i];
    count_r += bm_r[i];
  }

  return (float)count_l / (float)count_r;
}

} // namespace gpu
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_DEVICE_FUNC_SIM_CUH_
