#ifndef HYPERBLOCKER_CORE_DEVICE_FUNC_SIM_CUH_
#define HYPERBLOCKER_CORE_DEVICE_FUNC_SIM_CUH_

#include "core/gpu/kernel_data_structures/kernel_bitmap.cuh"

namespace sics {
namespace hyperblocker {
namespace core {
namespace gpu {

__device__ int min(int val_a, int val_b) {
  return val_a < val_b ? val_a : val_b;
}

__device__ int max(int val_a, int val_b) {
  return val_a > val_b ? val_a : val_b;
}

__host__ bool equalities(const char *term_l, const char *term_r,
                         size_t len_l = 0, size_t len_r = 0) {

  if (len_l == 0)
    for (int i = 0; term_l[i]; i++)
      len_l++;
  if (len_r == 0)
    for (int i = 0; term_r[i]; i++)
      len_r++;

  if (len_l != len_r)
    return false;

  for (size_t i = 0; i < len_l; i++) {
    if (*(term_l + i) != *(term_r + i))
      return false;
  }
  return true;
}

__device__ bool equalities_kernel(const char *term_l, const char *term_r) {
  int i = 0;
  while (term_l[i] && term_r[i]) {
    if (term_l[i] != term_r[i])
      return false;
    i++;
  }
  if (term_l[i])
    return false;
  if (term_r[i])
    return false;

  return true;
}

__device__ float lev_jaro_ratio(const char *term_l, const char *term_r) {
  size_t i, j, halflen, trans, match, to, len_l = 0, len_r = 0;

  while (term_l[len_l])
    len_l++;

  while (term_r[len_r])
    len_r++;

  float result = 0;
  float md;
  if (len_r == 0 || len_l == 0) {
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
  //uint16_t *idx = new uint16_t[256]();
   uint16_t idx[256] = {0};

  match = 0;

  for (i = 0; i < halflen; i++) {
    for (j = 0; j <= i + halflen; j++) {
      if (j >= 256)
        break;

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
      if (j > 256)
        break;
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
      if (j > 256)
        break;
      if (idx[j]) {
        i++;
        if (idx[j] != i)
          trans++;
      }
    }
    md = (float)match;
    result = (md / len_l + md / len_r + 1.0) / 3.0;
  }
  //delete[] idx;
  return result;
}

__host__ float host_lev_jaro_ratio(const char *term_l, const char *term_r,
                                   size_t len_l = 0, size_t len_r = 0) {
  size_t i, j, halflen, trans, match, to;
  if (len_l == 0)
    for (int i = 0; term_l[i]; i++)
      len_l++;
  if (len_r == 0)
    for (int i = 0; term_r[i]; i++)
      len_r++;

  float md;
  if (len_r == 0 || len_l == 0) {
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

__device__ double lev_jaro_winkler_ratio(const char *string1,
                                         const char *string2,
                                         double pfweight = 0.1) {
  double j = 0;
  size_t p, m;

  size_t len1 = 0, len2 = 0;
  for (int i = 0; string1[i]; i++)
    len1++;
  for (int i = 0; string2[i]; i++)
    len2++;

  j = lev_jaro_ratio(string1, string2);

  m = len1 < len2 ? len1 : len2;
  for (p = 0; p < m; p++) {
    if (string1[p] != string2[p])
      break;
  }
  j += (1.0 - j) * p * pfweight;
  return j > 1.0 ? 1.0 : j;
}

__host__ double host_lev_jaro_winkler_ratio(const char *string1,
                                            const char *string2,
                                            double pfweight = 0.1) {
  double j;
  size_t p, m;

  size_t len1 = 0, len2 = 0;
  for (int i = 0; string1[i]; i++)
    len1++;
  for (int i = 0; string2[i]; i++)
    len2++;

  j = host_lev_jaro_ratio(string1, string2, len1, len2);

  m = len1 < len2 ? len1 : len2;
  for (p = 0; p < m; p++) {
    if (string1[p] != string2[p])
      break;
  }
  j += (1.0 - j) * p * pfweight;
  return j > 1.0 ? 1.0 : j;
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

__host__ float host_jaccard_kernel(const char *term_l, const char *term_r) {

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

__device__ float LevenshteinDist(const char *word1, const char *word2) {
  int size1 = 0, size2 = 0;

  for (int i = 0; word1[i]; i++)
    size1++;
  for (int i = 0; word2[i]; i++)
    size2++;

  int *verif = (int *)malloc(sizeof(int) * (size1 + 1) * (size2 + 1));
  //  int verif[size1 + 1][size2 + 1]; // Verification matrix i.e. 2D array
  //  which
  // will store the calculated distance.

  // If one of the words has zero length, the distance is equal to the size of
  // the other word.
  if (size1 == 0)
    return size2;
  if (size2 == 0)
    return size1;

  // Sets the first row and the first column of the verification matrix with the
  // numerical order from 0 to the length of each word.
  for (int i = 0; i < size1; i++)
    *(verif + i * (size2 + 1) + 0) = i;
  for (int j = 0; j < size2; j++)
    *(verif + 0 * (size2 + 1) + j) = j;

  // // Verification step / matrix filling.
  // for (int i = 1; i <= size1; i++) {
  //   for (int j = 1; j <= size2; j++) {
  //     // Sets the modification cost.
  //     // 0 means no modification (i.e. equal letters) and 1 means that a
  //     // modification is needed (i.e. unequal letters).
  //     int cost = (word2[j - 1] == word1[i - 1]) ? 0 : 1;

  //     // Sets the current position of the matrix as the minimum value between
  //     a
  //     // (deletion), b (insertion) and c (substitution). a = the upper
  //     adjacent
  //     // value plus 1: verif[i - 1][j] + 1 b = the left adjacent value plus
  //     1:
  //     // verif[i][j - 1] + 1 c = the upper left adjacent value plus the
  //     // modification cost: verif[i - 1][j - 1] + cost
  //     // verif[i][j] =
  //     *(verif + i * (size2 + 1) + j) =
  //         min(min(*(verif + (i - 1) * size2 + j) + 1,
  //                 *(verif + i * size2 + j - 1) + 1),
  //             *(verif + (i - 1) * size2 + j - 1) + cost);
  //   }
  // }

  // float out = *(verif + size1 * size2) / max(size1, size2);
  // // The last position of the matrix will contain the Levenshtein distance.

  // free(verif);
  return 0;
}

} // namespace gpu
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_DEVICE_FUNC_SIM_CUH_
