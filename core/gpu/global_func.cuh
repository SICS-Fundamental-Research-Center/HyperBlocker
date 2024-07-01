#ifndef HYPERBLOCKER_CORE_GPU_GLOBAL_FUNC_CUH_
#define HYPERBLOCKER_CORE_GPU_GLOBAL_FUNC_CUH_

#include "core/components/execution_plan_generator.h"
#include "core/gpu/device_func.cuh"
#include "core/gpu/kernel_data_structures/kernel_bitmap.cuh"

namespace sics {
namespace hyperblocker {
namespace core {
namespace gpu {

using sics::hyperblocker::core::components::SerializedExecutionPlan;
using sics::hyperblocker::core::gpu::KernelBitmap;

__device__ void DFS(unsigned int row_index_l, unsigned int row_index_r,
                    size_t aligned_tuple_size_l, size_t aligned_tuple_size_r,
                    const char *d_tb_data_l, const char *d_tb_data_r,
                    const size_t *d_col_size_l, const size_t *d_col_size_r,
                    const size_t *d_col_offset_l, const size_t *d_col_offset_r,
                    SerializedExecutionPlan d_sep, int *result_offset,
                    char *d_candidate_char, bool *visited, int i,
                    bool *is_match) {

  if (i < d_sep.length) {

    int pred_index_l = d_sep.pred_index[i];
    int pred_index_r = d_sep.pred_index[i + 1];

    switch (d_sep.pred_type[i]) {
    case EQUALITIES:
      if (!equalities_kernel(d_tb_data_l + row_index_l * aligned_tuple_size_l +
                                 d_col_offset_l[pred_index_l],
                             d_tb_data_r + row_index_r * aligned_tuple_size_r +
                                 d_col_offset_r[pred_index_r])) {
        return;
      }
      break;
    case SIM:
      auto sim = lev_jaro_winkler_ratio(
          d_tb_data_l + row_index_l * aligned_tuple_size_l +
              d_col_offset_l[pred_index_l],
          d_tb_data_r + row_index_r * aligned_tuple_size_r +
              d_col_offset_r[pred_index_r]);

      if (sim < d_sep.pred_threshold[i]) {
        return;
      }
      break;
    }

    if (i + 2 < d_sep.length && d_sep.pred_index[i + 2] == CHECK_POINT) {
      int local_offset = atomicAdd(result_offset, 1);
      if (local_offset > MAX_CANDIDATE_COUNT - 1)
        return;

      memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset,
             d_tb_data_l + aligned_tuple_size_l * row_index_l,
             sizeof(char) * d_col_size_l[0]);
      memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset +
                 MAX_EID_COL_SIZE,
             d_tb_data_r + aligned_tuple_size_r * row_index_r,
             sizeof(char) * d_col_size_r[0]);
    }
    if (i + 2 < d_sep.length) {
      if (d_sep.pred_index[i + 2] != CHECK_POINT) {
        DFS(row_index_l, row_index_r, aligned_tuple_size_l,
            aligned_tuple_size_r, d_tb_data_l, d_tb_data_r, d_col_size_l,
            d_col_size_r, d_col_offset_l, d_col_offset_r, d_sep, result_offset,
            d_candidate_char, visited, i + 2, is_match);
      }
    }
  }
}

__global__ void
blocking_kernelDFS(size_t n_rows_l, size_t n_rows_r,
                   size_t aligned_tuple_size_l, size_t aligned_tuple_size_r,
                   const char *d_tb_data_l, const char *d_tb_data_r,
                   const size_t *d_col_size_l, const size_t *d_col_size_r,
                   const size_t *d_col_offset_l, const size_t *d_col_offset_r,
                   SerializedExecutionPlan d_sep, int *result_offset,
                   char *d_candidate_char) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;
  if (tid >= n_rows_l)
    return;

  for (unsigned int row_index_l = tid; row_index_l < n_rows_l;
       row_index_l += step) {
    for (unsigned int row_index_r = 0; row_index_r < n_rows_r; row_index_r++) {
      int pos = 0;
      bool is_match = false;
      bool *visited = new bool[32]();
      DFS(row_index_l, row_index_r, aligned_tuple_size_l, aligned_tuple_size_r,
          d_tb_data_l, d_tb_data_r, d_col_size_l, d_col_size_r, d_col_offset_l,
          d_col_offset_r, d_sep, result_offset, d_candidate_char, visited, pos,
          &is_match);
      delete[] visited;
    }
  }
}

__global__ void
blocking_kernel(size_t n_rows_l, size_t n_rows_r, size_t aligned_tuple_size_l,
                size_t aligned_tuple_size_r, const char *d_tb_data_l,
                const char *d_tb_data_r, const size_t *d_col_size_l,
                const size_t *d_col_size_r, const size_t *d_col_offset_l,
                const size_t *d_col_offset_r, SerializedExecutionPlan d_sep,
                int *result_offset, char *d_candidate_char) {
  // Compute tid.
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  for (unsigned int row_index_l = tid; row_index_l < n_rows_l;
       row_index_l += step) {
    for (unsigned int row_index_r = 0; row_index_r < n_rows_r; row_index_r++) {
      bool is_match = false;

      for (int i = 0; i < (d_sep.length); i += 2) {
        bool to_skip = false;

        int pred_index_l = d_sep.pred_index[i];
        int pred_index_r = d_sep.pred_index[i + 1];

        switch (d_sep.pred_type[i]) {
        case EQUALITIES:

          if (!equalities_kernel(
                  d_tb_data_l + row_index_l * aligned_tuple_size_l +
                      d_col_offset_l[pred_index_l],
                  d_tb_data_r + row_index_r * aligned_tuple_size_r +
                      d_col_offset_r[pred_index_r])) {
            to_skip = true;
          }
          break;
        case SIM:

          auto sim = lev_jaro_winkler_ratio(
              d_tb_data_l + row_index_l * aligned_tuple_size_l +
                  d_col_offset_l[pred_index_l],
              d_tb_data_r + row_index_r * aligned_tuple_size_r +
                  d_col_offset_r[pred_index_r]);
          if (sim < d_sep.pred_threshold[i]) {
            to_skip = true;
          }
          break;
        }

        // Move to the next check point. Ps: The next round to computation
        // start at check point + 1;
        if (to_skip) {
          while (i + 2 < d_sep.length) {
            if (d_sep.pred_index[i + 2] == CHECK_POINT) {
              i++;
              break;
            }
            i += 2;
          }
        } else {
          // two tuple is a match, break the loop.
          if (i < (d_sep.length) && d_sep.pred_index[i + 2] == CHECK_POINT) {
            is_match = true;
            break;
          }
        }
      }

      // is a match then output the result.
      if (is_match) {
        int local_offset = atomicAdd(result_offset, 1);
        if (local_offset > MAX_CANDIDATE_COUNT - 1)
          return;

        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset,
               d_tb_data_l + aligned_tuple_size_l * row_index_l,
               sizeof(char) * d_col_size_l[0]);
        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset +
                   MAX_EID_COL_SIZE,
               d_tb_data_r + aligned_tuple_size_r * row_index_r,
               sizeof(char) * d_col_size_r[0]);
      }
    }
  }
}

__global__ void blocking_no_psw_kernel(
    size_t n_rows_l, size_t n_rows_r, size_t aligned_tuple_size_l,
    size_t aligned_tuple_size_r, const char *d_tb_data_l,
    const char *d_tb_data_r, const size_t *d_col_size_l,
    const size_t *d_col_size_r, const size_t *d_col_offset_l,
    const size_t *d_col_offset_r, SerializedExecutionPlan d_sep,
    int *result_offset, char *d_candidate_char) {

  // Compute tid.
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  unsigned row_index_lower_bound = ceil(n_rows_l / (float)step) * tid;
  unsigned row_index_upper_bound = ceil(n_rows_l / (float)step) * (tid + 1);

  row_index_lower_bound = min(row_index_lower_bound, n_rows_l);
  row_index_upper_bound = min(row_index_upper_bound, n_rows_l);
  for (unsigned int row_index_l = row_index_lower_bound;
       row_index_l < row_index_upper_bound; row_index_l++) {
    for (unsigned int row_index_r = 0; row_index_r < n_rows_r; row_index_r++) {
      bool is_match = false;

      for (int i = 0; i < (d_sep.length); i += 2) {
        bool to_skip = false;

        int pred_index_l = d_sep.pred_index[i];
        int pred_index_r = d_sep.pred_index[i + 1];

        switch (d_sep.pred_type[i]) {
        case EQUALITIES:
          if (!equalities_kernel(
                  d_tb_data_l + row_index_l * aligned_tuple_size_l +
                      d_col_offset_l[pred_index_l],
                  d_tb_data_r + row_index_r * aligned_tuple_size_r +
                      d_col_offset_r[pred_index_r])) {
            to_skip = true;
          }
          break;
        case SIM:
          auto sim = lev_jaro_winkler_ratio(
              d_tb_data_l + row_index_l * aligned_tuple_size_l +
                  d_col_offset_l[pred_index_l],
              d_tb_data_r + row_index_r * aligned_tuple_size_r +
                  d_col_offset_r[pred_index_r]);
          if (sim < d_sep.pred_threshold[i]) {
            to_skip = true;
          }
          break;
        }

        // Move to the next check point. Ps: The next round to computation
        // start at check point + 1;
        if (to_skip) {
          while (i + 2 < d_sep.length) {
            if (d_sep.pred_index[i + 2] == CHECK_POINT) {
              i++;
              break;
            }
            i += 2;
          }
        } else {
          // two tuple is a match, break the loop.
          if (i < (d_sep.length) && d_sep.pred_index[i + 2] == CHECK_POINT) {
            is_match = true;
            break;
          }
        }
      }

      // is a match then output the result.
      if (is_match) {
        int local_offset = atomicAdd(result_offset, 1);
        if (local_offset > MAX_CANDIDATE_COUNT - 1)
          return;

        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset,
               d_tb_data_l + aligned_tuple_size_l * row_index_l,
               sizeof(char) * d_col_size_l[0]);
        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset +
                   MAX_EID_COL_SIZE,
               d_tb_data_r + aligned_tuple_size_r * row_index_r,
               sizeof(char) * d_col_size_r[0]);
      }
    }
  }
}

__global__ void blocking_task_stealing_kernel(
    size_t n_rows_l, size_t n_rows_r, size_t aligned_tuple_size_l,
    size_t aligned_tuple_size_r, const char *d_tb_data_l,
    const char *d_tb_data_r, const size_t *d_col_size_l,
    const size_t *d_col_size_r, const size_t *d_col_offset_l,
    const size_t *d_col_offset_r, SerializedExecutionPlan d_sep,
    int *result_offset, char *d_candidate_char, bool *kbm,
    unsigned int *inner_scope_lower_bound,
    unsigned int *inner_scope_upper_bound, bool *d_finished) {

  // Compute tid.
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x * gridDim.x;

  for (unsigned int row_index_l = tid; row_index_l < n_rows_l;
       row_index_l += step) {
    if (kbm[row_index_l])
      continue;
    kbm[row_index_l] = true;
    for (unsigned int row_index_r = inner_scope_lower_bound[row_index_l];
         row_index_r < inner_scope_upper_bound[row_index_l]; row_index_r++) {
      inner_scope_lower_bound[row_index_l]++;
      bool is_match = false;

      for (int i = 0; i < (d_sep.length); i += 2) {
        bool to_skip = false;

        int pred_index_l = d_sep.pred_index[i];
        int pred_index_r = d_sep.pred_index[i + 1];

        switch (d_sep.pred_type[i]) {
        case EQUALITIES:
          if (!equalities_kernel(
                  d_tb_data_l + row_index_l * aligned_tuple_size_l +
                      d_col_offset_l[pred_index_l],
                  d_tb_data_r + row_index_r * aligned_tuple_size_r +
                      d_col_offset_r[pred_index_r])) {
            to_skip = true;
          }
          break;
        case SIM:
          auto sim = lev_jaro_winkler_ratio(
              d_tb_data_l + row_index_l * aligned_tuple_size_l +
                  d_col_offset_l[pred_index_l],
              d_tb_data_r + row_index_r * aligned_tuple_size_r +
                  d_col_offset_r[pred_index_r]);
          if (sim < d_sep.pred_threshold[i]) {
            to_skip = true;
          }
          break;
        }

        // Move to the next check point. Ps: The next round to computation
        // start at check point + 1;
        if (to_skip) {
          while (i + 2 < d_sep.length) {
            if (d_sep.pred_index[i + 2] == CHECK_POINT) {
              i++;
              break;
            }
            i += 2;
          }
        } else {
          // two tuple is a match, break the loop.
          if (i < (d_sep.length) && d_sep.pred_index[i + 2] == CHECK_POINT) {
            is_match = true;
            break;
          }
        }
      }

      // is a match then output the result.
      if (is_match) {
        int local_offset = atomicAdd(result_offset, 1);
        if (local_offset > MAX_CANDIDATE_COUNT - 1)
          return;

        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset,
               d_tb_data_l + aligned_tuple_size_l * row_index_l,
               sizeof(char) * d_col_size_l[0]);
        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset +
                   MAX_EID_COL_SIZE,
               d_tb_data_r + aligned_tuple_size_r * row_index_r,
               sizeof(char) * d_col_size_r[0]);
      }
    }
    d_finished[row_index_l] = true;
  }

  // Task Stealing
  for (int row_index_l = n_rows_l - tid; row_index_l > n_rows_l / 2;
       row_index_l -= step) {
    if (kbm[row_index_l] || row_index_l > n_rows_l / 2)
      break;
    kbm[row_index_l] = true;
    for (int row_index_r = inner_scope_lower_bound[row_index_l];
         row_index_r < inner_scope_upper_bound[row_index_l]; row_index_r++) {
      bool is_match = false;
      inner_scope_lower_bound[row_index_l]++;

      for (int i = 0; i < (d_sep.length); i += 2) {
        bool to_skip = false;

        int pred_index_l = d_sep.pred_index[i];
        int pred_index_r = d_sep.pred_index[i + 1];

        switch (d_sep.pred_type[i]) {
        case EQUALITIES:
          if (!equalities_kernel(
                  d_tb_data_l + row_index_l * aligned_tuple_size_l +
                      d_col_offset_l[pred_index_l],
                  d_tb_data_r + row_index_r * aligned_tuple_size_r +
                      d_col_offset_r[pred_index_r])) {
            to_skip = true;
          }
          break;
        case SIM:
          auto sim = lev_jaro_winkler_ratio(
              d_tb_data_l + row_index_l * aligned_tuple_size_l +
                  d_col_offset_l[pred_index_l],
              d_tb_data_r + row_index_r * aligned_tuple_size_r +
                  d_col_offset_r[pred_index_r]);
          if (sim < d_sep.pred_threshold[i]) {
            to_skip = true;
          }
          break;
        }

        // Move to the next check point. Ps: The next round to computation
        // start at check point + 1;
        if (to_skip) {
          while (i + 2 < d_sep.length) {
            if (d_sep.pred_index[i + 2] == CHECK_POINT) {
              i++;
              break;
            }
            i += 2;
          }
        } else {
          // two tuple is a match, break the loop.
          if (i < (d_sep.length) && d_sep.pred_index[i + 2] == CHECK_POINT) {
            is_match = true;
            break;
          }
        }
      }

      // is a match then output the result.
      if (is_match) {
        int local_offset = atomicAdd(result_offset, 1);
        if (local_offset > MAX_CANDIDATE_COUNT - 1)
          return;

        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset,
               d_tb_data_l + aligned_tuple_size_l * row_index_l,
               sizeof(char) * d_col_size_l[0]);
        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset +
                   MAX_EID_COL_SIZE,
               d_tb_data_r + aligned_tuple_size_r * row_index_r,
               sizeof(char) * d_col_size_r[0]);
      }
    }
    d_finished[row_index_l] = true;
  }

  // Inner task Stealing
  unsigned int row_index_l = tid % n_rows_l;
  if (!d_finished[row_index_l]) {
    d_finished[row_index_l] = true;
    auto local_upper_bound = inner_scope_upper_bound[row_index_l];
    auto local_lower_bound =
        min(n_rows_r, inner_scope_lower_bound[row_index_l] +
                          (inner_scope_upper_bound[row_index_l] -
                           inner_scope_lower_bound[row_index_l]) /
                              2);
    inner_scope_upper_bound[row_index_l] = local_lower_bound;

    for (unsigned int row_index_r = local_lower_bound;
         row_index_r < min(local_upper_bound, n_rows_r); row_index_r++) {
      bool is_match = false;

      for (int i = 0; i < (d_sep.length); i += 2) {
        bool to_skip = false;

        int pred_index_l = d_sep.pred_index[i];
        int pred_index_r = d_sep.pred_index[i + 1];

        switch (d_sep.pred_type[i]) {
        case EQUALITIES:
          if (!equalities_kernel(
                  d_tb_data_l + row_index_l * aligned_tuple_size_l +
                      d_col_offset_l[pred_index_l],
                  d_tb_data_r + row_index_r * aligned_tuple_size_r +
                      d_col_offset_r[pred_index_r])) {
            to_skip = true;
          }
          break;
        case SIM:
          auto sim = lev_jaro_winkler_ratio(
              d_tb_data_l + row_index_l * aligned_tuple_size_l +
                  d_col_offset_l[pred_index_l],
              d_tb_data_r + row_index_r * aligned_tuple_size_r +
                  d_col_offset_r[pred_index_r]);
          if (sim < d_sep.pred_threshold[i]) {
            to_skip = true;
          }
          break;
        }

        // Move to the next check point. Ps: The next round to computation
        // start at check point + 1;
        if (to_skip) {
          while (i + 2 < d_sep.length) {
            if (d_sep.pred_index[i + 2] == CHECK_POINT) {
              i++;
              break;
            }
            i += 2;
          }
        } else {
          // two tuple is a match, break the loop.
          if (i < (d_sep.length) && d_sep.pred_index[i + 2] == CHECK_POINT) {
            is_match = true;
            break;
          }
        }
      }

      // is a match then output the result.
      if (is_match) {
        int local_offset = atomicAdd(result_offset, 1);
        if (local_offset > MAX_CANDIDATE_COUNT - 1)
          return;

        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset,
               d_tb_data_l + aligned_tuple_size_l * row_index_l,
               sizeof(char) * d_col_size_l[0]);
        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset +
                   MAX_EID_COL_SIZE,
               d_tb_data_r + aligned_tuple_size_r * row_index_r,
               sizeof(char) * d_col_size_r[0]);
      }
    }
  }
}

} // namespace gpu
} // namespace core
} // namespace hyperblocker
} // namespace sics

#endif // HYPERBLOCKER_CORE_GPU_GLOBAL_FUNC_CUH_