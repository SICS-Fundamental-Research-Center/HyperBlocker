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

__device__ bool test1() { return false; }

__global__ void
blocking_kernel(size_t n_rows_l, size_t n_rows_r, size_t aligned_tuple_size_l,
                size_t aligned_tuple_size_r, const char *d_tb_data_l,
                const char *d_tb_data_r, const size_t *d_col_size_l,
                const size_t *d_col_size_r, const size_t *d_col_offset_l,
                const size_t *d_col_offset_r, SerializedExecutionPlan d_sep,
                int *d_candidate, int *result_offset, char *d_candidate_char,
                float *d_test_float) {
  // Compute tid.
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int step = blockDim.x;

  // extern __shared__ size_t shared_output_buffer[];
  // extern __shared__ bool to_skip[];

  size_t row_index_l = tid;
  while (row_index_l < n_rows_l) {

    for (int row_index_r = 0; row_index_r < n_rows_r; row_index_r++) {

      bool is_match = false;

      for (int i = 0; i < *(d_sep.length); i++) {
        bool to_skip = false;

        int pred_index = d_sep.pred_index[i];
        switch (d_sep.pred_type[i]) {
        case EQUALITIES:
          if (!equalities_kernel(
                  d_tb_data_l + row_index_l * aligned_tuple_size_l +
                      d_col_offset_l[pred_index],
                  d_tb_data_r + row_index_r * aligned_tuple_size_r +
                      d_col_offset_r[pred_index],
                  d_col_size_l[i], d_col_size_r[i])) {
            to_skip = true;
          }
          break;
        case SIM:
          if (lev_jaro_ratio(d_tb_data_l + row_index_l * aligned_tuple_size_l +
                                 d_col_offset_l[i],
                             d_tb_data_r + row_index_r * aligned_tuple_size_r +
                                 d_col_offset_r[i]) < d_sep.pred_threshold[i]) {
            to_skip = true;
          }
          break;
        }

        // Move to the next check point. the next round to computation start
        // at check point +1;
        if (to_skip) {
          while (d_sep.pred_index[i] != CHECK_POINT && i < *(d_sep.length))
            i++;
        } else {
          // two tuple is a match, break the loop.
          if (i < *(d_sep.length) && d_sep.pred_index[i + 1] == CHECK_POINT) {
            is_match = true;
            break;
          }
        }
      }

      // TODO: is a match then output the result.
      if (is_match) {
        // if (row_index_l == row_index_r)
        //   continue;
        int local_offset = atomicAdd(result_offset, 1);
        if (local_offset > MAX_CANDIDATE_COUNT)
          continue;
        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset,
               d_tb_data_l + aligned_tuple_size_l * row_index_l,
               sizeof(char) * d_col_size_l[0]);
        memcpy(d_candidate_char + 2 * MAX_EID_COL_SIZE * local_offset +
                   MAX_EID_COL_SIZE,
               d_tb_data_r + aligned_tuple_size_r * row_index_r,
               sizeof(char) * d_col_size_r[0]);
      }
    }
    row_index_l += step;
  }
}

__global__ void AplusB(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

} // namespace gpu
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_GPU_GLOBAL_FUNC_CUH_
