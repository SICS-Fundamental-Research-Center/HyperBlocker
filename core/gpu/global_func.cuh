#ifndef HYPERBLOCKER_CORE_GPU_GLOBAL_FUNC_CUH_
#define HYPERBLOCKER_CORE_GPU_GLOBAL_FUNC_CUH_

__global__ void Blocking(char* d_tb_data_l, char* d_tb_data_r, size_t* d_col_size,
                       size_t* d_col_offset, size_t* d_candidate) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t num_blocks = gridDim.x * blockDim.x;
  d_candidate[0] = 9;
  d_candidate[1] = 10;
  d_candidate[3] = 23;
  d_candidate[4] = 23;
  d_candidate[5] = 23;
  d_candidate[6] = 23;
  // for (size_t i = tid; i < *d_tb_size; i += num_blocks) {
  //   d_tb_data[i] = 0;
  // }
  //*d_tb_offset = tid;
}

__global__ void AplusB(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

#endif  // HYPERBLOCKER_CORE_GPU_GLOBAL_FUNC_CUH_
