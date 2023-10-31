
#ifndef HYPERBLOCKER_CORE_DEVICE_FUNC_HOST_FUNC_CUH_
#define HYPERBLOCKER_CORE_DEVICE_FUNC_HOST_FUNC_CUH_

#include <cuda_runtime.h>

#include "core/common/types.cuh"
#include "core/data_structures/table.cuh"
#include "core/gpu/global_func.cuh"

namespace sics::hyper_blocker::core::gpu {
namespace host {

template <typename T>
void copy_to_device(T* device_ptr, T* host_ptr, size_t size);

template <typename T>
void copy_to_host(T* device_ptr, T* host_ptr, size_t size);

template <typename T>
void copy_to_device(T* device_ptr, T* host_ptr, size_t size, size_t offset);

__host__ void AsyncBlockingSubmit(
    const sics::hyper_blocker::core::data_structures::Table& h_tb) {
  //dim3 dimBlock(32);
  //dim3 dimGrid(1024);

  //cudaStream_t* p_stream = new cudaStream_t;
  //cudaStreamCreate(p_stream);

  //// Malloc device memory for the table.
  //char* d_tb_data;
  //cudaMalloc((void**)&d_tb_data,
  //           h_tb.get_n_rows() * h_tb.get_aligned_tuple_size() * sizeof(char));
  //size_t *d_col_size, *d_col_offset;
  //cudaMalloc((void**)&d_col_size, h_tb.get_n_cols() * sizeof(size_t));
  //cudaMalloc((void**)&d_col_offset, h_tb.get_n_cols() * sizeof(size_t));

  //// Copy the table data to the device.
  //// cudaMemcpyAsync(
  ////    d_tb_data, h_tb.get_data(),
  ////    h_tb.get_n_rows() * h_tb.get_aligned_tuple_size() * sizeof(char),
  ////    cudaMemcpyHostToDevice, *p_stream);
  //cudaMemcpy(d_tb_data, h_tb.get_data(),
  //           h_tb.get_n_rows() * h_tb.get_aligned_tuple_size() * sizeof(char),
  //           cudaMemcpyHostToDevice);

  //// Malloc space to store the candidates.
  //size_t *h_candidates, *d_candidates;
  //cudaMalloc((void**)&d_candidates, MAX_CANDIDATE_COUNT * 2 * sizeof(size_t));

  //cudaError_t error_h_candidates =
  //    cudaHostAlloc(&h_candidates, MAX_CANDIDATE_COUNT * 2 * sizeof(size_t),
  //                  cudaHostAllocPortable);

  //size_t shared_memory_size = MAX_CANDIDATE_COUNT * 2 * sizeof(size_t);

  //// Submit task
  //sics::hyper_blocker::core::gpu::global::
  //    Blocking<<<dimBlock, dimGrid, shared_memory_size>>>(
  //        d_tb_data, d_tb_data, d_col_size, d_col_offset, d_candidates);

  //cudaError_t err = cudaStreamQuery(*p_stream);

  //std::cout << "XX" << std::endl;
  //if (err == cudaSuccess) {
  //  // cudaMemcpyAsync(h_candidates, d_candidates,
  //  //                 MAX_CANDIDATE_COUNT * 2 * sizeof(size_t),
  //  //                 cudaMemcpyDeviceToHost, *p_stream);
  //  cudaMemcpy(h_candidates, d_candidates,
  //             MAX_CANDIDATE_COUNT * 2 * sizeof(size_t), cudaMemcpyDefault);
  //  std::cout << "XX" << std::endl;
  //  for (size_t i = 0; i < 10; i++) {
  //    std::cout << h_candidates[i] << std::endl;
  //  }
  //}



  int *ret;
  cudaMalloc(&ret, 1000 * sizeof(int));
  sics::hyper_blocker::core::gpu::global::AplusB<<< 1, 1000 >>>(ret, 10, 100);
  int *host_ret = (int *)malloc(1000 * sizeof(int));
  cudaMemcpy(host_ret, ret, 1000 * sizeof(int), cudaMemcpyDefault);
  for(int i = 0; i < 1000; i++)
    printf("%d: A+B = %d\n", i, host_ret[i]);
  free(host_ret);
  cudaFree(ret);



}

__host__ void AsyncDirtyBlockingReceive() {}

}  // namespace host
}  // namespace sics::hyper_blocker::core::gpu
#endif  // HYPERBLOCKER_CORE_DEVICE_FUNC_HOST_FUNC_CUH_
