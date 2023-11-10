#ifndef HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_H_

#include <cuda_runtime.h>

#include "core/common/types.h"
#include "core/components/scheduler/even_split_scheduler.h"
#include "core/data_structures/table.h"
#include "core/gpu/global_func.cuh"

namespace sics {
namespace hyperblocker {
namespace core {
namespace components {

using sics::hyperblocker::core::components::ExecutionPlanGenerator;
using sics::hyperblocker::core::components::SerializedExecutionPlan;
using sics::hyperblocker::core::data_structures::SerializableTable;
using sics::hyperblocker::core::data_structures::SerializedTable;
using sics::hyperblocker::core::gpu::blocking_kernel;
using sics::hyperblocker::core::gpu::equalities;
using sics::hyperblocker::core::gpu::Jaccard;
using sics::hyperblocker::core::gpu::jaro;

class HostProducer {
public:
  HostProducer(const sics::hyperblocker::core::components::DataMngr &data_mngr,
               const ExecutionPlanGenerator &epg)
      : data_mngr_(data_mngr), epg_(epg) {

    // Get device information.
    cudaError_t cudaStatus;
    std::cout << "Device properties" << std::endl;
    cudaDeviceProp devProp;
    cudaStatus = cudaGetDeviceCount(&n_device_);
    scheduler_ = new EvenSplitScheduler(n_device_);
  }

  void Run() {
    std::cout << "Host Producer running on " << n_device_ << " devices."
              << std::endl;

    data_mngr_.DataPartitioning(epg_.GetExecutionPlan(), 1);

    // TODO: Add procedure of submitting tasks.

    std::cout << data_mngr_.get_n_partitions() << std::endl;

    auto sep = epg_.GetExecutionPlan();
    for (size_t i = 0; i < data_mngr_.get_n_partitions(); i++) {
      auto partition = data_mngr_.GetPartition(i);
      auto device_id = scheduler_->GetBinID(i);
      std::cout << "table 1: " << partition.first.get_n_rows() << std::endl;
      std::cout << "table 2: " << partition.second.get_n_rows() << std::endl;
      std::cout << "device_id: " << device_id << std::endl;
      AsyncSubmit(partition.first, partition.second, sep, device_id);
    }
  }

  __host__ void AsyncSubmit(const SerializedTable &h_tb_l,
                            const SerializedTable &h_tb_r,
                            const SerializedExecutionPlan &h_sep,
                            int device_id) {
    dim3 dimBlock(32);
    dim3 dimGrid(256);

    // Generate serialized execution plan in Device.
    for (size_t i = 0; i < *(h_sep.length); i++) {
      std::cout << i << "/" << *h_sep.length << "pred_index "
                << h_sep.pred_index[i] << std::endl;
      std::cout << i << "/" << *h_sep.length << "pred_type "
                << h_sep.pred_type[i] << std::endl;
      std::cout << i << "/" << *h_sep.length << "pred_threshold "
                << h_sep.pred_threshold[i] << std::endl;
    }

    h_tb_l.Show();
    h_tb_r.Show();
    // Prepare serialized execution plan in Device.
    SerializedExecutionPlan d_sep;
    cudaMalloc(&(d_sep.n_rules), sizeof(int));
    cudaMalloc(&(d_sep.length), sizeof(int));
    cudaMalloc(&(d_sep.pred_index), sizeof(int) * (*h_sep.length));
    cudaMalloc(&d_sep.pred_type, sizeof(char) * (*h_sep.length));
    cudaMalloc(&d_sep.pred_threshold, sizeof(float) * (*h_sep.length));

    cudaMemcpy(d_sep.n_rules, h_sep.n_rules, sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sep.length, h_sep.length, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sep.pred_index, h_sep.pred_index, sizeof(int) * *h_sep.length,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sep.pred_type, h_sep.pred_type, sizeof(char) * *h_sep.length,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sep.pred_threshold, h_sep.pred_threshold,
               sizeof(float) * *h_sep.length, cudaMemcpyHostToDevice);

    // Prepare Table data in Device.
    char *d_tb_data_l, *d_tb_data_r;
    cudaMalloc((void **)&d_tb_data_l, h_tb_l.get_n_rows() *
                                          h_tb_l.get_aligned_tuple_size() *
                                          sizeof(char));
    cudaMalloc((void **)&d_tb_data_r, h_tb_r.get_n_rows() *
                                          h_tb_r.get_aligned_tuple_size() *
                                          sizeof(char));

    cudaMemcpy(d_tb_data_l, h_tb_l.get_data_base_ptr(),
               h_tb_l.get_n_rows() * h_tb_l.get_aligned_tuple_size() *
                   sizeof(char),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_tb_data_r, h_tb_r.get_data_base_ptr(),
               h_tb_r.get_n_rows() * h_tb_r.get_aligned_tuple_size() *
                   sizeof(char),
               cudaMemcpyHostToDevice);

    size_t *d_col_size_l, *d_col_offset_l, *d_col_size_r, *d_col_offset_r;
    cudaMalloc((void **)&d_col_size_l, h_tb_l.get_n_cols() * sizeof(size_t));
    cudaMalloc((void **)&d_col_offset_l, h_tb_l.get_n_cols() * sizeof(size_t));
    cudaMalloc((void **)&d_col_size_r, h_tb_r.get_n_cols() * sizeof(size_t));
    cudaMalloc((void **)&d_col_offset_r, h_tb_r.get_n_cols() * sizeof(size_t));

    // Copy the table data to the device.
    cudaMemcpy(d_tb_data_l, h_tb_l.get_data_base_ptr(),
               h_tb_l.get_n_rows() * h_tb_l.get_aligned_tuple_size() *
                   sizeof(char),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_tb_data_r, h_tb_r.get_data_base_ptr(),
               h_tb_r.get_n_rows() * h_tb_r.get_aligned_tuple_size() *
                   sizeof(char),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_col_size_l, h_tb_l.get_col_size_base_ptr(),
               h_tb_l.get_n_cols() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_size_r, h_tb_r.get_col_size_base_ptr(),
               h_tb_r.get_n_cols() * sizeof(size_t), cudaMemcpyHostToDevice);

    // Malloc space to store the candidates.
    int *h_candidates, *d_candidates, *d_result_offset, *h_result_offset;
    cudaMalloc((void **)&d_candidates, MAX_CANDIDATE_COUNT * 2 * sizeof(int));

    cudaMalloc((void **)&d_result_offset, sizeof(int));
    cudaHostAlloc(&h_result_offset, sizeof(int), cudaHostAllocPortable);

    cudaError_t error_h_candidates =
        cudaHostAlloc(&h_candidates, MAX_CANDIDATE_COUNT * 2 * sizeof(int),
                      cudaHostAllocPortable);

    size_t shared_memory_size = 1024;
    // MAX_CANDIDATE_COUNT * 2 * sizeof(size_t);

    char *d_test_char, *h_test_char;
    cudaMalloc((void **)&d_test_char, MAX_CANDIDATE_COUNT *
                                          (h_tb_l.get_col_size_base_ptr()[0] +
                                           h_tb_r.get_col_size_base_ptr()[0]) *
                                          sizeof(char));
    cudaHostAlloc(&h_test_char,
                  MAX_CANDIDATE_COUNT *
                      (h_tb_l.get_col_size_base_ptr()[0] +
                       h_tb_r.get_col_size_base_ptr()[0]) *
                      sizeof(char),
                  cudaHostAllocPortable);

    float *d_test_float, *h_test_float;
    cudaMalloc((void **)&d_test_float, 65536 * sizeof(float));
    cudaHostAlloc(&h_test_float, 65536 * sizeof(float), cudaHostAllocPortable);

    std::cout << "submit task" << std::endl;
    // Submit task
    blocking_kernel<<<dimBlock, dimGrid>>>(
        h_tb_l.get_n_rows(), h_tb_r.get_n_rows(),
        h_tb_l.get_aligned_tuple_size(), h_tb_r.get_aligned_tuple_size(),
        d_tb_data_l, d_tb_data_r, d_col_size_l, d_col_size_r, d_col_offset_l,
        d_col_offset_r, d_sep, d_candidates, d_result_offset, d_test_char,
        d_test_float);
    std::cout << "#submit task" << std::endl;

    // cudaError_t err = cudaStreamQuery(*p_stream);

    // if (err == cudaSuccess) {
    //  cudaMemcpyAsync(h_candidates, d_candidates,
    //                  MAX_CANDIDATE_COUNT * 2 * sizeof(size_t),
    //                  cudaMemcpyDeviceToHost, *p_stream);
    cudaMemcpy(h_result_offset, d_result_offset, sizeof(int),
               cudaMemcpyDefault);
    cudaMemcpy(h_candidates, d_candidates, (*h_result_offset) * 2 * sizeof(int),
               cudaMemcpyDefault);
    for (size_t i = 0; i < *h_result_offset; i++) {
      std::cout << i << ", " << h_candidates[i] << std::endl;
    }
    std::cout << "--------" << std::endl;

    cudaMemcpy(h_test_char, d_test_char,
               MAX_CANDIDATE_COUNT *
                   (h_tb_l.get_col_size_base_ptr()[0],
                    h_tb_r.get_col_size_base_ptr()[0]) *
                   sizeof(char),
               cudaMemcpyDefault);
    std::cout << h_test_char << std::endl;

    std::cout << "--------" << std::endl;
    cudaMemcpy(h_test_float, d_test_float, 65536 * sizeof(float),
               cudaMemcpyDefault);
    for (size_t i = 0; i < 10; i++) {
      std::cout << "float " << i << ", " << h_test_float[i] << std::endl;
    }

    cudaFreeHost(d_candidates);
    cudaFreeHost(d_tb_data_l);
    cudaFreeHost(d_tb_data_r);
    cudaFree(d_col_size_l);
    cudaFree(d_col_size_r);
    cudaFree(d_col_offset_l);
    cudaFree(d_col_offset_r);
    // cudaFree(d_sep->pred_index);
    // cudaFree(d_sep->pred_type);
    // cudaFree(d_sep->pred_threshold);
  }

  __host__ void AsyncSubmit(const SerializedTable &h_tb, int device_id) {
    dim3 dimBlock(32);
    dim3 dimGrid(1024);

    cudaStream_t *p_stream = new cudaStream_t;
    cudaStreamCreate(p_stream);

    // Malloc device memory for the table.
    char *d_tb_data;
    cudaMalloc((void **)&d_tb_data, h_tb.get_n_rows() *
                                        h_tb.get_aligned_tuple_size() *
                                        sizeof(char));
    size_t *d_col_size, *d_col_offset;
    cudaMalloc((void **)&d_col_size, h_tb.get_n_cols() * sizeof(size_t));
    cudaMalloc((void **)&d_col_offset, h_tb.get_n_cols() * sizeof(size_t));

    // Copy the table data to the device.
    // cudaMemcpyAsync(
    //    d_tb_data, h_tb.get_data(),
    //    h_tb.get_n_rows() * h_tb.get_aligned_tuple_size() * sizeof(char),
    //    cudaMemcpyHostToDevice, *p_stream);
    cudaMemcpy(d_tb_data, h_tb.get_data_base_ptr(),
               h_tb.get_n_rows() * h_tb.get_aligned_tuple_size() * sizeof(char),
               cudaMemcpyHostToDevice);

    // Malloc space to store the candidates.
    size_t *h_candidates, *d_candidates;
    cudaMalloc((void **)&d_candidates,
               MAX_CANDIDATE_COUNT * 2 * sizeof(size_t));

    cudaError_t error_h_candidates =
        cudaHostAlloc(&h_candidates, MAX_CANDIDATE_COUNT * 2 * sizeof(size_t),
                      cudaHostAllocPortable);

    size_t shared_memory_size = MAX_CANDIDATE_COUNT * 2 * sizeof(size_t);

    // Submit task
    // blocking_kernel<<<dimBlock, dimGrid, shared_memory_size>>>(
    //    h_tb.get_n_rows(), h_tb.get_n_rows(), d_tb_data, d_tb_data,
    //    d_col_size, d_col_size, d_col_offset, d_col_offset, d_candidates);

    cudaError_t err = cudaStreamQuery(*p_stream);

    if (err == cudaSuccess) {
      // cudaMemcpyAsync(h_candidates, d_candidates,
      //                 MAX_CANDIDATE_COUNT * 2 * sizeof(size_t),
      //                 cudaMemcpyDeviceToHost, *p_stream);
      cudaMemcpy(h_candidates, d_candidates,
                 MAX_CANDIDATE_COUNT * 2 * sizeof(size_t), cudaMemcpyDefault);
      for (size_t i = 0; i < 10; i++) {
        std::cout << h_candidates[i] << std::endl;
      }
    }

    cudaFree(d_candidates);
    cudaFree(d_tb_data);
    cudaFree(d_col_size);
    cudaFree(d_col_offset);
  }

private:
  int n_device_ = 0;
  Scheduler *scheduler_;

  sics::hyperblocker::core::components::DataMngr data_mngr_;
  ExecutionPlanGenerator epg_;
};

} // namespace components
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_CUH_
