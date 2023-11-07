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
using sics::hyperblocker::core::data_structures::SerializableTable;
using sics::hyperblocker::core::data_structures::SerializedTable;

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

    data_mngr_.DataPartitioning(epg_.GetExecutionPlan(), n_device_);

    // TODO: Add procedure of submitting tasks.

    std::cout << data_mngr_.get_n_partitions() << std::endl;

    for (size_t i = 0; i < data_mngr_.get_n_partitions(); i++) {
      auto partition = data_mngr_.GetPartition(i);
      auto device_id = scheduler_->GetBinID(i);
      std::cout << "table 1: " << partition.first.get_n_rows() << std::endl;
      std::cout << "table 2: " << partition.second.get_n_rows() << std::endl;
      std::cout << "device_id: " << device_id << std::endl;
      AsyncSubmit(partition.first, partition.second, device_id);
    }
  }

  __host__ void AsyncSubmit(const SerializedTable &h_tb_l,
                            const SerializedTable &h_tb_r, int device_id) {
    dim3 dimBlock(32);
    dim3 dimGrid(1024);
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
    Blocking<<<dimBlock, dimGrid, shared_memory_size>>>(
        d_tb_data, d_tb_data, d_col_size, d_col_offset, d_candidates);

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

    cudaFreeHost(d_candidates);
    cudaFreeHost(d_tb_data);
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
