#ifndef HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_H_

#include <cuda_runtime.h>
#include <unistd.h>

#include <condition_variable>
#include <mutex>

#include "core/common/types.h"
#include "core/components/scheduler/even_split_scheduler.h"
#include "core/data_structures/match.h"
#include "core/data_structures/table.h"
#include "core/gpu/global_func.cuh"
#include "core/gpu/kernel_data_structures/kernel_table.cuh"

namespace sics {
namespace hyperblocker {
namespace core {
namespace components {

using sics::hyperblocker::core::components::DataMngr;
using sics::hyperblocker::core::components::ExecutionPlanGenerator;
using sics::hyperblocker::core::components::SerializedExecutionPlan;
using sics::hyperblocker::core::data_structures::Match;
using sics::hyperblocker::core::data_structures::SerializableTable;
using sics::hyperblocker::core::data_structures::SerializedTable;
using sics::hyperblocker::core::gpu::blocking_kernel;
using sics::hyperblocker::core::gpu::equalities;
using sics::hyperblocker::core::gpu::jaro;

class HostProducer {
public:
  HostProducer(DataMngr *data_mngr, ExecutionPlanGenerator *epg,
               std::unordered_map<int, cudaStream_t *> *p_streams,
               Match *p_match, std::unique_lock<std::mutex> *p_hr_start_lck,
               std::condition_variable *p_hr_start_cv)
      : data_mngr_(data_mngr), epg_(epg), p_streams_(p_streams),
        p_match_(p_match), p_hr_start_lck_(p_hr_start_lck),
        p_hr_start_cv_(p_hr_start_cv) {

    std::cout << "Host Producer initializing." << std::endl;

    p_streams_ = p_streams;
    // Get device information.
    cudaError_t cudaStatus;
    cudaDeviceProp devProp;
    cudaStatus = cudaGetDeviceCount(&n_device_);
    scheduler_ = new EvenSplitScheduler(n_device_);
  }

  void Run() {
    std::cout << "Host Producer running on " << n_device_ << " devices."
              << std::endl;

    data_mngr_->DataPartitioning(epg_->GetExecutionPlan(), n_device_);
    auto sep = epg_->GetExecutionPlan();

    // TODO: Add procedure of submitting tasks.
    for (size_t i = 0; i < data_mngr_->get_n_partitions(); i++) {
      auto partition = data_mngr_->GetPartition(i);
      auto bin_id = scheduler_->GetBinID(i);
      std::cout << "Table 1 size: " << partition.first.get_n_rows()
                << std::endl;
      std::cout << "Table 2 size: " << partition.second.get_n_rows()
                << std::endl;
      std::cout << "Device_id: " << bin_id << std::endl;

      cudaStream_t *p_stream = new cudaStream_t;
      cudaStreamCreate(p_stream);
      p_streams_->insert(std::make_pair(i, p_stream));

      AsyncSubmit(partition.first, partition.second, sep, i, bin_id, p_stream);
    }
    p_hr_start_cv_->notify_all();
  }

  __host__ void AsyncSubmit(const SerializedTable &h_tb_l,
                            const SerializedTable &h_tb_r,
                            const SerializedExecutionPlan &h_sep, int ball_id,
                            int bin_id, cudaStream_t *p_stream) {
    dim3 dimBlock(32);
    dim3 dimGrid(128);

    // h_tb_l.Show();
    // h_tb_r.Show();
    //  Generate serialized execution plan in Device.

    SerializedExecutionPlan d_sep;
    cudaMalloc(&(d_sep.n_rules), sizeof(int));
    cudaMalloc(&(d_sep.length), sizeof(int));
    cudaMalloc(&(d_sep.pred_index), sizeof(int) * (*h_sep.length));
    cudaMalloc(&d_sep.pred_type, sizeof(char) * (*h_sep.length));
    cudaMalloc(&d_sep.pred_threshold, sizeof(float) * (*h_sep.length));

    cudaMemcpyAsync(d_sep.n_rules, h_sep.n_rules, sizeof(int),
                    cudaMemcpyHostToDevice, *p_stream);
    cudaMemcpyAsync(d_sep.length, h_sep.length, sizeof(int),
                    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_sep.pred_index, h_sep.pred_index,
                    sizeof(int) * *h_sep.length, cudaMemcpyHostToDevice,
                    *p_stream);
    cudaMemcpyAsync(d_sep.pred_type, h_sep.pred_type,
                    sizeof(char) * *h_sep.length, cudaMemcpyHostToDevice,
                    *p_stream);
    cudaMemcpyAsync(d_sep.pred_threshold, h_sep.pred_threshold,
                    sizeof(float) * *h_sep.length, cudaMemcpyHostToDevice,
                    *p_stream);

    // Prepare Table data in Device.
    char *d_tb_data_l, *d_tb_data_r;
    cudaMalloc((void **)&d_tb_data_l, h_tb_l.get_n_rows() *
                                          h_tb_l.get_aligned_tuple_size() *
                                          sizeof(char));
    cudaMalloc((void **)&d_tb_data_r, h_tb_r.get_n_rows() *
                                          h_tb_r.get_aligned_tuple_size() *
                                          sizeof(char));

    cudaMemcpyAsync(d_tb_data_l, h_tb_l.get_data_base_ptr(),
                    h_tb_l.get_n_rows() * h_tb_l.get_aligned_tuple_size() *
                        sizeof(char),
                    cudaMemcpyHostToDevice, *p_stream);
    cudaMemcpyAsync(d_tb_data_r, h_tb_r.get_data_base_ptr(),
                    h_tb_r.get_n_rows() * h_tb_r.get_aligned_tuple_size() *
                        sizeof(char),
                    cudaMemcpyHostToDevice, *p_stream);

    size_t *d_col_size_l, *d_col_offset_l, *d_col_size_r, *d_col_offset_r;
    cudaMalloc((void **)&d_col_size_l, h_tb_l.get_n_cols() * sizeof(size_t));
    cudaMalloc((void **)&d_col_offset_l, h_tb_l.get_n_cols() * sizeof(size_t));
    cudaMalloc((void **)&d_col_size_r, h_tb_r.get_n_cols() * sizeof(size_t));
    cudaMalloc((void **)&d_col_offset_r, h_tb_r.get_n_cols() * sizeof(size_t));

    // Copy the table data to the device.
    cudaMemcpyAsync(d_tb_data_l, h_tb_l.get_data_base_ptr(),
                    h_tb_l.get_n_rows() * h_tb_l.get_aligned_tuple_size() *
                        sizeof(char),
                    cudaMemcpyHostToDevice, *p_stream);
    cudaMemcpyAsync(d_tb_data_r, h_tb_r.get_data_base_ptr(),
                    h_tb_r.get_n_rows() * h_tb_r.get_aligned_tuple_size() *
                        sizeof(char),
                    cudaMemcpyHostToDevice, *p_stream);

    cudaMemcpyAsync(d_col_size_l, h_tb_l.get_col_size_base_ptr(),
                    h_tb_l.get_n_cols() * sizeof(size_t),
                    cudaMemcpyHostToDevice, *p_stream);
    cudaMemcpyAsync(d_col_size_r, h_tb_r.get_col_size_base_ptr(),
                    h_tb_r.get_n_cols() * sizeof(size_t),
                    cudaMemcpyHostToDevice, *p_stream);
    cudaMemcpyAsync(d_col_offset_l, h_tb_l.get_col_offset_base_ptr(),
                    h_tb_l.get_n_cols() * sizeof(size_t),
                    cudaMemcpyHostToDevice, *p_stream);
    cudaMemcpyAsync(d_col_offset_r, h_tb_r.get_col_offset_base_ptr(),
                    h_tb_r.get_n_cols() * sizeof(size_t),
                    cudaMemcpyHostToDevice, *p_stream);

    // Malloc space to store the candidates.
    int *h_candidates, *d_candidates, *d_result_offset, *h_result_offset;
    cudaMalloc((void **)&d_candidates, MAX_CANDIDATE_COUNT * 2 * sizeof(int));

    cudaMalloc((void **)&d_result_offset, sizeof(int));
    cudaHostAlloc(&h_result_offset, sizeof(int), cudaHostAllocPortable);

    cudaError_t error_h_candidates =
        cudaHostAlloc(&h_candidates, MAX_CANDIDATE_COUNT * 2 * sizeof(int),
                      cudaHostAllocPortable);

    char *d_candidates_char, *h_candidates_char;
    cudaMalloc((void **)&d_candidates_char,
               MAX_CANDIDATE_COUNT * MAX_EID_COL_SIZE * 2 * sizeof(char));
    cudaHostAlloc(&h_candidates_char,
                  MAX_CANDIDATE_COUNT * MAX_EID_COL_SIZE * 2 * sizeof(char),
                  cudaHostAllocPortable);

    float *d_test_float, *h_test_float;
    cudaMalloc((void **)&d_test_float, 65536 * sizeof(float));
    cudaHostAlloc(&h_test_float, 65536 * sizeof(float), cudaHostAllocPortable);

    // Submit task
    blocking_kernel<<<dimBlock, dimGrid, 48 * 1024, *p_stream>>>(
        h_tb_l.get_n_rows(), h_tb_r.get_n_rows(),
        h_tb_l.get_aligned_tuple_size(), h_tb_r.get_aligned_tuple_size(),
        d_tb_data_l, d_tb_data_r, d_col_size_l, d_col_size_r, d_col_offset_l,
        d_col_offset_r, d_sep, d_candidates, d_result_offset, d_candidates_char,
        d_test_float);

    cudaMemcpyAsync(h_result_offset, d_result_offset, sizeof(int),
                    cudaMemcpyDeviceToHost, *p_stream);
    cudaMemcpyAsync(h_candidates, d_candidates,
                    MAX_CANDIDATE_COUNT * 2 * sizeof(int),
                    cudaMemcpyDeviceToHost, *p_stream);
    cudaMemcpyAsync(h_candidates_char, d_candidates_char,
                    MAX_CANDIDATE_COUNT * MAX_EID_COL_SIZE * 2 * sizeof(char),
                    cudaMemcpyDeviceToHost, *p_stream);

    p_match_->Append(ball_id, h_result_offset, h_candidates_char);

    // cudaFreeHost(d_candidates);
    // cudaFreeHost(d_tb_data_l);
    // cudaFreeHost(d_tb_data_r);
    // cudaFree(d_col_size_l);
    // cudaFree(d_col_size_r);
    // cudaFree(d_col_offset_l);
    // cudaFree(d_col_offset_r);
    // cudaFree(d_sep->pred_index);
    // cudaFree(d_sep->pred_type);
    // cudaFree(d_sep->pred_threshold);
    std::cout << "AsyncSubmit finished." << std::endl;
  }

private:
  int n_device_ = 0;

  std::unique_lock<std::mutex> *p_hr_start_lck_;
  std::condition_variable *p_hr_start_cv_;

  Scheduler *scheduler_;
  sics::hyperblocker::core::components::DataMngr *data_mngr_;
  ExecutionPlanGenerator *epg_;

  std::unordered_map<int, cudaStream_t *> *p_streams_;

  Match *p_match_;
};

} // namespace components
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_CUH_
