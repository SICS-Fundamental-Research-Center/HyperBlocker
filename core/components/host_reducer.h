#ifndef HYPERBLOCKER_CORE_COMPONENTS_HOST_REDUCER_CUH_
#define HYPERBLOCKER_CORE_COMPONENTS_HOST_REDUCER_CUH_

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <mutex>

#include <unistd.h>

namespace sics {
namespace hyperblocker {
namespace core {
namespace components {

using sics::hyperblocker::core::data_structures::Match;

class HostReducer {
public:
  HostReducer(const std::string &output_path, scheduler::Scheduler *scheduler,
              std::unordered_map<int, cudaStream_t *> *p_streams,
              std::mutex *p_streams_mtx, Match *p_match,
              std::unique_lock<std::mutex> *p_hr_start_lck,
              std::condition_variable *p_hr_start_cv, bool *p_hr_terminable)
      : output_path_(output_path), p_streams_(p_streams),
        p_streams_mtx_(p_streams_mtx), p_match_(p_match),
        p_hr_start_lck_(p_hr_start_lck), p_hr_start_cv_(p_hr_start_cv),
        scheduler_(scheduler), p_hr_terminable_(p_hr_terminable) {}

  ~HostReducer() = default;

  void Run() {
    p_hr_start_cv_->wait(*p_hr_start_lck_,
                         [&] { return p_streams_->size() > 0; });

    auto start_time = std::chrono::system_clock::now();
    std::cout << "Host Reducer running" << std::endl;

    while (true) {
      if (*p_hr_terminable_ && p_streams_->size() == 0)
        break;
      for (auto iter = p_streams_->begin(); iter != p_streams_->end();) {
        auto cudaStatus = cudaStreamQuery(*iter->second);
        if (cudaStatus == cudaSuccess) {
          cudaStreamSynchronize(*iter->second);
          std::cout << "Ball id " << iter->first << "/" << p_streams_->size()
                    << " output: "
                    << p_match_->GetNCandidatesbyBallID(iter->first)
                    << std::endl;
          auto bin_id = scheduler_->get_bin_id_by_ball_id(iter->first);
          scheduler_->Release(iter->first, bin_id);
          WriteMatch(p_match_->GetNCandidatesbyBallID(iter->first),
                     p_match_->GetCandidatesBasePtr(iter->first));
          std::lock_guard<std::mutex> lock(*p_streams_mtx_);
          cudaStreamDestroy(*iter->second);
          iter = p_streams_->erase(iter);
          //p_match_->Evict(iter->first);
        } else if (cudaStatus == cudaErrorIllegalAddress) {
          std::cout << cudaGetErrorString(cudaStatus) << std::endl;
        } else if (cudaStatus == cudaErrorNotReady) {
          std::lock_guard<std::mutex> lock(*p_streams_mtx_);
          ++iter;
        } else {
          std::cout << cudaGetErrorString(cudaStatus) << std::endl;
        }
      }
    }
  }

private:
  void WriteMatch(int n_candidates, int *candidates) {
    std::ofstream out;

    out.open(output_path_, std::ios::app);
    for (int i = 0; i < n_candidates; i++)
      out << candidates[i * 2] << "," << candidates[i * 2 + 1] << std::endl;
    out.close();
  }

  void WriteMatch(int n_candidates, char *candidates) {
    std::ofstream out;

    out.open(output_path_, std::ios::app);
    for (int i = 0; i < n_candidates && i < MAX_CANDIDATE_COUNT; i++) {
      out << candidates + MAX_EID_COL_SIZE * 2 * i << ","
          << candidates + MAX_EID_COL_SIZE * 2 * i + MAX_EID_COL_SIZE
          << std::endl;
    }
    out.close();
  }

  const std::string output_path_;

  scheduler::Scheduler *scheduler_;

  std::mutex *p_streams_mtx_;
  std::unique_lock<std::mutex> *p_hr_start_lck_;
  std::condition_variable *p_hr_start_cv_;

  std::unordered_map<int, cudaStream_t *> *p_streams_;

  bool *p_hr_terminable_;

  Match *p_match_;
};

} // namespace components
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_HOST_REDUCER_CUH_
