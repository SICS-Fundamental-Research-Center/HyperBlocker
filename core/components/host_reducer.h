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
  HostReducer(const std::string &output_path,
              std::unordered_map<int, cudaStream_t *> *p_streams,
              Match *p_match, std::unique_lock<std::mutex> *p_hr_start_lck,
              std::condition_variable *p_hr_start_cv)
      : output_path_(output_path), p_streams_(p_streams), p_match_(p_match),
        p_hr_start_lck_(p_hr_start_lck), p_hr_start_cv_(p_hr_start_cv) {
    // Get device information.
    cudaError_t cudaStatus;
    std::cout << "Device properties" << std::endl;
    cudaDeviceProp devProp;
    cudaStatus = cudaGetDeviceCount(&n_device_);
  }

  void Run() {
    p_hr_start_cv_->wait(*p_hr_start_lck_,
                         [&] { return p_streams_->size() > 0; });

    auto start_time = std::chrono::system_clock::now();
    std::cout << "Host Reducer running on " << n_device_ << " devices."
              << std::endl;

    while (p_streams_->size() > 0) {
      for (auto iter = p_streams_->begin(); iter != p_streams_->end();) {

        if (cudaStreamQuery(*iter->second) == cudaSuccess) {
          cudaStreamSynchronize(*iter->second);
          std::cout << "Ball id " << iter->first << "/" << p_streams_->size()
                    << " output: "
                    << p_match_->GetNCandidatesbyBallID(iter->first)
                    << std::endl;
          WriteMatch(p_match_->GetNCandidatesbyBallID(iter->first),
                     p_match_->GetCandidatesBasePtr(iter->first));
          iter = p_streams_->erase(iter);
        } else {
          ++iter;
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
  int n_device_ = 0;

  std::unique_lock<std::mutex> *p_hr_start_lck_;
  std::condition_variable *p_hr_start_cv_;

  std::unordered_map<int, cudaStream_t *> *p_streams_;

  Match *p_match_;
};

} // namespace components
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_HOST_REDUCER_CUH_
