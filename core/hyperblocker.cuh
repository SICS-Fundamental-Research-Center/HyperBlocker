#ifndef HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_
#define HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_

#include <condition_variable>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/components/data_mngr.h"
#include "core/components/execution_plan_generator.h"
#include "core/components/host_producer.h"
#include "core/components/host_reducer.h"
#include "core/data_structures/match.h"

namespace sics {
namespace hyperblocker {
namespace core {

using sics::hyperblocker::core::components::DataMngr;
using sics::hyperblocker::core::components::ExecutionPlanGenerator;
using sics::hyperblocker::core::components::HostProducer;
using sics::hyperblocker::core::components::HostReducer;
using sics::hyperblocker::core::data_structures::Match;
using sics::hyperblocker::core::data_structures::Rule;

class HyperBlocker {
public:
  HyperBlocker() = delete;

  HyperBlocker(const std::string &rule_dir, const std::string &data_path_l,
               const std::string &data_path_r, const std::string &output_path,
               int n_partitions)
      : rule_dir_(rule_dir), data_path_l_(data_path_l),
        data_path_r_(data_path_r), output_path_(output_path),
        n_partitions_(n_partitions) {
    auto start_time = std::chrono::system_clock::now();

    p_hr_start_mtx_ = std::make_unique<std::mutex>();
    p_hr_start_lck_ =
        std::make_unique<std::unique_lock<std::mutex>>(*p_hr_start_mtx_.get());
    p_hr_start_cv_ = std::make_unique<std::condition_variable>();
    streams_ = std::make_unique<std::unordered_map<int, cudaStream_t *>>();

    epg_ = std::make_unique<ExecutionPlanGenerator>(rule_dir_);

    if (data_path_r_.empty()) {
      std::cout << "DirtyER" << std::endl;
      data_mngr_ =
          std::make_unique<sics::hyperblocker::core::components::DataMngr>(
              data_path_l_, ",", false);
    } else {
      std::cout << "CleanCleanER" << std::endl;
      data_mngr_ =
          std::make_unique<sics::hyperblocker::core::components::DataMngr>(
              data_path_l_, data_path_r_, ",", false);
    }

    p_match_ = std::make_unique<Match>();

    auto end_time = std::chrono::system_clock::now();
    std::cout << "HyperBlocker.Initialize() elapsed: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end_time - start_time)
                         .count() /
                     (double)CLOCKS_PER_SEC
              << std::endl;
  }

  ~HyperBlocker() = default;

  void Run() {

    auto start_time = std::chrono::system_clock::now();

    std::cout << streams_->size() << std::endl;
    HostProducer hp(n_partitions_, data_mngr_.get(), epg_.get(), streams_.get(),
                    p_match_.get(), p_hr_start_lck_.get(),
                    p_hr_start_cv_.get());
    HostReducer hr(output_path_, streams_.get(), p_match_.get(),
                   p_hr_start_lck_.get(), p_hr_start_cv_.get());

    std::thread hp_thread(&HostProducer::Run, &hp);
    std::thread hr_thread(&HostReducer::Run, &hr);

    hp_thread.join();
    hr_thread.join();

    auto end_time = std::chrono::system_clock::now();

    std::cout << "HyperBlocker.Run() elapsed: "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     end_time - start_time)
                         .count() /
                     (double)CLOCKS_PER_SEC
              << std::endl;
  }

  void ShowDeviceProperties() {
    cudaError_t cudaStatus;
    std::cout << "Device properties" << std::endl;
    int dev = 0;
    cudaDeviceProp devProp;
    cudaStatus = cudaGetDeviceCount(&dev);
    printf("error %d\n", cudaStatus);
    // if (cudaStatus) return;
    for (int i = 0; i < dev; i++) {
      cudaGetDeviceProperties(&devProp, i);
      std::cout << "Device " << dev << ": " << devProp.name << std::endl;
      std::cout << "multiProcessorCount: " << devProp.multiProcessorCount
                << std::endl;
      std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock / 1024.0
                << " KB" << std::endl;
      std::cout << "maxThreadsPerBlock：" << devProp.maxThreadsPerBlock
                << std::endl;
      std::cout << "maxThreadsPerMultiProcessor："
                << devProp.maxThreadsPerMultiProcessor << std::endl;
      std::cout << "maxThreadsPerMultiProcessor："
                << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
      std::cout << std::endl;
    }
  }

  std::vector<Rule> rule_vec_;

private:
  const std::string rule_dir_;
  const std::string data_path_l_;
  const std::string data_path_r_;
  const std::string output_path_;

  const int n_partitions_ = 1;

  std::unique_ptr<std::mutex> p_hr_start_mtx_;
  std::unique_ptr<std::unique_lock<std::mutex>> p_hr_start_lck_;
  std::unique_ptr<std::condition_variable> p_hr_start_cv_;

  std::unique_ptr<ExecutionPlanGenerator> epg_;
  std::unique_ptr<DataMngr> data_mngr_;
  std::unique_ptr<std::unordered_map<int, cudaStream_t *>> streams_;

  std::unique_ptr<Match> p_match_;
};

} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_
