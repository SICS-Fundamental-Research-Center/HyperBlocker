#ifndef HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_
#define HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/components/data_mngr.h"
#include "core/components/execution_plan_generator.h"
#include "core/components/host_producer.h"
#include "core/components/host_reducer.h"

namespace sics {
namespace hyperblocker {
namespace core {

using sics::hyperblocker::core::data_structures::Rule;
using sics::hyperblocker::core::components::HostProducer;
using sics::hyperblocker::core::components::HostReducer;
using sics::hyperblocker::core::components::ExecutionPlanGenerator;
using sics::hyperblocker::core::components::DataMngr;

class HyperBlocker {
public:
  HyperBlocker(const std::string &rule_dir, const std::string &data_path_l)
      : rule_dir_(rule_dir), data_path_l_(data_path_l) {}

  HyperBlocker(const std::string &rule_dir, const std::string &data_path_l,
               const std::string &data_path_r)
      : rule_dir_(rule_dir), data_path_l_(data_path_l),
        data_path_r_(data_path_r){} {}

  ~HyperBlocker() = default;

  void Initialize() {
    std::cout << " Initialize ():  \n  rule_dir - " << rule_dir_ << std::endl
              << "  data_path_l - " << data_path_l_ << " data_path_r_"
              << data_path_r_ << std::endl;
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
    HostProducer hp(*(data_mngr_.get()), *(epg_.get()));
    HostReducer hr;
    // TODO: run hp in parallel.
    // TODO:: run hr in parallel.
    hp.Run();
    hr.Run();
  }

  void Run();

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

  std::unique_ptr<ExecutionPlanGenerator> epg_;
  std::unique_ptr<DataMngr> data_mngr_;
};

} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_
