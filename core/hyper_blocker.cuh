#ifndef HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_
#define HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/components/host_producer.cuh"
#include "core/data_structures/rule.cuh"
#include "core/data_structures/table.cuh"

namespace sics::hyper_blocker::core {

class HyperBlocker {
 public:
  HyperBlocker(const std::string& rule_dir, const std::string& data_path)
      : rule_dir_(rule_dir), data_path_(data_path) {}

  ~HyperBlocker() = default;

  void Initialize() {
    std::cout << " Initialize ():  \n  rule_dir - " << rule_dir_ << std::endl
              << "  data_path - " << data_path_ << std::endl;
    std::vector<std::string> file_list;
    for (auto& iter :
         std::experimental::filesystem::directory_iterator(rule_dir_)) {
      auto rule_path = iter.path();
      file_list.push_back(rule_path.string());
      YAML::Node yaml_node;
      yaml_node = YAML::LoadFile(rule_path.string());
      auto rule =
          yaml_node.as<sics::hyper_blocker::core::data_structures::Rule>();
      rule_vec_.push_back(rule);
    }

    tb_ = std::make_unique<sics::hyper_blocker::core::data_structures::Table>(
        sics::hyper_blocker::core::data_structures::Table(data_path_, ",",
                                                          true));
    sics::hyper_blocker::core::components::HostProducer hp(tb_.get());
    hp.Run();

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

  std::vector<sics::hyper_blocker::core::data_structures::Rule> rule_vec_;

 private:
  const std::string rule_dir_;
  const std::string data_path_;

  std::unique_ptr<sics::hyper_blocker::core::data_structures::Table> tb_;
};

}  // namespace sics::hyper_blocker::core
#endif  // HYPERBLOCKER_CORE_HYPER_BLOCKER_CUH_
