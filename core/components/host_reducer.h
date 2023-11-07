#ifndef HYPERBLOCKER_CORE_COMPONENTS_HOST_REDUCER_CUH_
#define HYPERBLOCKER_CORE_COMPONENTS_HOST_REDUCER_CUH_

#include <cuda_runtime.h>

namespace sics {
namespace hyperblocker {
namespace core {
namespace components {
class HostReducer {
public:
  HostReducer() {
    // Get device information.
    cudaError_t cudaStatus;
    std::cout << "Device properties" << std::endl;
    cudaDeviceProp devProp;
    cudaStatus = cudaGetDeviceCount(&n_device_);
  }

  void Run() {
    std::cout << "Host Reducer running on " << n_device_ << " devices."
              << std::endl;
  }

private:
  int n_device_ = 0;
};

} // namespace components
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_HOST_REDUCER_CUH_
