#ifndef HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_CUH_
#define HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_CUH_

#include <cuda_runtime.h>

#include "core/data_structures/table.cuh"
#include "core/gpu/host_func.cuh"

namespace sics::hyper_blocker::core::components {

class HostProducer {
 public:
  HostProducer(sics::hyper_blocker::core::data_structures::Table* tb) {
    tb_ = tb;
  }

  void Run() {
    sics::hyper_blocker::core::gpu::host::AsyncBlockingSubmit(*tb_);
  }

 private:
  sics::hyper_blocker::core::data_structures::Table* tb_;
};

}  // namespace sics::hyper_blocker::core::components
#endif  // HYPERBLOCKER_CORE_COMPONENTS_HOST_PRODUCER_CUH_
