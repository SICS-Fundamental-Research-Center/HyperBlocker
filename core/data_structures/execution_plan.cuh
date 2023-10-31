#ifndef HYPERBLOCKER_CORE_DATA_STRUCTURES_EXECUTION_PLAN_CUH_
#define HYPERBLOCKER_CORE_DATA_STRUCTURES_EXECUTION_PLAN_CUH_

namespace sics::hyper_blocker::core::data_structures {

struct ExecutionPlan {
 public:
  size_t deepth_ = 0;
  size_t width_ = 0;
  uint8_t* pred_location_;
  char* pred_kind_;
  float* pred_threshold_;
};

}  // namespace sics::hyper_blocker::core::data_structures
#endif  // HYPERBLOCKER_CORE_DATA_STRUCTURES_EXECUTION_PLAN_CUH_
