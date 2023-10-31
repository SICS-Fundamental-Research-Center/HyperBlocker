#ifndef HYPERBLOCKER_CORE_COMPONENTS_EXECUTIONPLANGENERATOR_CUH_
#define HYPERBLOCKER_CORE_COMPONENTS_EXECUTIONPLANGENERATOR_CUH_

#include <string>

namespace sics::hyper_blocker::core::data_structures {
class ExecutionPlanGenerator {
 public:
  ExecutionPlanGenerator(const std::string& root_path)
      : root_path_(root_path) {}

  void Prioritizer();

 private:
  const std::string root_path_;
};

}  // namespace sics::hyper_blocker::core::data_structures
#endif  // HYPERBLOCKER_CORE_COMPONENTS_EXECUTIONPLANGENERATOR_CUH_
