#ifndef HYPERBLOCKER_CORE_DATA_STRUCTURES_RULE_CUH_
#define HYPERBLOCKER_CORE_DATA_STRUCTURES_RULE_CUH_

#include <vector>

namespace sics::hyper_blocker::core::data_structures {

struct Preconditions {
  std::vector<unsigned> relation_l;
  std::vector<unsigned> relation_r;
  std::vector<unsigned> eq;
  std::vector<unsigned> sim;
  std::vector<float> threshold;
};

struct Conseq {
  std::vector<unsigned> eq;
};

class Rule {
 public:
  Preconditions pre;
  Conseq con;

  void ShowRule() {
    std::cout << "Rule: " << std::endl;
    std::cout << "  Preconditions: " << std::endl;
    std::cout << "    Relation l: ";
    for (auto& i : pre.relation_l) std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "    Relation r: ";
    for (auto& i : pre.relation_r) std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "    Equalities: ";
    for (auto& i : pre.eq) std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "    Sim: ";
    for (auto& i : pre.sim) std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "    Threshold: ";
    for (auto& i : pre.threshold) std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "  Conseq: " << std::endl;
    std::cout << "    Equality: ";
    for (auto& i : con.eq) std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "#Rule " << std::endl;
  }
};

}  // namespace sics::hyper_blocker::core::data_structures
#endif  // HYPERBLOCKER_CORE_DATA_STRUCTURES_RULE_H_
