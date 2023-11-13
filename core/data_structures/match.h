#ifndef HYPERBLOCKER_CORE_DATA_STRUCTURES_MATCH_H_
#define HYPERBLOCKER_CORE_DATA_STRUCTURES_MATCH_H_

#include <mutex>
#include <vector>

namespace sics {
namespace hyperblocker {
namespace core {
namespace data_structures {

class Match {
public:
  Match() { p_mtx_ = new std::mutex(); }

  void Append(int ball_id, int *p_n_candidates, int *p_candidates) {
    const std::lock_guard<std::mutex> lock(*p_mtx_);
    candidates_map_.insert(std::make_pair(ball_id, p_candidates));
    n_candidates_map_.insert(std::make_pair(ball_id, p_n_candidates));
  }

  int GetNCandidatesbyBallID(int ball_id) {
    const std::lock_guard<std::mutex> lock(*p_mtx_);
    auto iter = n_candidates_map_.find(ball_id);
    if (iter != n_candidates_map_.end()) {
      return *(iter->second);
    } else {
      return INT_MAX;
    }
  }

  int *GetCandidatesBasePtr(int ball_id) {
    const std::lock_guard<std::mutex> lock(*p_mtx_);
    auto iter = candidates_map_.find(ball_id);
    if (iter != candidates_map_.end()) {
      return iter->second;
    } else {
      return nullptr;
    }
  }

private:
  std::mutex *p_mtx_;
  std::unordered_map<int, int *> n_candidates_map_;
  std::unordered_map<int, int *> candidates_map_;
};

} // namespace data_structures
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // INC_51_11_ER_HYPERBLOCKER_CORE_DATA_STRUCTURES_MATCH_H_
