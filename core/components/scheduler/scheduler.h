#ifndef HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_SCHEDULER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_SCHEDULER_H_

namespace sics {
namespace hyperblocker {
namespace core {
namespace components {
namespace scheduler {

enum SchedulerType {
  kEvenSplit,
  kCHBL, // default
  kRoundRobin,
};

class Scheduler {
public:
  Scheduler(int n_device) : n_device_(n_device) {
    bin_id_by_ball_id_ = new int[n_device]();
  }

  virtual int GetBinID(int ball_id) = 0;

  int get_n_device() const { return n_device_; }

  virtual void Release(int bin_id, int n_threads) = 0;

  virtual void Consume(int bin_id, int n_threads) = 0;

  int get_bin_id_by_ball_id(int ball_id) const {
    return bin_id_by_ball_id_[ball_id];
  }

  void set_bin_id_by_ball_id(int ball_id, int bin_id) {
    bin_id_by_ball_id_[ball_id] = bin_id;
  }

protected:
  int n_device_ = 0;

  int *bin_id_by_ball_id_;
};

} // namespace scheduler
} // namespace components
} // namespace core
} // namespace hyperblocker
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_SCHEDULER_H_
