#ifndef HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_

#include "core/components/scheduler/scheduler.h"
#include "core/util/bitmap.h"

class RoundRobinScheduler : public Scheduler {
public:
  RoundRobinScheduler(int n_device) : Scheduler(n_device) {
    bitmap_.Init(n_device);
  }

  int GetBinID(int ball_id = 0) override { return ball_id % get_n_device(); }

private:
  Bitmap bitmap_;
};

#endif // HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_
