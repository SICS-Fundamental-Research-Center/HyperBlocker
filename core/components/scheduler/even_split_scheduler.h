#ifndef HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_EVEN_SPLIT_SCHEDULER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_EVEN_SPLIT_SCHEDULER_H_

#include "core/components/scheduler/scheduler.h"
#include "core/util/bitmap.h"

class EvenSplitScheduler : public Scheduler {
public:
  EvenSplitScheduler(int n_device) : Scheduler(n_device) {
    bitmap_.Init(n_device);
  }

  int GetBinID(int ball_id = 0) override {
    for (int i = 0; i < get_n_device(); i++)
      if (!bitmap_.GetBit(i)) {
        bitmap_.SetBit(i);
        return i;
      }
    return 0;
  }

private:
  Bitmap bitmap_;
};

#endif // HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_EVEN_SPLIT_SCHEDULER_H_
