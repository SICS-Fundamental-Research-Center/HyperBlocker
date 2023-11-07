#ifndef HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_SCHEDULER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_SCHEDULER_H_

class Scheduler {
public:
  Scheduler(int n_device) : n_device_(n_device) {}

  virtual int GetBinID(int ball_id) = 0;

  int get_n_device() const { return n_device_; }

private:
  int n_device_ = 0;
};

#endif // HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_SCHEDULER_H_
