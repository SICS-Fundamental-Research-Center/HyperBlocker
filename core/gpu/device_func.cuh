#ifndef HYPERBLOCKER_CORE_DEVICE_FUNC_SIM_CUH_
#define HYPERBLOCKER_CORE_DEVICE_FUNC_SIM_CUH_

__device__ bool eq(char* attr_l, char* attr_r) {
  size_t i = 0;
  size_t len_r = 0, len_l = 0;
  while (*(attr_l + i) != '\0' && *(attr_r + i) != '\0') {
    i++;
    len_l++;
    len_r++;
    if (*(attr_l + i) != *(attr_r + i)) {
      return false;
    }
  }
  while (*(attr_l + len_l) != '\0' && len_l < 256) {
    len_l++;
  }
  while (*(attr_r + len_r) != '\0' && len_r < 256) {
    len_r++;
  }
  if (len_l != len_r) {
    return false;
  } else {
    return true;
  }
}

#endif  // HYPERBLOCKER_CORE_DEVICE_FUNC_SIM_CUH_
