/**
 * @section DESCRIPTION
 *
 * Run HyperBlocker
 */

#include <csv.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <list>
#include <utility>
//#include <filesystem>

#include "core/common/yaml_config.cuh"
#include "core/data_structures/rule.cuh"
#include "core/hyper_blocker.cuh"

using sics::hyper_blocker::core::HyperBlocker;
using sics::hyper_blocker::core::data_structures::Rule;

DEFINE_string(i, "", "input path.");
DEFINE_string(rule_dir, "", "root path of rules.");
DEFINE_string(o, "", "output path.");
DEFINE_string(sep, "", "separator to split a line of csv file.");
DEFINE_bool(read_header, false, "whether to read header of csv.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  //HyperBlocker hb(FLAGS_rule_dir, FLAGS_i);
  //hb.ShowDeviceProperties();
  //hb.Initialize();


  int *ret;
  cudaMalloc(&ret, 1000 * sizeof(int));
  sics::hyper_blocker::core::gpu::global::AplusB<<< 1, 1000 >>>(ret, 10, 100);
  int *host_ret = (int *)malloc(1000 * sizeof(int));
  cudaMemcpy(host_ret, ret, 1000 * sizeof(int), cudaMemcpyDefault);
  for(int i = 0; i < 1000; i++)
    printf("%d: A+B = %d\n", i, host_ret[i]);
  free(host_ret);
  cudaFree(ret);

  gflags::ShutDownCommandLineFlags();
  return EXIT_SUCCESS;
}
