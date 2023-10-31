/**
 * @section DESCRIPTION
 *
 * Read csv file
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

using sics::hyper_blocker::core::data_structures::Rule;

cudaError_t cudaStatus;

DEFINE_string(i, "", "input path.");
DEFINE_string(rule, "", "root path of rules.");
DEFINE_string(o, "", "output path.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int dev = 0;
  cudaDeviceProp devProp;

  cudaStatus = cudaGetDeviceCount(&dev);
  printf("error %d\n", cudaStatus);
  for (int i = 0; i < dev; i++) {
    cudaGetDeviceProperties(&devProp, i);
    std::cout << "Device " << dev << ": " << devProp.name << std::endl;
    std::cout << "multiProcessorCount: " << devProp.multiProcessorCount
              << std::endl;
    std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock / 1024.0
              << " KB" << std::endl;
    std::cout << "maxThreadsPerBlock：" << devProp.maxThreadsPerBlock
              << std::endl;
    std::cout << "maxThreadsPerMultiProcessor："
              << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerMultiProcessor："
              << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    std::cout << std::endl;
  }

  io::CSVReader<3> in(FLAGS_i);
  // in.read_header(io::ignore_extra_column, "vendor", "size", "speed");
  std::string vendor;
  std::string col1, col2, col3;
  while (in.read_row(col1, col2, col3)) {
    std::cout << col1 << " " << col2 << " " << col3 << std::endl;
  }

  // Load Yaml node (Edgelist metadata).
  YAML::Node input_node;
  input_node = YAML::LoadFile(FLAGS_rule + "rules/0.yaml");
  std::cout << FLAGS_rule + "rules/0.yaml" << std::endl;
  auto rule = input_node.as<Rule>();
  rule.ShowRule();

  gflags::ShutDownCommandLineFlags();
  return 0;
}
