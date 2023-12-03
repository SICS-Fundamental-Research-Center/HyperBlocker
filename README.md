# HyperBlocker overview

## What is HyperBlocker?
HyperBlocker is a GPU-accelerated system for blocking in ER. 
As opposed to previous blocking
algorithms and parallel blocking solvers, HyperBlocker employs
a pipelined architecture to overlap data transfer and GPU
operations, and improve parallelism by effectively collaborating
GPUs and CPUs. It also generates a data-aware and rule-aware
execution plan on CPUs, for specifying how rules are evaluated in
blocking. Moreover, it develops a variety of hardware-aware optimization 
and scheduling strategies to achieve massive parallelism
and workload balancing across multiple GPUs. 

## Features
HyperBlocker has the following unique features.

* A pipelined architecture. HyperBlocker adopts an architecture
that pipelines the memory access from/to CPUs for data
transfer, and operations on GPUs for rule-based blocking. In
this way, the data transfer and the computation on GPUs can be
overlapped, so as to “cancel” the excessive data transfer cost.
* Execution plan generation on CPUs. To effectively filter
unqualified tuple pairs, the blocking must be optimized for
the underlying data (resp. blocking rules); in this case, we say
that the blocking is data-aware (resp. rule-aware). To the best
of our knowledge, no prior methods, neither on CPUs nor on
GPUs, have considered data/rule-awareness for their execution
models. HyperBlocker designs an effective execution plan
generator to warrant efficient rule-based blocking.
* Hardware-aware parallelism on GPUs. Due to the different characteristics of CPUs and GPUs, a naive approach that
applies existing CPU-based blocking methods on GPUs makes
substantial processing capacity untapped. HyperBlocker develops a variety of GPUs-based parallelism strategies, designated
for rule-based blocking, by effectively exploiting the hardware
characteristics of GPUs, to achieve massive parallelism.
* Multi-GPUs collaboration. It is already hard to balance the
workload on CPUs. This problem is even exacerbated under
multi-GPU scenarios, since tens of thousands of threads will
compete for limited GPU resources. HyperBlocker provides an
effective task-scheduling strategy to scale with multiple GPUs.


## Getting Started
### Dependencies
HyperBlocker builds, runs, and has been tested on GNU/Linux. 
At the minimum, HyperBlocker depends on the following software:
* A modern C++ compiler compliant with the C++17 standard 
(gcc >= 9)
* CMake (>= 3.2)
* GoogleTest (>= 1.11.0)
* RapidCSV (>= 8.65)
* CUDA (>= 10.0)




### Build

First, clone the project and install dependencies on your environment.

```shell
# Clone the project (SSH).
# Make sure you have your public key has been uploaded to GitHub!
git clone git@github.com:hsiaoko/HyperBlocker.git
# Install dependencies.
$SRC_DIR=`HyperBlocker` # top-level HyperBlocker source dir
$cd $SRC_DIR
$./dependencies.sh
```

Build the project.
```shell
$BUILD_DIR=<path-to-your-build-dir>
$mkdir -p $BUILD_DIR
$cd $BUILD_DIR
$cmake ..
$make
```

### Running HyperBlocker
```shell
$cd $SRC_DIR
$../bin/run_sys_exec -data_l [csv file path] -data_r [csv file path]  -rule_dir [rule path] -n_partitions [the number of partitions] -o [output path]
```


## Contact Us
For bugs, please raise an issue on GiHub. 
Questions and comments are also welcome at my email: 
zhuxk@buaa.edu.cn



