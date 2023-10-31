cmake_minimum_required(VERSION 3.15)
project(hyper_blocker LANGUAGES CUDA)

###### Cmake options ######

# Set default cmake type to Debug
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif ()

message(STATUS "[Graph Systems] will build in type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER} - ${CMAKE_CXX_COMPILER_VERSION}, ${CMAKE_CUDA_COMPILER} - ${CMAKE_CUDA_COMPILER_VERSION}")

###### Directories shortcuts ######
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRD_PARTY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)
set(CMAKE_MODULE_PATH ${PROJECT_ROOT_DIR}/cmake)

#######################
# Libraries
#######################
if (USE_JEMALLOC)
    include("cmake/FindJemalloc.cmake" OPTIONAL)
    if (NOT JEMALLOC_FOUND)
        message(STATUS "jemalloc not found, build without jemalloc")
    else ()
        add_definitions(-DUSE_JEMALLOC)
        include_directories(SYSTEM ${JEMALLOC_INCLUDE_DIRS})
    endif ()
endif ()

###### Add third party to the project, if necessary ######
# googleTest
if (TEST)
    enable_testing()
    message(STATUS "Enabling GoogleTest...")
    set(BUILD_GMOCK ON CACHE BOOL "Option: build gmock and gtest -- ON" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Option: install gtest -- OFF" FORCE)

    include_directories(SYSTEM ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
    include_directories(SYSTEM ${gmock_SOURCE_DIR}/include ${gmock_SOURCE_DIR})
    add_subdirectory("${THIRD_PARTY_ROOT}/googletest" EXCLUDE_FROM_ALL)
endif ()

# gflags
find_package(gflags CONFIG REQUIRED)
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIR})

# find TBB
find_package(TBB)
if (TBB_FOUND)
    message(STATUS "TBB found")
    add_compile_definitions(TBB_FOUND)
endif ()

# yaml
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
set(YAML_CPP_INSTALL OFF)
set(YAML_CPP_BUILD_TESTS OFF)
set(YAML_CPP_BUILD_TOOLS OFF)
include_directories(SYSTEM ${THIRD_PARTY_ROOT}/yaml-cpp/include/)
add_subdirectory("${THIRD_PARTY_ROOT}/yaml-cpp")

# csv
include_directories(SYSTEM ${THIRD_PARTY_ROOT}/fast-cpp-csv-parser)

# rapidcsv
include_directories(SYSTEM ${THIRD_PARTY_ROOT}/rapidcsv/src)

#######################
# Artifacts
#######################
include_directories(
        ${PROJECT_ROOT_DIR}
)

add_subdirectory(core)
#add_subdirectory(tools)
add_subdirectory(examples)
#include_directories(
#        ${CMAKE_CURRENT_SOURCE_DIR}/core
#)
