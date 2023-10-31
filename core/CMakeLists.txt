project(hyper_blocker)

# Directories
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)

#######################
# Libraries
#######################
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
)

#######################
# Artifacts
#######################

file(GLOB CORE_SOURCES
        "*.cpp"
        "*.cu"
        "components/*.cu"
        "components/*.cpp"
        "data_structures/*.cu"
        "data_structures/*.cpp"
        "util/*.cu"
        "util/*.cpp"
        "gpu/*.cpp"
        "gpu/*.cu"
        )

list(FILTER GRAPH_CORE_SOURCES EXCLUDE REGEX ".*_test\\.cpp$")
message(STATUS "CORE source files: ${CORE_SOURCES}")

file(GLOB CORE_HEADERS
        "*.h"
        "*.cuh"
        "components/*.h"
        "components/*.cuh"
        "data_structures/*.h"
        "data_structures/*.cuh"
        "util/*.h"
        "util/*.cuh"
        "gpu/*.cuh"
        )

#######################
# Tests
#######################
if (TEST)
    add_subdirectory(tests)
endif ()

#######################
# Generate libgraph_core
#######################
add_library(hyper_blocker ${CORE_SOURCES} ${CORE_HEADERS})

target_link_libraries(hyper_blocker
        yaml-cpp
        gflags
        )
