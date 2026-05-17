cmake_minimum_required(VERSION 3.12)
project(code_opt_bonus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release for performance benchmarking.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Optional dependencies for bonus optimizations ----
find_package(Threads REQUIRED)
find_package(OpenMP QUIET)

# ---- Kernel library ----
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/kernel/*.cpp"
)

add_library(kernels STATIC ${KERNEL_SOURCES})

target_include_directories(kernels
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
)

# Keep symbols and frame pointers for perf, but compile aggressively.
target_compile_options(kernels
    PRIVATE
        -O3
        -DNDEBUG
        -march=native
        -mtune=native
        -ffast-math
        -fno-math-errno
        -fno-trapping-math
        -funroll-loops
        -Wall
        -g
        -fno-omit-frame-pointer
)

# Enable link-time optimization when supported by the compiler/toolchain.
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if (IPO_SUPPORTED)
    set_property(TARGET kernels PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Link threading support for std::thread.
target_link_libraries(kernels PUBLIC Threads::Threads)

# Link OpenMP if the server provides it. This allows #pragma omp in bonus kernels.
if (OpenMP_CXX_FOUND)
    target_link_libraries(kernels PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(kernels PUBLIC BONUS_USE_OPENMP=1)
else()
    target_compile_definitions(kernels PUBLIC BONUS_USE_OPENMP=0)
endif()

# ---- Executables ----
add_executable(single_bench src/main/single_bench.cpp)
target_link_libraries(single_bench PRIVATE kernels)

add_executable(run_all src/main/run_all.cpp)
target_link_libraries(run_all PRIVATE kernels)

# Apply IPO to executables too when available.
if (IPO_SUPPORTED)
    set_property(TARGET single_bench PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET run_all PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()