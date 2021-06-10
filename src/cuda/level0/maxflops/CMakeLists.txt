cmake_minimum_required(VERSION 3.8)

# required since half required arch flags
# TODO: FindCUDA deprecated
find_package(CUDA)
set(CUDA_ARCH_LIST Auto CACHE STRING
    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default).")

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)

add_library(maxflopsLib MaxFlops.cu)
target_compile_options(maxflopsLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ARCH_FLAGS}>)
