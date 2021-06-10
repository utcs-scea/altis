cmake_minimum_required(VERSION 3.8)

find_package(CUDA)
cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)

add_library(gemmLib Gemm.cu)
target_compile_options(gemmLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ARCH_FLAGS}>)
target_link_libraries(gemmLib -lcublas)
