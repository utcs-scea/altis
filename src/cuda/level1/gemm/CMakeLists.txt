cmake_minimum_required(VERSION 3.8)

add_library(gemmLib Gemm.cu)
target_compile_options(gemmLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ARCH_FLAGS}>)
target_link_libraries(gemmLib -lcublas)
