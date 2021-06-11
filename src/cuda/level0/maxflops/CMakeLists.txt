cmake_minimum_required(VERSION 3.8)

add_library(maxflopsLib MaxFlops.cu)
target_compile_options(maxflopsLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ARCH_FLAGS}>)
