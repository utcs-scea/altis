cmake_minimum_required(VERSION 3.8)

add_library(lavamdLib
    lavaMD.cu
    kernel/kernel_gpu_cuda_wrapper.cu
)
target_link_libraries(lavamdLib -lm)
target_include_directories(lavamdLib PRIVATE kernel)
