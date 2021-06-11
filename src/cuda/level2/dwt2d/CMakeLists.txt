cmake_minimum_required(VERSION 3.8)

set(DWT_CUDA_DIR ${CMAKE_CURRENT_LIST_DIR}/dwt_cuda)

add_library(dwt2dLib
    dwt_main.cu
    dwt.cu
    components.cu
    ${DWT_CUDA_DIR}/fdwt53.cu
    ${DWT_CUDA_DIR}/rdwt53.cu
    ${DWT_CUDA_DIR}/fdwt97.cu
    ${DWT_CUDA_DIR}/rdwt97.cu
    ${DWT_CUDA_DIR}/common.cu
)
target_include_directories(dwt2dLib PRIVATE dwt_cuda)
