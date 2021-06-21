cmake_minimum_required(VERSION 3.8)

add_library(gemmLib Gemm.cu)
if(${CMAKE_VERSION} VERSION_LESS "3.18.0") 
    target_compile_options(gemmLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ARCH_FLAGS}>)
endif()
target_link_libraries(gemmLib -lcublas)
