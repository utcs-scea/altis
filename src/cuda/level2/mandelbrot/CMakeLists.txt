cmake_minimum_required(VERSION 3.8)

add_library(mandelbrotLib
    mandelbrot.cu
)
target_link_libraries(mandelbrotLib -lm -lcudadevrt -lcudart)
if(${CMAKE_VERSION} VERSION_LESS "3.18.0") 
    target_compile_options(mandelbrotLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${ARCH_FLAGS} -rdc=true -dlink>)
else()
    target_compile_options(mandelbrotLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -rdc=true -dlink>)
endif()
set_target_properties(mandelbrotLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

