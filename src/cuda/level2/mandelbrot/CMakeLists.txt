cmake_minimum_required(VERSION 3.18)

add_library(mandelbrotLib
    mandelbrot.cu
)
target_link_libraries(mandelbrotLib -lm -lcudadevrt -lcudart)
target_compile_options(mandelbrotLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${ARCH_FLAGS} -rdc=true -dlink>) 
set_target_properties(mandelbrotLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

