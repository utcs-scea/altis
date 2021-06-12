cmake_minimum_required(VERSION 3.18)

add_library(sradLib
    srad.cu
)
target_link_libraries(sradLib -lm -lcudadevrt -lcudart -lcublas)
target_compile_options(sradLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${ARCH_FLAGS} -rdc=true>) 
set_target_properties(sradLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

