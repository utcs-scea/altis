cmake_minimum_required(VERSION 3.8)

add_library(sradLib
    srad.cu
)
target_link_libraries(sradLib -lm -lcudadevrt -lcudart -lcublas)
if(${CMAKE_VERSION} VERSION_LESS "3.18.0") 
    target_compile_options(sradLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${ARCH_FLAGS} -rdc=true>)
else()
    target_compile_options(sradLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -rdc=true>)
endif()
set_target_properties(sradLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

