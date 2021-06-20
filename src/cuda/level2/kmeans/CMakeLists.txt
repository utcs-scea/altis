cmake_minimum_required(VERSION 3.18)

add_library(kmeansLib
    kmmain.cu
    kmeansraw.cu
)
target_link_libraries(kmeansLib -lm)
