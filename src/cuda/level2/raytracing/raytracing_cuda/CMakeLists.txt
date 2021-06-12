cmake_minimum_required(VERSION 3.18)

add_library(raytracingLib raytracing.cu)
target_link_libraries(raytracingLib -lm)
