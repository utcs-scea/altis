cmake_minimum_required(VERSION 3.8)

add_library(cfdLib euler3d.cu)
target_link_libraries(cfdLib -lm)

add_library(cfddoubleLib euler3d_double.cu)
target_link_libraries(cfddoubleLib -lm)
