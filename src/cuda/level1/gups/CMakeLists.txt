cmake_minimum_required(VERSION 3.18)

add_library(gupsLib gups.cu)
target_link_libraries(gupsLib -lm)
