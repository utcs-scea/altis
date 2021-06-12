cmake_minimum_required(VERSION 3.18)

add_library(whereLib where.cu)
target_link_libraries(whereLib -lm)
