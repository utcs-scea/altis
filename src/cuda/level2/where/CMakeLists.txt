cmake_minimum_required(VERSION 3.8)

add_library(whereLib where.cu)
target_link_libraries(whereLib -lm)
