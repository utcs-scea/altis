cmake_minimum_required(VERSION 3.8)

add_library(nwLib needle.cu)
target_link_libraries(nwLib -lm)
