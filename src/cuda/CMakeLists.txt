cmake_minimum_required (VERSION 3.8)

include_directories(common ../common)

set(exe_files
    ${CMAKE_CURRENT_LIST_DIR}/common/main.cu
)

add_subdirectory(level0)
add_subdirectory(level1)
