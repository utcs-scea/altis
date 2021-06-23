cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(${CMAKE_VERSION} VERSION_LESS "3.17.0")

else()
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()

project(Altis LANGUAGES C CXX CUDA)

add_subdirectory(src)
