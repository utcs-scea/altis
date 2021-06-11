cmake_minimum_required(VERSION 3.8)

add_subdirectory(cfd)
add_subdirectory(dwt2d)
add_subdirectory(gups)
add_subdirectory(kmeans)
add_subdirectory(lavamd)

add_executable(cfd ${exe_files})
target_link_libraries(cfd AltisCommon cfdLib)
set_target_properties(cfd
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(cfd_double ${exe_files})
target_link_libraries(cfd_double AltisCommon cfddoubleLib)
set_target_properties(cfd_double
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(dwt2d ${exe_files} dwt2d/data/bmp.cpp)
target_include_directories(dwt2d PRIVATE dwt2d/data)
target_link_libraries(dwt2d AltisCommon dwt2dLib)
set_target_properties(dwt2d
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(gups ${exe_files})
target_link_libraries(gups AltisCommon gupsLib)
set_target_properties(gups
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(kmeans ${exe_files})
target_link_libraries(kmeans AltisCommon kmeansLib)
set_target_properties(kmeans
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(lavamd ${exe_files} lavamd/util/timer/timer.c)
target_include_directories(lavamd PRIVATE lavamd/util)
target_link_libraries(lavamd AltisCommon lavamdLib)
set_target_properties(lavamd
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)
