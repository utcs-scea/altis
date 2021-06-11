cmake_minimum_required(VERSION 3.8)

add_subdirectory(bfs)
add_subdirectory(gemm)
add_subdirectory(pathfinder)
add_subdirectory(sort)
add_subdirectory(gups)

add_executable(bfs ${exe_files})
target_link_libraries(bfs AltisCommon bfsLib)
set_target_properties(bfs
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level1"
)

add_executable(gemm ${exe_files})
target_link_libraries(gemm AltisCommon gemmLib)
set_target_properties(gemm
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level1"
)

add_executable(pathfinder ${exe_files})
target_link_libraries(pathfinder AltisCommon pathfinderLib)
set_target_properties(pathfinder
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level1"
)

add_executable(sort ${exe_files})
target_link_libraries(sort AltisCommon sortLib)
set_target_properties(sort
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level1"
)

add_executable(gups ${exe_files})
target_link_libraries(gups AltisCommon gupsLib)
set_target_properties(gups
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level1"
)
