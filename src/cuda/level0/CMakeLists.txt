cmake_minimum_required(VERSION 3.8)

add_subdirectory(busspeeddownload)
add_subdirectory(busspeedreadback)
add_subdirectory(devicememory)
add_subdirectory(maxflops)

add_executable(busspeeddownload ${exe_files})
target_link_libraries(busspeeddownload AltisCommon busspeeddownloadLib)
set_target_properties(busspeeddownload
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level0"
)

add_executable(busspeedreadback ${exe_files})
target_link_libraries(busspeedreadback AltisCommon busspeedreadbackLib)
set_target_properties(busspeedreadback
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level0"
)

add_executable(devicememory ${exe_files})
target_link_libraries(devicememory AltisCommon devicememoryLib)
set_target_properties(devicememory
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level0"
)


add_executable(maxflops
    ${exe_files}
    ${CMAKE_CURRENT_LIST_DIR}/../../common/ProgressBar.cpp
)
target_link_libraries(maxflops AltisCommon maxflopsLib)
set_target_properties(maxflops
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level0"
)
