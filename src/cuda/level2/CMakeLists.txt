cmake_minimum_required(VERSION 3.8)

add_subdirectory(cfd)
add_subdirectory(dwt2d)
add_subdirectory(fdtd2d)
add_subdirectory(kmeans)
add_subdirectory(lavamd)
add_subdirectory(mandelbrot)
add_subdirectory(nw)
add_subdirectory(particlefilter)
add_subdirectory(raytracing)
add_subdirectory(srad)
add_subdirectory(where)
add_subdirectory(yolo)

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

add_executable(fdtd2d ${exe_files})
target_link_libraries(fdtd2d AltisCommon fdtd2dLib)
set_target_properties(fdtd2d
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

add_executable(mandelbrot ${exe_files})
target_link_libraries(mandelbrot AltisCommon mandelbrotLib)
set_target_properties(mandelbrot
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(nw ${exe_files})
target_link_libraries(nw AltisCommon nwLib)
set_target_properties(nw
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(particlefilter_naive ${exe_files})
target_link_libraries(particlefilter_naive AltisCommon particlefilterLib)
set_target_properties(particlefilter_naive
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(particlefilter_float ${exe_files})
target_link_libraries(particlefilter_float AltisCommon particlefilterfloatLib)
set_target_properties(particlefilter_float
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(raytracing ${exe_files})
target_link_libraries(raytracing AltisCommon raytracingLib)
set_target_properties(raytracing
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(srad ${exe_files})
set_property(TARGET srad
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(srad AltisCommon sradLib)
set_target_properties(srad
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)

add_executable(where ${exe_files})
target_link_libraries(where AltisCommon whereLib)
set_target_properties(where
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/level2"
)
