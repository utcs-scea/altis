cmake_minimum_required(VERSION 3.8)

set(altis_common_source_files
    ResultDatabase.cpp
    ResultDatabase.h
    OptionParser.cpp
    OptionParser.h
    Option.cpp
    Option.h
    ProgressBar.cpp
    ProgressBar.h)

add_library(AltisCommon STATIC ${altis_common_source_files})
