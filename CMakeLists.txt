cmake_minimum_required(VERSION 2.8.12)
project(djcdata)

include(CMakePrintHelpers)

# Set source directory
set(SOURCE_DIR "src/djcdata/src")
# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR}/../interface)

file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.c")

add_subdirectory(lib/pybind11)
pybind11_add_module(compiled ${SOURCES} "${SOURCE_DIR}/bind/bindings.cpp")

#shared lib lib
include_directories(lib/pybind11/include)
# add python etc to be added, also copying includes etc
# add_library(djcdata SHARED ${SOURCES})
