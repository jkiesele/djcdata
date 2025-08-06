cmake_minimum_required(VERSION 3.4)
project(djcdata)

include(CMakePrintHelpers)

# Set source directory
set(SOURCE_DIR "src/djcdata/src")
# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR}/../interface)

file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.c")

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
pybind11_add_module(compiled ${SOURCES})

# shared lib lib
# add python etc to be added, also copying includes etc
# add_library(djcdata SHARED ${SOURCES})
