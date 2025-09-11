cmake_minimum_required(VERSION 3.24)

project(stuff)

set(CMAKE_C_STANDARD 99)


set(PROJECT_SOURCES
        src/main.cpp
)
set(MAIN_TARGET disp)

add_executable(${MAIN_TARGET})
target_sources(${MAIN_TARGET} PRIVATE ${PROJECT_SOURCES})

set_property(TARGET ${MAIN_TARGET} PROPERTY CXX_STANDARD 23)
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
target_compile_options(${MAIN_TARGET} PRIVATE -Wall -Wextra -Wpedantic)

target_include_directories(${MAIN_TARGET} PRIVATE include)

if(result AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Using LTO")
    set_property(TARGET ${MAIN_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(VERBOSE "Build type is not release, LTO not enabled.")
else()
    message(WARNING "LTO is not supported: ${output}")
endif()
find_package(Eigen3 REQUIRED NO_MODULE)
FIND_PACKAGE(HDF5 REQUIRED COMPONENTS C CXX HL)

target_link_libraries(${MAIN_TARGET} PRIVATE Eigen3::Eigen hdf5::hdf5_cpp hdf5::hdf5)
