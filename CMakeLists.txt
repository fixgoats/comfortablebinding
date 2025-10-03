cmake_minimum_required(VERSION 3.24)

project(stuff)

set(CMAKE_C_STANDARD 99)


set(PROJECT_SOURCES
        src/main.cpp
)
set(MAIN_TARGET disp)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

find_package(OpenMP)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
set(EIGEN_USE_BLAS)
set(EIGEN_USE_LAPACKE)

add_executable(${MAIN_TARGET})
target_sources(${MAIN_TARGET} PRIVATE ${PROJECT_SOURCES})
add_compile_options(
       -Wall -Werror -Wno-error=maybe-uninitialized
       $<$<CONFIG:RELEASE>:-O3>
       $<$<CONFIG:DEBUG>:-O0>
       $<$<CONFIG:DEBUG>:-ggdb3>
)
add_compile_definitions(
        $<$<CONFIG:RELEASE>:NDEBUG>
        $<$<CONFIG:RELEASE>:BOOST_DISABLE_ASSERTS>
)

set_property(TARGET ${MAIN_TARGET} PROPERTY CXX_STANDARD 23)
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
target_compile_options(${MAIN_TARGET} PRIVATE -Wall -Wextra -Wpedantic)

target_include_directories(${MAIN_TARGET} PRIVATE include deps/eigen-5.0.0)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
if(result AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Using LTO")
    set_property(TARGET ${MAIN_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(VERBOSE "Build type is not release, LTO not enabled.")
else()
    message(WARNING "LTO is not supported: ${output}")
endif()
# find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

if(NOT OpenMP_CXX_FOUND)
  target_link_libraries(${MAIN_TARGET} PRIVATE hdf5::hdf5_cpp hdf5::hdf5 lodepng)# Eigen3::Eigen)
else()
  target_link_libraries(${MAIN_TARGET} PRIVATE hdf5::hdf5_cpp hdf5::hdf5 lodepng OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()
