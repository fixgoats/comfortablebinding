cmake_minimum_required(VERSION 3.24)

project(stuff)

set(CMAKE_C_STANDARD 99)

set(PROJECT_SOURCES src/main.cpp src/SDF.cpp src/periodic.cpp src/dynamic.cpp
                    src/vkcore.cpp)
set(MAIN_TARGET disp)

set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

add_subdirectory(deps/spdlog)
add_library(comp_base src/geometry.cpp src/io.cpp)
target_link_libraries(comp_base PUBLIC hdf5::hdf5)

find_package(GSL REQUIRED)

message(STATUS ${GSL_FOUND})
message(STATUS ${GSL_LIBRARIES})
include_directories(include deps/eigen-5.0.1 deps/highfive_dep)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED COMPONENTS openmp)
find_package(LAPACK REQUIRED)
message("Found BLAS libraries ${BLAS_LIBRARIES}")
message("Found LAPACK libraries ${BLAS_LIBRARIES}")
find_package(Vulkan REQUIRED COMPONENTS glslang SPIRV-Tools)
set(EIGEN_USE_BLAS)
set(EIGEN_USE_LAPACKE)

add_executable(${MAIN_TARGET})
add_executable(test src/test.cpp src/vkcore.cpp)
add_executable(dynamic)
add_executable(couplings)

target_sources(dynamic PRIVATE src/coupledosc.cpp src/dynamic.cpp
                               src/vkcore.cpp)
target_sources(${MAIN_TARGET} PRIVATE ${PROJECT_SOURCES})
target_sources(couplings PRIVATE src/couplingmaker.cpp)

add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wno-error=maybe-uninitialized
  $<$<CONFIG:RELEASE>:-Ofast
  -march=native>
  $<$<CONFIG:DEBUG>:-O0>
  $<$<CONFIG:DEBUG>:-ggdb3
  -fsanitize=address>)
add_compile_definitions($<$<CONFIG:RELEASE>:NDEBUG>
                        $<$<CONFIG:RELEASE>:BOOST_DISABLE_ASSERTS>)

set_property(
  TARGET ${MAIN_TARGET} dynamic couplings
  test comp_base
  PROPERTY CXX_STANDARD 26)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipoSupported OUTPUT output)
target_compile_options(${MAIN_TARGET} PRIVATE -Wall)

if(ipoSupported AND CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Using LTO")
  set_property(
    TARGET ${MAIN_TARGET} dynamic couplings
    test
    PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(VERBOSE "Build type is not release, LTO not enabled.")
else()
  message(WARNING "LTO is not supported: ${output}")
endif()
# find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS C HL)

set_property(
  TARGET ${MAIN_TARGET} dynamic couplings
  test
  PROPERTY LINK_LIBRARIES comp_base spdlog ${BLAS_LIBRARIES}
           ${LAPACK_LIBRARIES})
target_link_libraries(test PRIVATE Vulkan::Vulkan)
set_property(
  TARGET ${MAIN_TARGET} dynamic couplings
  test comp_base
  PROPERTY PRECOMPILE_HEADERS
           hermEigen.h
           geometry.h
           io.h
           SDF.h
           typedefs.h
           mathhelpers.h
           kdtree.h
           betterexc.h)

target_link_libraries(${MAIN_TARGET} PRIVATE GSL::gsl Vulkan::Vulkan)
target_link_libraries(dynamic PRIVATE GSL::gsl Vulkan::Vulkan)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${MAIN_TARGET} PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(dynamic PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(couplings PRIVATE OpenMP::OpenMP_CXX)
endif()
