cmake_minimum_required(VERSION 3.24)

project(stuff)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 26)

set(PROJECT_SOURCES
    src/main.cpp
    src/SDF.cpp
    src/periodic.cpp
    src/dynamic.cpp
    src/vkcore.cpp
)
set(MAIN_TARGET disp)

set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
# $<$<CONFIG>:-Ofast -march=native> $<$<CONFIG>:-fsanitize=address
# -static-libasan -g>) add_compile_definitions($<$<CONFIG>:NDEBUG>
# $<$<CONFIG>:BOOST_DISABLE_ASSERTS>)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wno-error=uninitialized
    -fexperimental-library
    "$<$<CONFIG:debug>:-fsanitize=address;-ggdb3>"
    "$<$<CONFIG:release>:-O3;-ffast-math;-march=native>"
)
add_compile_definitions("$<$<CONFIG:release>:NDEBUG;BOOST_DISABLE_ASSERTS>")
add_link_options("$<$<CONFIG:debug>:-fsanitize=address>")

add_subdirectory(deps/spdlog)
find_package(GSL REQUIRED)

# Adding Raylib
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
set(BUILD_GAMES OFF CACHE BOOL "" FORCE) # don't build the supplied example games

FetchContent_Declare(
    raylib
    GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
    GIT_TAG "5.5"
    GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(raylib)
message(STATUS "Found GSL: ${GSL_FOUND}")
message(STATUS "GSL library path: ${GSL_LIBRARIES}")
include_directories(include deps/eigen-5.0.1 deps/highfive_dep)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED COMPONENTS openmp)
find_package(LAPACK REQUIRED)
message("Found BLAS libraries ${BLAS_LIBRARIES}")
message("Found LAPACK libraries ${BLAS_LIBRARIES}")
find_package(Vulkan REQUIRED COMPONENTS glslang SPIRV-Tools)
set(EIGEN_USE_BLAS)
set(EIGEN_USE_LAPACKE)

add_library(comp_base src/geometry.cpp src/io.cpp)
target_link_libraries(
    comp_base
    PUBLIC spdlog hdf5::hdf5 ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
)

add_executable(disp)
add_executable(test src/test.cpp src/vkcore.cpp)
add_executable(dynamic)
add_executable(couplings)

target_sources(
    dynamic
    PRIVATE src/coupledosc.cpp src/dynamic.cpp src/vkcore.cpp
)
target_sources(disp PRIVATE ${PROJECT_SOURCES})
target_sources(couplings PRIVATE src/couplingmaker.cpp)

# set_property( TARGET disp dynamic couplings test comp_base PROPERTY
# CXX_STANDARD 26)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipoSupported OUTPUT output)
target_compile_options(disp PRIVATE -Wall)

if(ipoSupported AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Using LTO")
    set_property(
        TARGET disp dynamic couplings test
        PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE
    )
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build type is not release, LTO not enabled.")
else()
    message(STATUS "LTO is not supported: ${output}")
endif()
# find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS C HL)

set_property(
    TARGET disp dynamic couplings test
    PROPERTY LINK_LIBRARIES comp_base
)
target_link_libraries(test PRIVATE GSL::gsl Vulkan::Vulkan)
set_property(
    TARGET disp dynamic couplings test comp_base
    PROPERTY
        PRECOMPILE_HEADERS
            hermEigen.h
            geometry.h
            io.h
            SDF.h
            typedefs.h
            mathhelpers.h
            kdtree.h
            betterexc.h
)

target_link_libraries(disp PRIVATE GSL::gsl Vulkan::Vulkan)
target_link_libraries(dynamic PRIVATE GSL::gsl Vulkan::Vulkan)
target_link_libraries(couplings PRIVATE raylib)
if(OpenMP_CXX_FOUND)
    target_link_libraries(disp PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(dynamic PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(couplings PRIVATE OpenMP::OpenMP_CXX)
endif()
