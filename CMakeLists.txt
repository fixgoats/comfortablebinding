cmake_minimum_required(VERSION 3.24)

project(stuff)

set(CMAKE_C_STANDARD 99)

set(PROJECT_SOURCES
    src/main.cpp
    src/geometry.cpp
    src/hermEigen.cpp
    src/SDF.cpp
    #src/vkcore.cpp
    src/io.cpp
    src/periodic.cpp
    src/dynamic.cpp
    #src/vkhelpers.cpp
)
set(MAIN_TARGET disp)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED COMPONENTS openmp)
find_package(LAPACK REQUIRED)
find_package(Vulkan REQUIRED COMPONENTS glslang SPIRV-Tools)
set(EIGEN_USE_BLAS)
message("Found BLAS libraries ${BLAS_LIBRARIES}")
message("Found LAPACK libraries ${BLAS_LIBRARIES}")
set(EIGEN_USE_LAPACKE)

add_executable(${MAIN_TARGET})
add_executable(test src/test.cpp)
add_executable(dynamic)
add_executable(couplings)
target_sources(
    dynamic
    PRIVATE
        src/coupledosc.cpp
        src/dynamic.cpp
        src/io.cpp
        src/geometry.cpp
        src/hermEigen.cpp
)

target_sources(${MAIN_TARGET} PRIVATE ${PROJECT_SOURCES})
target_sources(couplings
PRIVATE
src/couplingmaker.cpp
src/io.cpp
src/geometry.cpp)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wno-error=maybe-uninitialized
    $<$<CONFIG:RELEASE>:-Ofast
    -march=native>
    $<$<CONFIG:DEBUG>:-O0>
    $<$<CONFIG:DEBUG>:-ggdb3
    -fsanitize=address>
)
add_compile_definitions(
    $<$<CONFIG:RELEASE>:NDEBUG>
    $<$<CONFIG:RELEASE>:BOOST_DISABLE_ASSERTS>
)

set_property(TARGET ${MAIN_TARGET} PROPERTY CXX_STANDARD 26)
set_property(TARGET dynamic PROPERTY CXX_STANDARD 26)
set_property(TARGET couplings PROPERTY CXX_STANDARD 26)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipoSupported OUTPUT output)
target_compile_options(${MAIN_TARGET} PRIVATE -Wall)

target_include_directories(${MAIN_TARGET} PRIVATE include deps/eigen-5.0.1 deps/highfive_dep)
target_include_directories(test PRIVATE include deps/eigen-5.0.1 deps/highfive_dep)
target_include_directories(dynamic PRIVATE include deps/eigen-5.0.1 deps/highfive_dep)
target_include_directories(couplings PRIVATE include deps/eigen-5.0.1 deps/highfive_dep)

if(ipoSupported AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Using LTO")
    set_property(
        TARGET ${MAIN_TARGET}
        PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE
    )
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(VERBOSE "Build type is not release, LTO not enabled.")
else()
    message(WARNING "LTO is not supported: ${output}")
endif()
# find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS C HL)

if(NOT OpenMP_CXX_FOUND)
    target_link_libraries(${MAIN_TARGET} PRIVATE hdf5::hdf5 GSL::gsl)
    target_link_libraries(dynamic PRIVATE hdf5::hdf5 GSL::gsl)
    target_link_libraries(
        couplings
        PRIVATE
            hdf5::hdf5
            GSL::gsl
    )
else()
    target_link_libraries(
        ${MAIN_TARGET}
        PRIVATE
            hdf5::hdf5
            OpenMP::OpenMP_CXX
            GSL::gsl
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
    )
    target_link_libraries(
        dynamic
        PRIVATE
            hdf5::hdf5
            OpenMP::OpenMP_CXX
            Vulkan::Vulkan
            GSL::gsl
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
    )
    target_link_libraries(
       couplings 
        PRIVATE
            hdf5::hdf5
            OpenMP::OpenMP_CXX
            Vulkan::Vulkan
            GSL::gsl
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
    )
endif()
