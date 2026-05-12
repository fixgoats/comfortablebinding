cmake_minimum_required(VERSION 3.24)

project(stuff)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 26)

set(PROJECT_SOURCES src/main.cpp src/SDF.cpp src/periodic.cpp src/vkcore.cpp)
set(MAIN_TARGET disp)

set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ ")
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi -lstdc++")

add_compile_options(-w -fexperimental-library
                    "$<$<CONFIG:release>:-O3;-ffast-math;-march=native>")
add_compile_definitions(VK_NO_PROTOTYPES
                        "$<$<CONFIG:release>:NDEBUG;BOOST_DISABLE_ASSERTS>")

add_subdirectory(deps/spdlog)
find_package(GSL REQUIRED)

# Adding Raylib
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
set(BUILD_EXAMPLES
    OFF
    CACHE BOOL "" FORCE) # don't build the supplied examples
set(BUILD_GAMES
    OFF
    CACHE BOOL "" FORCE) # don't build the supplied example games

FetchContent_Declare(
  raylib
  GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
  GIT_TAG "5.5"
  GIT_PROGRESS TRUE)

FetchContent_MakeAvailable(raylib)
message(STATUS "Found GSL: ${GSL_FOUND}")
message(STATUS "GSL library path: ${GSL_LIBRARIES}")
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED COMPONENTS openmp)
find_package(LAPACK REQUIRED)
message("Found BLAS libraries ${BLAS_LIBRARIES}")
message("Found LAPACK libraries ${BLAS_LIBRARIES}")
find_package(Vulkan REQUIRED COMPONENTS glslang SPIRV-Tools)
set(EIGEN_USE_BLAS)
set(EIGEN_USE_LAPACKE)

add_library(comp_base src/geometry.cpp src/io.cpp)
add_library(tomlpp src/localtomlpp.cpp)
add_library(vma_impl src/vma_impl.cpp)
target_link_libraries(comp_base PUBLIC spdlog hdf5::hdf5 ${BLAS_LIBRARIES}
                                       ${LAPACK_LIBRARIES})
target_link_libraries(vma_impl PRIVATE Vulkan::Vulkan)

add_executable(disp)
add_executable(test src/test.cpp src/vkcore.cpp)
add_executable(dynamic src/coupledosc.cpp)
add_executable(couplings src/couplingmaker.cpp)
add_executable(makemonotile src/makemonotile.cpp)
add_executable(visual src/visual.cpp)

# Shader part
add_custom_command(
  OUTPUT Shaders/schroedrstep.spv
  COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/schroedstep.slang -entry
          rstep -target spirv -stage compute -o Shaders/schroedrstep.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/schroedstep.slang
  VERBATIM)

add_custom_command(
  OUTPUT Shaders/schroedkstep.spv
  COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/schroedstep.slang -entry
          kstep -target spirv -stage compute -o Shaders/schroedkstep.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/schroedstep.slang
  VERBATIM)
add_custom_command(
  OUTPUT Shaders/firstminmax.spv
  COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/firstminmax.slang -entry
          firstminmax -target spirv -stage compute -o Shaders/firstminmax.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/firstminmax.slang
  VERBATIM)
add_custom_command(
  OUTPUT Shaders/xfer.spv
  COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/xfer.slang -entry xfer
          -target spirv -stage compute -o Shaders/xfer.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/xfer.slang
  VERBATIM)
add_custom_command(
  OUTPUT Shaders/minmax.spv
  COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/minmax.slang -entry minmax
          -target spirv -stage compute -o Shaders/minmax.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/minmax.slang
  VERBATIM)
add_custom_command(
  OUTPUT Shaders/colormap.spv
  COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/colormap.slang -entry
          writecolor -stage compute -target spirv -o Shaders/colormap.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/colormap.slang
  VERBATIM)
add_custom_command(
  OUTPUT Shaders/quad.vert.spv
  COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/quad.slang -stage vertex
          -entry vertexMain -target spirv -o Shaders/quad.vert.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/quad.slang
  VERBATIM)

add_custom_command(
  OUTPUT Shaders/quad.frag.spv
  COMMAND slangc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/quad.slang -stage fragment
          -entry fragmentMain -target spirv -o Shaders/quad.frag.spv
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/quad.slang
  VERBATIM)

set_property(
  SOURCE src/sim2d.cpp
  APPEND
  PROPERTY OBJECT_DEPENDS
           Shaders/schroedrstep.spv
           Shaders/schroedkstep.spv
           Shaders/quad.vert.spv
           Shaders/quad.frag.spv
           Shaders/firstminmax.spv
           Shaders/minmax.spv
           Shaders/colormap.spv
           Shaders/xfer.spv)
add_executable(sim2d src/sim2d.cpp src/vkcore.cpp)

set(ALL_TARGETS
    comp_base
    tomlpp
    vma_impl
    disp
    dynamic
    couplings
    makemonotile
    visual
    sim2d
    test)
set(ALL_LIBS comp_base tomlpp vma_impl)
set(ALL_EXES
    disp
    dynamic
    couplings
    makemonotile
    visual
    sim2d
    test)
target_sources(disp PRIVATE ${PROJECT_SOURCES})

# set_property( TARGET disp dynamic couplings test comp_base PROPERTY
# CXX_STANDARD 26)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipoSupported OUTPUT output)
add_definitions(-DVKFFT_BACKEND=0)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/glslang)

if(ipoSupported AND CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Using LTO")
  set_property(
    TARGET disp dynamic couplings
    test
    PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Build type is not release, LTO not enabled.")
else()
  message(STATUS "LTO is not supported: ${output}")
endif()
# find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS C HL)

set_property(
  TARGET ${ALL_EXES} ${ALL_LIBS}
  PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include
           ${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen-5.0.1
           ${CMAKE_CURRENT_SOURCE_DIR}/deps/highfive_dep)
target_include_directories(
  sim2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/deps/vkFFT
                ${CMAKE_CURRENT_SOURCE_DIR}/deps/glslang/glslang/Include)
set_property(TARGET ${ALL_EXES} PROPERTY LINK_LIBRARIES ${ALL_LIBS})
set_property(
  TARGET ${ALL_EXES} ${ALL_LIBS}
  PROPERTY COMPILE_OPTIONS -Wall -Wextra -Wpedantic -Wno-error=uninitialized
           "$<$<CONFIG:debug>:-fsanitize=address;-ggdb3>")
set_property(TARGET ${ALL_EXES} ${ALL_LIBS}
             PROPERTY LINK_OPTIONS "$<$<CONFIG:debug>:-fsanitize=address>")
target_link_libraries(test PRIVATE GSL::gsl Vulkan::Vulkan vma_impl SDL3)
# set_property(TARGET disp sim2d PROPERTY PRECOMPILE_HEADERS hack.hpp)
# set_property( TARGET disp dynamic couplings test comp_base visual PROPERTY
# PRECOMPILE_HEADERS hermEigen.h geometry.h io.h SDF.h typedefs.h mathhelpers.h
# kdtree.h betterexc.h )

target_link_libraries(disp PRIVATE GSL::gsl Vulkan::Vulkan vma_impl)
target_link_libraries(dynamic PRIVATE GSL::gsl Vulkan::Vulkan)
target_link_libraries(couplings PRIVATE raylib)
target_link_libraries(makemonotile PRIVATE raylib spdlog)
target_link_libraries(visual PRIVATE raylib GSL::gsl)
target_link_libraries(sim2d PRIVATE spdlog Vulkan::Vulkan vma_impl glslang SDL3)
if(OpenMP_CXX_FOUND)
  target_link_libraries(disp PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(dynamic PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(couplings PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(makemonotile PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(sim2d PRIVATE OpenMP::OpenMP_CXX)
endif()
