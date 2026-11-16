# ============================================================================
#  SPH CUDA -- GPU port of the 3D SPH fluid-structure interaction solver.
#
#  The solver is templated on the scalar type; this project decides which
#  instantiations are built and which one is the default:
#
#      -DSPH_ENABLE_FP64=ON        build the double-precision path      (default ON)
#      -DSPH_ENABLE_FP32=ON        build the single-precision path      (default ON)
#      -DSPH_DEFAULT_PRECISION=... default when no flag/env is given    (double)
#
#  With both enabled (default) the choice is made at run time:
#      sph_cuda.exe --fp32        /  --fp64        /  set SPH_PRECISION=fp32
#  Turning one off halves the compile time and the binary size.
#
#  Conventions inherited from the original CMake project:
#    * no absolute paths anywhere;
#    * out-of-source build only (build/ next to this file);
#    * build with build.bat, or manually:
#         cmake -S . -B build && cmake --build build --config Release
#
#  CUDA_ARCHITECTURES = native  ->  compile for the GPU actually installed.
# ============================================================================
cmake_minimum_required(VERSION 3.24)

# Must be set before project(): CMake otherwise picks the compiler default
# (sm_75 for CUDA 13), which is both slow and forward-incompatible.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()

project(sph_cuda
        VERSION 1.1.0
        DESCRIPTION "3D SPH fluid-structure interaction solver (CUDA, fp64/fp32)"
        LANGUAGES CXX CUDA)

if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed. Use:\n"
        "  cmake -S . -B build && cmake --build build --config Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# precision instantiations
# ---------------------------------------------------------------------------
option(SPH_ENABLE_FP64 "build the double-precision path" ON)
option(SPH_ENABLE_FP32 "build the single-precision path" ON)
set(SPH_DEFAULT_PRECISION "double" CACHE STRING "default precision when no flag is given (double|float)")
set_property(CACHE SPH_DEFAULT_PRECISION PROPERTY STRINGS double float)

if(NOT SPH_ENABLE_FP64 AND NOT SPH_ENABLE_FP32)
    message(FATAL_ERROR "at least one of SPH_ENABLE_FP64 / SPH_ENABLE_FP32 must be ON")
endif()

add_executable(sph_cuda
    src/sph_kernels.cu
    src/sph_host.cu)

target_include_directories(sph_cuda PRIVATE src)

# keep the fp64 maths in the same order as the reference build
target_compile_options(sph_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /wd4819>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>)

target_compile_definitions(sph_cuda PRIVATE
    $<$<BOOL:${SPH_ENABLE_FP64}>:SPH_HAVE_FP64=1>
    $<$<NOT:$<BOOL:${SPH_ENABLE_FP64}>>:SPH_HAVE_FP64=0>
    $<$<BOOL:${SPH_ENABLE_FP32}>:SPH_HAVE_FP32=1>
    $<$<NOT:$<BOOL:${SPH_ENABLE_FP32}>>:SPH_HAVE_FP32=0>)

if(SPH_DEFAULT_PRECISION MATCHES "^[Ff]")
    target_compile_definitions(sph_cuda PRIVATE SPH_DEFAULT_FP32=1)
endif()

message(STATUS "CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA compiler      : ${CMAKE_CUDA_COMPILER}")
message(STATUS "fp64 path          : ${SPH_ENABLE_FP64}")
message(STATUS "fp32 path          : ${SPH_ENABLE_FP32}")
message(STATUS "default precision  : ${SPH_DEFAULT_PRECISION}")
message(STATUS "executable         : ${CMAKE_BINARY_DIR}/sph_cuda.exe")
