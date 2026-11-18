# ============================================================================
#  SPH -- 3D SPH fluid-structure interaction solver (single file 1.cpp)
#
#  Conventions: no absolute paths anywhere in the configuration; everything is
#  relative to the project directory / build directory.
#
#  Standard usage (build directory fixed to build):
#      mkdir build
#      cd build
#      cmake ..                     # generates a Visual Studio 2019 (v142/C++17) solution by default
#      cmake --build . --config Release
#
#  Note: the CMake VS generator always adds the ALL_BUILD helper project and
#        cannot be told not to.  For a clean solution containing only SPH, use
#        build.bat in the project root (it calls tools\strip-helper-projects.ps1
#        after generation).
# ============================================================================
cmake_minimum_required(VERSION 3.20)

project(SPH
        VERSION 1.0.0
        DESCRIPTION "3D SPH fluid-structure interaction solver"
        LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Only out-of-source builds are allowed: the source tree stays clean and the
# build directory can be deleted at any time
# ---------------------------------------------------------------------------
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed.  Use:\n"
        "  mkdir build && cd build && cmake ..")
endif()

# ---------------------------------------------------------------------------
# Build type (single-config generators; with VS, --config decides)
# ---------------------------------------------------------------------------
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SPH_USE_CXX20      "Use C++20 (requires VS2019 16.11+ or VS2022+)" OFF)
option(SPH_GBK_SOURCE     "Force the source to be parsed as GBK (cp936); only needed when the system locale is not Chinese" OFF)
option(SPH_ENABLE_FAST_MATH "Enable /fp:fast (changes numerical results; normally leave off)" OFF)
option(SPH_SUPPRESS_REGENERATION "Do not generate the ZERO_CHECK project; the price is that CMakeLists changes require a manual re-run of cmake" ON)

if(SPH_SUPPRESS_REGENERATION)
    set(CMAKE_SUPPRESS_REGENERATION ON)
endif()

# ---------------------------------------------------------------------------
# Source files: relative paths (relative to this CMakeLists.txt)
# ---------------------------------------------------------------------------
add_executable(SPH 1.cpp)

# ---------------------------------------------------------------------------
# C++ standard
#   Original VS2026 project: v145 + stdcpp20; the VS2019 adaptation uses C++17
#   (fully supported by v142).  For C++20: cmake -DSPH_USE_CXX20=ON ..
# ---------------------------------------------------------------------------
if(SPH_USE_CXX20)
    set_target_properties(SPH PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
else()
    set_target_properties(SPH PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
endif()

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
target_link_libraries(SPH PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# MSVC adjustments
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_definitions(SPH PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(SPH PRIVATE /W3 /MP)

    # The source is now ASCII (the comments were translated from GBK), so the
    # default code page is fine.  The option remains for old GBK checkouts.
    if(SPH_GBK_SOURCE)
        target_compile_options(SPH PRIVATE /source-charset:.936)
    endif()

    if(SPH_ENABLE_FAST_MATH)
        target_compile_options(SPH PRIVATE /fp:fast)
    else()
        target_compile_options(SPH PRIVATE /fp:precise)
    endif()

    # Also generate a PDB in Release / MinSizeRel (/Zi + linker /DEBUG):
    #   when VS debugs an exe with no debug information it goes looking for
    #   symbols on the symbol servers (including Microsoft's); on an offline or
    #   intranet machine it then sits at "Loading symbols..." for a long time.
    #   A Release PDB makes that step instant and allows breakpoints in Release.
    #   It does not affect numerical results.
    foreach(_cfg RELEASE MINSIZEREL)
        target_compile_options(SPH PRIVATE $<$<CONFIG:${_cfg}>:/Zi>)
        target_link_options(SPH PRIVATE $<$<CONFIG:${_cfg}>:/DEBUG>)
    endforeach()

    # Debug settings all use MSBuild macros, never absolute paths:
    #   VS_DEBUGGER_COMMAND        = $(TargetPath), i.e. the exe that was built,
    #                                so F5 works in VS even if OutDir is relative
    #   VS_DEBUGGER_WORKING_DIRECTORY = the directory above the solution = project
    #                                root, because the program reads input.dat
    #                                through a relative path and must run there
    set_target_properties(SPH PROPERTIES
        VS_DEBUGGER_COMMAND "$(TargetPath)"
        VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
else()
    # R1 (review): the neighbour-membership test d0*d0+d1*d1+d2*d2 < R2 is
    # knife-edge on this lattice -- more than a thousand pairs sit exactly at
    # r = 3*hsml -- so FMA contraction changes the reference pair count
    # (-O3 measured 11 102 707 pairs, -march=native 11 103 787).  GCC and Clang
    # contract by default; /fp:precise already keeps MSVC from doing it.
    # -O3 is worth about 1.5x over -O2 on this solver.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(SPH PRIVATE -O3 -ffp-contract=off)
    endif()
endif()

# ---------------------------------------------------------------------------
# Output directories keep the CMake defaults (<build dir>/<Config>); no
# absolute paths
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Configuration summary: always shown with relative paths
# ---------------------------------------------------------------------------
file(RELATIVE_PATH SPH_REL_BUILD "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")
if(SPH_REL_BUILD STREQUAL "")
    set(SPH_REL_BUILD ".")
endif()
file(TO_CMAKE_PATH "${SPH_REL_BUILD}" SPH_REL_BUILD)

message(STATUS "")
message(STATUS "====================== project paths (relative) ======================")
message(STATUS " project root : .")
message(STATUS " build script : CMakeLists.txt")
message(STATUS " source       : 1.cpp")
message(STATUS " build dir    : ${SPH_REL_BUILD}")
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    message(STATUS " solution     : ${SPH_REL_BUILD}/${PROJECT_NAME}.sln")
endif()
message(STATUS " executable   : ${SPH_REL_BUILD}/<Config>/${PROJECT_NAME}.exe")
message(STATUS " input data   : input.dat / result<N>.dat  (working directory = project root)")
message(STATUS "======================================================================")
message(STATUS " generator : ${CMAKE_GENERATOR} platform=${CMAKE_VS_PLATFORM_NAME} toolset=${CMAKE_VS_PLATFORM_TOOLSET}")
if(SPH_USE_CXX20)
    message(STATUS " C++ standard: C++20")
else()
    message(STATUS " C++ standard: C++17")
endif()
if(SPH_SUPPRESS_REGENERATION)
    message(STATUS " ZERO_CHECK: disabled (re-run cmake after changing CMakeLists)")
endif()
message(STATUS "======================================================================")
