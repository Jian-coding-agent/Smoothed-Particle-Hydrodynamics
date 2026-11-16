# ============================================================================
#  SPH —— 三维 SPH 流固耦合求解器（单文件 1.cpp）
#
#  约定：配置里不出现任何绝对路径，全部相对于工程目录 / 构建目录。
#
#  标准用法（构建目录固定为 build）：
#      mkdir build
#      cd build
#      cmake ..                     # 默认生成 Visual Studio 2019 (v142/C++17) 解决方案
#      cmake --build . --config Release
#
#  说明：CMake 的 VS 生成器会强制附带 ALL_BUILD 辅助工程，无法在脚本里关掉。
#        需要“解决方案里只剩 SPH”的干净版本时，用工程根目录的 build.bat
#        （它会在生成后调用 tools\strip-helper-projects.ps1 清理）。
# ============================================================================
cmake_minimum_required(VERSION 3.20)

project(SPH
        VERSION 1.0.0
        DESCRIPTION "3D SPH fluid-structure interaction solver"
        LANGUAGES CXX)

# ---------------------------------------------------------------------------
# 只允许 out-of-source 构建：保证源码目录干净，且构建目录可随时整目录删除
# ---------------------------------------------------------------------------
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR
        "不允许在源码目录内构建。请这样用：\n"
        "  mkdir build && cd build && cmake ..")
endif()

# ---------------------------------------------------------------------------
# 构建类型（单配置生成器用；VS 多配置由 --config 决定）
# ---------------------------------------------------------------------------
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---------------------------------------------------------------------------
# 选项
# ---------------------------------------------------------------------------
option(SPH_USE_CXX20      "使用 C++20（需 VS2019 16.11+ 或 VS2022+）" OFF)
option(SPH_GBK_SOURCE     "强制按 GBK(cp936) 解析源码；仅当区域语言环境不是中文时使用" OFF)
option(SPH_ENABLE_FAST_MATH "启用 /fp:fast（会改变数值结果，一般不要开）" OFF)
option(SPH_SUPPRESS_REGENERATION "不生成 ZERO_CHECK 工程；代价是改了 CMakeLists 需手工重跑 cmake" ON)

if(SPH_SUPPRESS_REGENERATION)
    set(CMAKE_SUPPRESS_REGENERATION ON)
endif()

# ---------------------------------------------------------------------------
# 源文件：相对路径（相对本 CMakeLists.txt 所在目录）
# ---------------------------------------------------------------------------
add_executable(SPH 1.cpp)

# ---------------------------------------------------------------------------
# C++ 标准
#   原 VS2026 工程：v145 + stdcpp20；VS2019 适配为 C++17（v142 全面支持）
#   需要 C++20：cmake -DSPH_USE_CXX20=ON ..
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
# MSVC 适配
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_definitions(SPH PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(SPH PRIVATE /W3 /MP)

    # 1.cpp 的中文注释是 GBK：默认跟随系统代码页即可，不要加 /utf-8
    if(SPH_GBK_SOURCE)
        target_compile_options(SPH PRIVATE /source-charset:.936)
    endif()

    if(SPH_ENABLE_FAST_MATH)
        target_compile_options(SPH PRIVATE /fp:fast)
    else()
        target_compile_options(SPH PRIVATE /fp:precise)
    endif()

    # Release / MinSizeRel 也生成 PDB（/Zi + 链接器 /DEBUG）：
    #   VS 调试一个完全没有调试信息的 exe 时，会去符号服务器（含 Microsoft
    #   符号服务器）查找符号；在离线/内网机器上就会长时间停在状态栏
    #   “正在加载符号…”。Release 自带 PDB 后这一步立刻完成，顺便还能在
    #   Release 配置下打断点。不影响数值结果。
    foreach(_cfg RELEASE MINSIZEREL)
        target_compile_options(SPH PRIVATE $<$<CONFIG:${_cfg}>:/Zi>)
        target_link_options(SPH PRIVATE $<$<CONFIG:${_cfg}>:/DEBUG>)
    endforeach()

    # 调试设置全部用 MSBuild 宏，不写死绝对路径：
    #   VS_DEBUGGER_COMMAND        = $(TargetPath)，即“生成出来的那个 exe”，
    #                                保证 VS 里 F5 直接能启动（即使 OutDir 被改写成相对形式）
    #   VS_DEBUGGER_WORKING_DIRECTORY = 解决方案目录的上一级 = 工程根目录，
    #                                程序用相对路径读 input.dat，必须在这里跑
    set_target_properties(SPH PROPERTIES
        VS_DEBUGGER_COMMAND "$(TargetPath)"
        VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif()

# ---------------------------------------------------------------------------
# 输出目录保持 CMake 默认（<构建目录>/<Config>），不写绝对路径
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 配置摘要：一律用相对路径显示
# ---------------------------------------------------------------------------
file(RELATIVE_PATH SPH_REL_BUILD "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")
if(SPH_REL_BUILD STREQUAL "")
    set(SPH_REL_BUILD ".")
endif()
file(TO_CMAKE_PATH "${SPH_REL_BUILD}" SPH_REL_BUILD)

message(STATUS "")
message(STATUS "====================== 工程路径（相对）======================")
message(STATUS " 工程根目录 : .")
message(STATUS " 构建脚本   : CMakeLists.txt")
message(STATUS " 源码       : 1.cpp")
message(STATUS " 构建目录   : ${SPH_REL_BUILD}")
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    message(STATUS " 解决方案   : ${SPH_REL_BUILD}/${PROJECT_NAME}.sln")
endif()
message(STATUS " 可执行文件 : ${SPH_REL_BUILD}/<Config>/${PROJECT_NAME}.exe")
message(STATUS " 运行数据   : input.dat / result<N>.dat  （工作目录 = 工程根目录）")
message(STATUS "============================================================")
message(STATUS " 生成器 : ${CMAKE_GENERATOR} 平台=${CMAKE_VS_PLATFORM_NAME} 工具集=${CMAKE_VS_PLATFORM_TOOLSET}")
if(SPH_USE_CXX20)
    message(STATUS " C++ 标准: C++20")
else()
    message(STATUS " C++ 标准: C++17")
endif()
if(SPH_SUPPRESS_REGENERATION)
    message(STATUS " ZERO_CHECK: 已关闭（改 CMakeLists 后需重跑 cmake）")
endif()
message(STATUS "============================================================")
