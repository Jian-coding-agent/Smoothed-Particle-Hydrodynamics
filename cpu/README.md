# SPH（三维 SPH 流固耦合求解器）— CMake 工程

把原来的单文件手工工程 [`1.cpp`](1.cpp) + [`Project1.vcxproj`](Project1.vcxproj) 整理为 CMake 工程，
生成 **Visual Studio 2019（v142 工具集、C++17）** 可直接打开的 `build/SPH.sln`。

**本工程约定：配置与脚本里不出现任何绝对路径，全部相对工程目录 / 构建目录。**

## 1. 环境要求

| 项目 | 版本 |
|---|---|
| Visual Studio | 2019（16.x，需「使用 C++ 的桌面开发」与 v142 工具集） |
| CMake | ≥ 3.20（VS2019 16.10+ 自带的 CMake 已满足） |
| OpenMP | MSVC 自带 `/openmp`，无需额外安装 |

> 本机已验证：VS2019 Community 16.11（v142）+ CMake 3.24；
> 且 `cmake ..`（不带 `-G`）在这台机器上默认就选中 **Visual Studio 16 2019 / x64 / v142**。

## 2. 快速开始

### 方式 A：标准 CMake 流程（推荐）

```bat
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

产物：`build\Release\SPH.exe`。

> 注意：CMake 的 VS 生成器会附带一个 `ALL_BUILD` 辅助工程。想让它从
> 解决方案里消失、并把工程文件里的绝对路径改成相对路径，跑一次：

```bat
powershell -ExecutionPolicy Bypass -File ..\tools\finalize-generated.ps1 -Sln SPH.sln
cmake --build . --config Release --target SPH
```

（`build.bat` 会自动完成这一步，见方式 B。）

### 方式 B：一键脚本

```bat
build.bat                       :: 默认同时编译 Release + Debug
build.bat -Configuration Debug  :: 只编译 Debug
build.bat -Cxx20                :: 改用 C++20（需 VS2019 16.11+）
```

`build.bat` → `build.ps1`，内部等价于
`cmake -S . -B build -G "Visual Studio 16 2019" -A x64 -T v142`
→ `tools\finalize-generated.ps1` → `cmake --build build --config <Cfg> --target SPH`。

> 默认两个配置都编译，是为了在 VS 里随便切 `Debug`/`Release` 都有 exe 可启动。

> 有些机器禁止运行未签名的 `.ps1`（报 "not digitally signed"），所以用 `build.bat`
> 作为入口，它以 `-ExecutionPolicy Bypass` 调用，不需要改系统执行策略。

### 方式 C：CMake 预设

```powershell
cmake --preset vs2019-x64
cmake --build --preset release
```

### 方式 D：Visual Studio 界面

用 VS2019 打开 `build\SPH.sln`，选 `x64 | Release` → **直接 F5 就能跑**（工程里已写好
`LocalDebuggerCommand = $(TargetPath)`，工作目录 = 工程根目录）。

## 3. 关于“全相对路径”

| 位置 | 是否含绝对路径 | 说明 |
|---|---|---|
| `CMakeLists.txt` | 否 | 只用 `1.cpp`、`CMAKE_*` 这类相对/内置变量 |
| `build.bat` / `build.ps1` / `tools/*.ps1` | 否 | 一律相对工程根目录，可整目录搬到任意路径 |
| 生成的 `SPH.sln` | 否 | 只含工程名与相对路径 `SPH.vcxproj` |
| 生成的 `SPH.vcxproj` / `.filters` | 经 `finalize-generated.ps1` 后为否 | `OutDir=$(ProjectDir)Release\`、`ClCompile Include="..\1.cpp"`、`$(SolutionDir)..`、`$(TargetPath)` |
| `build/CMakeCache.txt`、`build/CMakeFiles/*` | **是** | CMake 自身的构建状态，必然写绝对路径；**构建目录不可跨机器/跨路径搬运**（见 §3.1），换机器直接删掉 `build/` 重新 `build.bat` |

### 3.1 构建目录 `build\` 不能跨机器拷贝 ⚠

`build\` 里记录的是**生成它的那台机器**的信息：源码/构建目录的绝对路径、`cl.exe` 的确切路径
（含工具集补丁号，如 `14.29.30133`）、`WindowsTargetPlatformVersion`（具体 SDK 版本号）。
把它拷到另一台机器（或另一个路径）再编译，编译器就会按那台机器的路径去找头文件 →
典型报错就是 `C1083: 无法打开包括文件: 'ctype.h'`。

- **正确做法**：只复制源码（`1.cpp`、`CMakeLists.txt`、`build.bat`、`build.ps1`、`tools\*`、`input.dat`），
  到目标机重新跑 `build.bat`。
- **已经拷了**：直接删掉 `build\` 再 `build.bat`。
- **新版 `build.bat` 会自动处理**：它比对 `CMakeCache.txt` 里的源码目录、构建目录、`cl.exe` 路径、
  SDK 版本，发现和本机不符就打印原因、删掉 `build\` 并重新配置（若被 VS 占用会提示先关 VS）。

**也能手工在 VS 里救急**（不想删 `build\` 时）：

1. 右键项目 `SPH` → 属性 → 配置属性 → 常规 → **Windows SDK 版本** → 选本机已装的（或"最新安装的"）；
2. 若提示的是工具集问题，再把 **平台工具集** 选成本机有的（如 `Visual Studio 2019 (v142)`）；
3. 或者直接 右键解决方案 → **重定目标解决方案（Retarget Solution）**，勾选本机 SDK 后确定。

改完 `重新生成` 即可。但这只是临时补丁：这些值写在 CMake 生成的 `SPH.vcxproj` 里，
下次跑 `cmake ..` / `build.bat` 会被重新写回。**根上解决**：本工程已把 SDK 版本改成 `10.0`
（= 用本机最新安装的 SDK，见下面 `finalize-generated.ps1` 的第 3 步），跨机器不再因为 SDK 版本号报 `ctype.h`。

`finalize-generated.ps1` 做四件事（幂等）：

1. 从 `.sln` 摘除 `ALL_BUILD` / `ZERO_CHECK` 工程，并删除其 `.vcxproj` 与中间目录；
2. 把 `SPH.vcxproj` / `.filters` / `.user` 里的绝对路径改写成：
   - 项路径（`Include="..\1.cpp"`）→ 相对路径；
   - 属性值（`OutDir` / `ImportLibrary` / `ProgramDataBaseFile`）→ `$(ProjectDir)…`，
     **不能写成裸 `Debug\`**，否则 `$(TargetPath)` 会停留在相对形式，VS 调试器就会
     报“找不到 exe / 系统找不到指定的文件”；
3. 把 `<WindowsTargetPlatformVersion>` 从具体版本号（如 `10.0.26100.0`）改成 **`10.0`**
   （= 本机最新安装的 SDK），换机器不再因 SDK 版本号不匹配而报 `ctype.h`；
4. 清掉 `build\.vs`、`*.suo` 等 VS 用户状态，避免它缓存旧的启动项目（例如 ALL_BUILD）。

> 注意：**每次重新 `cmake ..` 都会把生成文件写回绝对路径、写回具体 SDK 版本、并带回 ALL_BUILD**，
> 需要再跑一次 `finalize-generated.ps1`（`build.bat` 已自动做）。

## 4. 关于 ALL_BUILD / ZERO_CHECK

| 辅助工程 | 作用 | 处理方式 |
|---|---|---|
| `ZERO_CHECK` | `CMakeLists.txt` 变动后自动重新配置 | `SPH_SUPPRESS_REGENERATION=ON`（默认）彻底不生成 |
| `ALL_BUILD` | “生成解决方案”的聚合工程 | CMake 的 VS 生成器**强制**生成、无官方开关；只能由 `finalize-generated.ps1` 生成后摘除 |

**这就是“F5 报找不到 exe”的根源**：只要 `.sln` 里还有 `ALL_BUILD`（就是裸 `cmake ..`
刚生成完的状态），VS 打开后默认启动项目会是它——它是个 Utility 工程、没有 exe，
F5 立刻报“找不到 exe”。摘除后解决方案只剩 `SPH`，F5 直接用 `SPH` 启动。

所以请用一条命令生成：

```bat
build.bat
```

（等价于 `cmake ..` + `finalize-generated.ps1` + 编译）。如果你坚持只跑 `cmake ..`，
那么之后必须补一条：

```bat
cd build
powershell -ExecutionPolicy Bypass -File ..\tools\finalize-generated.ps1 -Sln SPH.sln
```

代价：

1. 关掉 `ZERO_CHECK` 后，改 `CMakeLists.txt` **不会自动生效**，需重跑 `cmake ..`（或 `build.bat`）。
2. 删掉 `ALL_BUILD` 后，命令行构建要写 `--target SPH`（`cmake --build build --config Release --target SPH`）；
   VS 里 F7 / F5 不受影响。

## 5. 生成物

| 路径 | 说明 |
|---|---|
| `build\SPH.sln` | VS2019 解决方案（清理后只含 `SPH` 一个工程） |
| `build\Release\SPH.exe` | Release 可执行文件 |
| `build\Debug\SPH.exe` | Debug 可执行文件（与 Release 分目录，互不覆盖） |
| `build\<Config>\SPH.pdb` | 调试符号 |

## 6. 关键配置项

| CMake 选项 | 默认 | 说明 |
|---|---|---|
| `SPH_USE_CXX20` | `OFF` | 打开后 `/std:c++20`。默认 `C++17`：代码不依赖 C++20，v142 对 C++17 支持最完整 |
| `SPH_GBK_SOURCE` | `OFF` | `1.cpp` 注释是 **GBK(cp936)**。中文系统默认代码页即可；非中文区域出现 `C4819` 时打开，强制 `/source-charset:.936` |
| `SPH_ENABLE_FAST_MATH` | `OFF` | `/fp:fast` 会改变浮点结果，与历史算例对比时**不要**开 |
| `SPH_SUPPRESS_REGENERATION` | `ON` | 不生成 `ZERO_CHECK`；代价是改 `CMakeLists.txt` 后要手工重跑 `cmake` |

### 与原 vcxproj 的对应关系

| 原工程（VS2026） | 本 CMake 工程（VS2019） |
|---|---|
| `PlatformToolset = v145` | `-T v142` |
| `LanguageStandard = stdcpp20` | `CXX_STANDARD 17`（可切 20） |
| `OpenMPSupport = true` | `find_package(OpenMP)` + `OpenMP::OpenMP_CXX` |
| `_CRT_SECURE_NO_WARNINGS` | 同名宏 |
| `ConformanceMode = true` | CMake 对 MSVC 默认加 `/permissive-`，等价 |
| `WholeProgramOptimization`（Release） | 默认不加 `/GL`（保证数值可复现） |
| runtime 库 | 保持 MSVC 默认：Release `/MD`、Debug `/MDd` |

## 7. 运行时注意事项

1. **工作目录**：程序用相对路径读 `input.dat`、写 `result<N>.dat`。
   VS 调试工作目录已设为 `$(SolutionDir)..`（相对宏，无绝对路径），**F5 直接可跑**；
   命令行请从工程根目录启动：
   ```powershell
   .\build\Release\SPH.exe
   ```
2. **输出文件**：每 50 步写一个约 20 MB 的 `result<N>.dat`（编号 `itime/50`），
   写在工作目录里。多次运行会**互相覆盖**，建议每次运行换独立目录。
3. **环境变量**：
   | 变量 | 作用 | 默认 |
   |---|---|---|
   | `SPH_THREADS` | OpenMP 线程数（上限 32） | 自动 |
   | `SPH_MAXSTEP` | 时间步上限 | `10000001` |

   ```powershell
   $env:SPH_THREADS = 8
   $env:SPH_MAXSTEP = 1000
   ```
4. **内存**：每线程累加缓冲约 66 MB，8 线程约 530 MB，16 线程约 1.06 GB。

## 8. 常见错误排查（换机器时最常踩）

`build.bat` 第一步 `[0/3] environment` 会打印本机实际环境，先看它：

```
[0/3] environment
      cmake : E:\Cmake\bin\cmake.exe
      Visual Studio 2019 Community  toolsets: 14.29.30133
      Windows SDK (ucrt) : 10.0.19041.0, 10.0.26100.0
```

| 现象 | 原因 | 处理 |
|---|---|---|
| `C1083: 无法打开包括文件: 'ctype.h'` | ① 本机真缺 Windows SDK（见 §10 前的环境自检）；② **把别处生成的 `build\` 目录一起拷过来了**（见 §3.1） | ① 装 Windows 10/11 SDK；② 删掉 `build\` 重跑 `build.bat`（新版会自动识别并清理外来缓存） |
| `MSB8020: 找不到 v142 生成工具` | 没装 v142 工具集 | VS Installer → 单个组件 → 装 `MSVC v142 - VS 2019 C++ x64/x86 生成工具`；或 `build.bat -Generator "Visual Studio 17 2022" -Toolset v143` |
| CMake 提示找不到 Visual Studio 实例 | 没装 VS 或没装 C++ 工作负载 | 安装 VS（含 C++ 桌面开发），或显式 `-Generator` |
| 解决方案里又出现 `ALL_BUILD`，F5 报找不到 exe | 用裸 `cmake ..` 生成，CMake 把辅助工程加回来了 | 重跑 `build.bat`，或补：`powershell -ExecutionPolicy Bypass -File tools\finalize-generated.ps1 -Sln build\SPH.sln` |
| **Debug 一启动就退出（退出码 3）**、或弹 `Debug Assertion Failed` | 原 `1.cpp` 里两个只在 Debug 下暴露的缺陷（见 §10） | **已修**，用本包即可；`build.bat` 现在默认同时编译 Release + Debug，切配置总有 exe |
| **Release 状态栏一直"正在加载符号…"** | VS 在给没有符号的 exe/DLL 找符号，离线/内网访问符号服务器会长时间卡住 | Release 现已生成 PDB；仍卡就在 工具→选项→调试→符号 里全部取消符号服务器，或直接用 `Ctrl+F5`／双击 `build\Release\SPH.exe`（长时间仿真本来就不需要调试器） |

> `build.bat` 在新建构建目录时会依次尝试 **VS2019/v142 → VS2022/v143 → 两者的默认工具集**，
> 所以目标机只有 VS2022 也能编译（前提是装了 C++ 工作负载 + Windows SDK）。

## 10. 相对原始 1.cpp 的修复（本次改动）

原 `1.cpp` 在 Release 下"看起来正常"，但有两处未定义行为，在 Debug 下必然暴露：

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | `pair_i[k]` / `pair_j[k]`（6 处：`continue_density` 两个 pass、`GraCorrec`、`GLStrain`、`NorVector`） | `direct_find()` 是**直接往 `vector` 的预留空间里写**（`data()` + 容量当长度用），`size()` 始终是 0。Release 的 `operator[]` 不检查所以"能用"；Debug 的 STL 会断言越界 → `abort()`（退出码 3） | 改成 `pair_i.data()[k]` / `pair_j.data()[k]`（返回裸指针，不检查、行为与 Release 一致） |
| 2 | 4 个归约循环的 `#pragma omp parallel for schedule(static) num_threads(nth)` | 循环体里 `i = ii;` 写的是**全局变量 `i`**（多线程共享），随后用 `i` 去读/写 `F11[i]` 等 —— 典型数据竞争。Debug 下常常把形变梯度归约成 0 → `Jacb=0` → `log10(0)`、`0/0` → 输出出现 `-nan(ind)`（实测 47943 行） | pragma 加 `firstprivate(i)`，让每个线程用自己的 `i` |

修复前后对比（本机实测，1 步）：

| | 修复前 | 修复后 |
|---|---|---|
| Debug 运行 | 退出码 3，无任何输出 | exit 0，27 s，正常输出 |
| Debug 结果里的 `nan` | **47943 行** | **0 行** |
| Debug vs Release | 不可比（Debug 是垃圾值） | 差异仅 4142/3,322,172 个值（0.13%），且全是打印精度末位（p 最大差 2.5e-08，vx 1e-13） |

> 建议用本包重跑一次正式算例，与历史结果比对一遍（预期：输出精度内一致）。备份文件：`_backup_1.cpp.before_debugfix`。

## 11. 文件说明

| 文件 | 作用 |
|---|---|
| `1.cpp` | 全部求解器源码（单一翻译单元，未改动） |
| `CMakeLists.txt` | 构建脚本（无绝对路径；禁止源内构建） |
| `CMakePresets.json` | VS2019 预设 |
| `build.bat` | 一键入口（绕过脚本执行策略）→ `build.ps1` |
| `build.ps1` | 配置 + 清理/相对化 + 编译，并打印相对路径 |
| `tools\finalize-generated.ps1` | 摘除 `ALL_BUILD`/`ZERO_CHECK`，把工程文件路径改相对 |
| `tools\CompareResults.cs/.exe` | 逐点比较两套 `result*.dat` 的小工具 |
| `input.dat` | 输入（237298 个粒子 × 14 列） |
| `Project1.vcxproj*` | 旧的手工工程，保留备查，不参与 CMake 构建 |

> `build.ps1` 与 `tools\*.ps1` 刻意保持 **纯 ASCII + CRLF**：Windows PowerShell 5.1
> 对没有 BOM 的 UTF-8 脚本按 ANSI 解析，带中文会直接语法报错；`.bat` 也必须 CRLF。
