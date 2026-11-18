# SPH (3D SPH fluid-structure interaction solver) — CMake project

This wraps the original hand-written single-file project [`1.cpp`](1.cpp)
in a CMake project that generates a `build/SPH.sln` ready to open in
**Visual Studio 2019 (v142 toolset, C++17)**.

**Project convention: no absolute paths in the configuration or scripts;
everything is relative to the project directory / build directory.**

## 1. Requirements

| Item | Version |
|---|---|
| Visual Studio | 2019 (16.x, with "Desktop development with C++" and the v142 toolset) |
| CMake | >= 3.20 (the CMake bundled with VS2019 16.10+ is sufficient) |
| OpenMP | provided by MSVC's `/openmp`; no separate installation |

> Verified locally: VS2019 Community 16.11 (v142) + CMake 3.24;
> and `cmake ..` (without `-G`) selects **Visual Studio 16 2019 / x64 / v142**
> by default on that machine.

## 2. Quick start

### Option A: standard CMake flow (recommended)

```bat
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Output: `build\Release\SPH.exe`.

> Note: the CMake VS generator adds an `ALL_BUILD` helper project. To remove it
> from the solution and rewrite the absolute paths in the project files as
> relative paths, run:

```bat
powershell -ExecutionPolicy Bypass -File ..\tools\finalize-generated.ps1 -Sln SPH.sln
cmake --build . --config Release --target SPH
```

(`build.bat` does this automatically; see Option B.)

### Option B: one-click script

```bat
build.bat                       :: build Release + Debug by default
build.bat -Configuration Debug  :: build Debug only
build.bat -Cxx20                :: use C++20 (requires VS2019 16.11+)
```

`build.bat` -> `build.ps1`, which is equivalent to
`cmake -S . -B build -G "Visual Studio 16 2019" -A x64 -T v142`
-> `tools\finalize-generated.ps1` -> `cmake --build build --config <Cfg> --target SPH`.

> Both configurations are built by default so that switching between
> `Debug` and `Release` in VS always finds an exe to launch.

> Some machines refuse to run unsigned `.ps1` files ("not digitally signed"),
> so `build.bat` is the entry point; it calls PowerShell with
> `-ExecutionPolicy Bypass` and does not require changing the system policy.

### Option C: CMake presets

```powershell
cmake --preset vs2019-x64
cmake --build --preset release
```

### Option D: Visual Studio UI

Open `build\SPH.sln` in VS2019, select `x64 | Release`, then press **F5** (the
project already sets `LocalDebuggerCommand = $(TargetPath)` and the working
directory to the project root).

## 3. About "all relative paths"

| Location | Contains absolute paths? | Notes |
|---|---|---|
| `CMakeLists.txt` | No | Uses only relative/built-in variables such as `1.cpp` and `CMAKE_*` |
| `build.bat` / `build.ps1` / `tools/*.ps1` | No | Always relative to the project root; the whole directory can be moved anywhere |
| Generated `SPH.sln` | No | Contains only the project name and the relative path `SPH.vcxproj` |
| Generated `SPH.vcxproj` / `.filters` | No after `finalize-generated.ps1` | `OutDir=$(ProjectDir)Release\`, `ClCompile Include="..\1.cpp"`, `$(SolutionDir)..`, `$(TargetPath)` |
| `build/CMakeCache.txt`, `build/CMakeFiles/*` | **Yes** | CMake's own build state necessarily contains absolute paths; **the build directory cannot be moved across machines or paths** (see §3.1). On a new machine, delete `build/` and run `build.bat` again |

### 3.1 The `build\` directory cannot be copied between machines ⚠

`build\` records information about **the machine that generated it**: the
absolute paths of the source and build directories, the exact path of `cl.exe`
(including the toolset patch number, e.g. `14.29.30133`), and
`WindowsTargetPlatformVersion` (a specific SDK version).
If it is copied to another machine (or another path) and built there, the
compiler looks for headers under the old machine's paths — the typical error is
`C1083: cannot open include file: 'ctype.h'`.

- **Correct approach:** copy only the sources (`1.cpp`, `CMakeLists.txt`,
  `build.bat`, `build.ps1`, `tools\*`, `input.dat`) and run `build.bat` on the
  target machine.
- **Already copied it:** delete `build\` and run `build.bat` again.
- **The new `build.bat` handles this automatically:** it compares the source
  directory, build directory, `cl.exe` path and SDK version in `CMakeCache.txt`
  with the local machine; on a mismatch it prints the reason, deletes `build\`
  and reconfigures (if VS has the directory open, it asks you to close VS first).

**You can also rescue it manually in VS** (without deleting `build\`):

1. Right-click the `SPH` project -> Properties -> Configuration Properties ->
   General -> **Windows SDK Version** -> choose one installed locally (or
   "latest installed").
2. If the error mentions the toolset, set **Platform Toolset** to one installed
   locally (e.g. `Visual Studio 2019 (v142)`).
3. Or right-click the solution -> **Retarget Solution**, select the local SDK
   and confirm.

Then rebuild. This is only a temporary patch: these values live in the
CMake-generated `SPH.vcxproj` and are overwritten the next time `cmake ..` /
`build.bat` runs. **The real fix:** this project sets the SDK version to `10.0`
(= use the latest SDK installed locally, see step 3 of `finalize-generated.ps1`
below), so moving between machines no longer produces a `ctype.h` error because
of a version mismatch.

`finalize-generated.ps1` does four things (idempotently):

1. Removes the `ALL_BUILD` / `ZERO_CHECK` projects from the `.sln` and deletes
   their `.vcxproj` files and intermediate directories.
2. Rewrites absolute paths in `SPH.vcxproj` / `.filters` / `.user`:
   - item paths (`Include="..\1.cpp"`) -> relative paths;
   - property values (`OutDir` / `ImportLibrary` / `ProgramDataBaseFile`) ->
     `$(ProjectDir)…`. They **must not** be bare `Debug\`, otherwise
     `$(TargetPath)` stays relative and the VS debugger reports
     "cannot find the exe / the system cannot find the file specified".
3. Changes `<WindowsTargetPlatformVersion>` from a specific version (e.g.
   `10.0.26100.0`) to **`10.0`** (= the latest SDK installed locally), so a
   different machine no longer fails with `ctype.h` due to an SDK mismatch.
4. Clears `build\.vs`, `*.suo` and other VS user state so it does not cache the
   old startup project (for example `ALL_BUILD`).

> Note: **every re-run of `cmake ..` writes absolute paths, a specific SDK
> version and `ALL_BUILD` back into the generated files**, so
> `finalize-generated.ps1` must run again (`build.bat` does this for you).

## 4. About ALL_BUILD / ZERO_CHECK

| Helper project | Purpose | Handling |
|---|---|---|
| `ZERO_CHECK` | Reconfigures automatically when `CMakeLists.txt` changes | `SPH_SUPPRESS_REGENERATION=ON` (default) prevents it from being generated at all |
| `ALL_BUILD` | Aggregate project for "Build Solution" | The CMake VS generator **always** creates it and has no official switch; it can only be removed after generation by `finalize-generated.ps1` |

**This is the root cause of "F5 cannot find the exe":** as long as `ALL_BUILD`
is still in the `.sln` (which is the state right after a bare `cmake ..`), VS
makes it the default startup project. It is a Utility project with no exe, so
F5 immediately reports that it cannot find the exe. Once it is removed, `SPH`
is the only project left and F5 launches `SPH` directly.

So generate with one command:

```bat
build.bat
```

(equivalent to `cmake ..` + `finalize-generated.ps1` + build). If you insist on
running only `cmake ..`, you must follow it with:

```bat
cd build
powershell -ExecutionPolicy Bypass -File ..\tools\finalize-generated.ps1 -Sln SPH.sln
```

Costs:

1. With `ZERO_CHECK` disabled, changes to `CMakeLists.txt` **do not take effect
   automatically**; re-run `cmake ..` (or `build.bat`).
2. With `ALL_BUILD` removed, command-line builds must pass `--target SPH`
   (`cmake --build build --config Release --target SPH`); F7 / F5 in VS are
   unaffected.

## 5. Outputs

| Path | Description |
|---|---|
| `build\SPH.sln` | VS2019 solution (after cleanup it contains only the `SPH` project) |
| `build\Release\SPH.exe` | Release executable |
| `build\Debug\SPH.exe` | Debug executable (separate directory, does not overwrite Release) |
| `build\<Config>\SPH.pdb` | Debug symbols |

## 6. Key configuration options

| CMake option | Default | Description |
|---|---|---|
| `SPH_USE_CXX20` | `OFF` | Enables `/std:c++20`. The default is `C++17`: the code does not require C++20, and v142 has the most complete C++17 support |
| `SPH_GBK_SOURCE` | `OFF` | The source is now ASCII; this option is only needed for old GBK checkouts. It forces `/source-charset:.936` when `C4819` appears in a non-Chinese locale |
| `SPH_ENABLE_FAST_MATH` | `OFF` | `/fp:fast` changes floating-point results; **do not** enable it when comparing against historical cases |
| `SPH_SUPPRESS_REGENERATION` | `ON` | Does not generate `ZERO_CHECK`; the price is that `CMakeLists.txt` changes require a manual `cmake` re-run |

### Mapping from the original vcxproj

| Original project (VS2026) | This CMake project (VS2019) |
|---|---|
| `PlatformToolset = v145` | `-T v142` |
| `LanguageStandard = stdcpp20` | `CXX_STANDARD 17` (can switch to 20) |
| `OpenMPSupport = true` | `find_package(OpenMP)` + `OpenMP::OpenMP_CXX` |
| `_CRT_SECURE_NO_WARNINGS` | same macro |
| `ConformanceMode = true` | CMake adds `/permissive-` for MSVC by default; equivalent |
| `WholeProgramOptimization` (Release) | `/GL` is not added, so numerical results stay reproducible |
| runtime library | MSVC defaults: Release `/MD`, Debug `/MDd` |

## 7. Runtime notes

1. **Working directory:** the program reads `input.dat` and writes
   `result<N>.dat` through relative paths. The VS debug working directory is set
   to `$(SolutionDir)..` (a relative macro, no absolute path), so **F5 just
   works**. From the command line, start it from the project root:
   ```powershell
   .\build\Release\SPH.exe
   ```
2. **Output files:** every 50 steps it writes a ~20 MB `result<N>.dat` (numbered
   `itime/50`) in the working directory. Repeated runs **overwrite each other**;
   use a separate directory per run.
3. **Environment variables:**
   | Variable | Effect | Default |
   |---|---|---|
   | `SPH_THREADS` | number of OpenMP threads (max 32) | automatic |
   | `SPH_MAXSTEP` | time-step limit | `10000001` |

   ```powershell
   $env:SPH_THREADS = 8
   $env:SPH_MAXSTEP = 1000
   ```
4. **Memory:** each thread's accumulation buffer is about 66 MB; 8 threads
   about 530 MB, 16 threads about 1.06 GB.

## 8. Troubleshooting (most common when changing machines)

The first step of `build.bat`, `[0/3] environment`, prints the actual local
environment; look at that first:

```
[0/3] environment
      cmake : E:\Cmake\bin\cmake.exe
      Visual Studio 2019 Community  toolsets: 14.29.30133
      Windows SDK (ucrt) : 10.0.19041.0, 10.0.26100.0
```

| Symptom | Cause | Fix |
|---|---|---|
| `C1083: cannot open include file: 'ctype.h'` | (1) Windows SDK genuinely missing locally (see the environment check before §10); (2) **a `build\` directory generated elsewhere was copied in** (see §3.1) | (1) install the Windows 10/11 SDK; (2) delete `build\` and re-run `build.bat` (the new version detects and cleans foreign caches) |
| `MSB8020: cannot find the v142 build tools` | the v142 toolset is not installed | VS Installer -> Individual components -> install `MSVC v142 - VS 2019 C++ x64/x86 build tools`; or `build.bat -Generator "Visual Studio 17 2022" -Toolset v143` |
| CMake cannot find a Visual Studio instance | VS is not installed, or the C++ workload is missing | install VS (with Desktop development with C++), or pass `-Generator` explicitly |
| `ALL_BUILD` is back in the solution and F5 cannot find the exe | it was generated with a bare `cmake ..`, which adds the helper projects back | re-run `build.bat`, or run `powershell -ExecutionPolicy Bypass -File tools\finalize-generated.ps1 -Sln build\SPH.sln` |
| **Debug exits immediately (exit code 3)**, or a `Debug Assertion Failed` dialog appears | two defects in the original `1.cpp` that only surface in Debug (see §10) | **already fixed** in this package; `build.bat` now builds Release + Debug by default, so every configuration has an exe |
| **Release stays at "Loading symbols..."** | VS is looking for symbols for an exe/DLL that has none; offline/intranet symbol-server access can hang for a long time | Release now generates a PDB; if it still hangs, clear all symbol servers under Tools -> Options -> Debugging -> Symbols, or use `Ctrl+F5` / double-click `build\Release\SPH.exe` (long simulations do not need a debugger anyway) |

> When creating a new build directory, `build.bat` tries
> **VS2019/v142 -> VS2022/v143 -> the default toolset of either**, so a target
> machine that only has VS2022 can still build (provided the C++ workload and
> the Windows SDK are installed).

## 10. Fixes relative to the original 1.cpp (this change)

The original `1.cpp` "looked fine" in Release but had two undefined behaviours
that are guaranteed to surface in Debug:

| # | Location | Problem | Fix |
|---|---|---|---|
| 1 | `pair_i[k]` / `pair_j[k]` (6 places: the two passes of `continue_density`, `GraCorrec`, `GLStrain`, `NorVector`) | `direct_find()` writes **directly into a vector's reserved storage** (uses `data()` and treats capacity as length), so `size()` is always 0. Release's `operator[]` does not check and therefore "works"; the Debug STL asserts an out-of-range access -> `abort()` (exit code 3) | changed to `pair_i.data()[k]` / `pair_j.data()[k]` (returns a raw pointer, no check, same behaviour as Release) |
| 2 | `#pragma omp parallel for schedule(static) num_threads(nth)` on the four reduction loops | the loop body assigns `i = ii;` to the **global `i`** (shared by all threads) and then uses `i` to read/write `F11[i]` etc. — a classic data race. In Debug the deformation gradient is often reduced to 0 -> `Jacb=0` -> `log10(0)`, `0/0` -> `-nan(ind)` in the output (47943 lines measured) | added `firstprivate(i)` to the pragma so each thread has its own `i` |

Before/after comparison (measured locally, 1 step):

| | Before the fix | After the fix |
|---|---|---|
| Debug run | exit code 3, no output at all | exit 0, 27 s, normal output |
| `nan` in Debug results | **47943 lines** | **0 lines** |
| Debug vs Release | not comparable (Debug is garbage) | only 4142 / 3,322,172 values differ (0.13%), all in the last printed digit (p max 2.5e-08, vx 1e-13) |

> It is worth re-running a production case with this package and comparing it
> with the historical results (expected: identical to printed precision).

## 11. File overview

| File | Purpose |
|---|---|
| `1.cpp` | the complete solver source (single translation unit; comments are English, code unchanged) |
| `CMakeLists.txt` | build script (no absolute paths; in-source builds forbidden) |
| `CMakePresets.json` | VS2019 presets |
| `build.bat` | one-click entry point (bypasses the script execution policy) -> `build.ps1` |
| `build.ps1` | configure + cleanup/relativise + build, printing relative paths |
| `tools\finalize-generated.ps1` | removes `ALL_BUILD`/`ZERO_CHECK` and makes project paths relative |
| `tools\CompareResults.cs/.exe` | small utility that compares two sets of `result*.dat` point by point |
| `input.dat` | input (237298 particles x 14 columns) |

> `build.ps1` and `tools\*.ps1` are deliberately **pure ASCII + CRLF**: Windows
> PowerShell 5.1 parses UTF-8 scripts without a BOM as ANSI, and non-ASCII
> characters cause syntax errors; `.bat` files must also be CRLF.
