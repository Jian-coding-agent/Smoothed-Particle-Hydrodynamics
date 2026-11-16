@echo off
rem ===========================================================================
rem  SPH CUDA -- one-click build
rem
rem  * locates Visual Studio (vswhere), sets up the MSVC x64 environment
rem  * uses CMake + the Ninja that ships with Visual Studio
rem    (CMake on this machine has no "Visual Studio 18 2026" generator)
rem  * no absolute paths in this file: everything is relative to %~dp0
rem
rem  usage:  build.bat            Release (default)
rem          build.bat Debug
rem ===========================================================================
setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
set "CFG=%~1"
if "%CFG%"=="" set "CFG=Release"

echo [1/4] locate Visual Studio
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo     ERROR: vswhere.exe not found - is Visual Studio installed?
    exit /b 1
)
set "VSINSTALL="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    echo     ERROR: no Visual Studio with the C++ workload found
    exit /b 1
)
echo     %VSINSTALL%

echo [2/4] MSVC x64 environment
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo     ERROR: vcvars64.bat failed
    exit /b 1
)

echo [3/4] locate ninja
set "NINJA="
if exist "%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" (
    set "NINJA=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
) else (
    for /f "delims=" %%i in ('where ninja 2^>nul') do set "NINJA=%%i"
)
if not defined NINJA (
    echo     ERROR: ninja.exe not found ^(install Ninja or CMake^)
    exit /b 1
)
echo     %NINJA%

echo [4/4] configure + build (%CFG%)
cd /d "%ROOT%"
if exist "build\CMakeCache.txt" (
    rem CMake stores CMAKE_MAKE_PROGRAM with forward slashes; search both
    rem spellings, otherwise this test never matches and every run wipes the tree
    set "NINJA_FWD=%NINJA:\=/%"
    findstr /i /c:"CMAKE_MAKE_PROGRAM:FILEPATH=%NINJA%" /c:"CMAKE_MAKE_PROGRAM:FILEPATH=%NINJA_FWD%" "build\CMakeCache.txt" >nul 2>&1
    if errorlevel 1 (
        echo     build tree was configured with another toolchain - wiping it
        rmdir /s /q build
    )
)
cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=%CFG%
if errorlevel 1 exit /b 1
cmake --build build
if errorlevel 1 exit /b 1

echo.
echo done:  %ROOT%build\sph_cuda.exe
echo run :  copy input.dat next to the exe, then  set SPH_MAXSTEP=500 ^&^& sph_cuda.exe [--fp32^|--fp64]
endlocal
