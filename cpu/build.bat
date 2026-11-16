@echo off
rem ===========================================================================
rem  One-click: generate the Visual Studio 2019 solution and build.
rem  This is just a wrapper around build.ps1 (needed because some machines
rem  block running unsigned .ps1 files directly).
rem
rem  Usage:
rem    build.bat                    -^> Release
rem    build.bat -Configuration Debug
rem    build.bat -Cxx20
rem ===========================================================================
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
