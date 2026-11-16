<#
.SYNOPSIS
    Configure and build SPH using only paths relative to the project root,
    with an environment pre-flight check and toolchain auto-fallback.

.DESCRIPTION
    Steps:
      [0/3] print what the machine actually has: cmake, Visual Studio
            toolchains, Windows SDKs (and whether ucrt\ctype.h exists)
      [1/3] configure (reuses an existing CMakeCache.txt; otherwise tries
            VS2019/v142, then VS2022/v143, then the default toolsets)
      [2/3] drop ALL_BUILD / ZERO_CHECK and make project paths relative
      [3/3] build --target SPH --config <Configuration>

    The ALL_BUILD project cannot be disabled in CMake's VS generator, so it
    is removed after generation; that is why the build step names the target.

.NOTES
    ASCII-only on purpose: Windows PowerShell 5.1 reads .ps1 files as the
    system ANSI code page unless a UTF-8 BOM is present.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -Configuration Debug
    .\build.ps1 -Cxx20
    .\build.ps1 -Generator 'Visual Studio 17 2022' -Toolset v143
#>
[CmdletBinding()]
param(
    # 'All' builds both Release and Debug, so whichever configuration you pick
    # in Visual Studio there is always an executable to launch.
    [ValidateSet('All', 'Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')]
    [string]$Configuration = 'All',

    [switch]$Cxx20,

    [string]$BuildDir = 'build',

    # Toolchain: default is the VS2019 (v142) setup this project targets.
    # Leave them alone to let the script fall back automatically.
    [string]$Generator = 'Visual Studio 16 2019',
    [string]$Arch = 'x64',
    [string]$Toolset = 'v142'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-CMake {
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $patterns = @(
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe',
        'C:\Program Files\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe',
        'C:\Program Files\CMake\bin\cmake.exe',
        'C:\Program Files (x86)\CMake\bin\cmake.exe'
    )
    foreach ($p in $patterns) {
        $f = Get-ChildItem $p -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($f) { return $f.FullName }
    }
    throw 'cmake.exe not found. Install CMake 3.20+ and add it to PATH.'
}

function Get-VsToolchains {
    # returns @{ Vs = '2019', Edition = 'Community', Toolsets = @('14.29...') }
    $result = @()
    $bases = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio'),
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio')
    )
    foreach ($base in $bases) {
        if (-not (Test-Path -LiteralPath $base)) { continue }
        foreach ($vs in (Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue)) {
            foreach ($ed in (Get-ChildItem -LiteralPath $vs.FullName -Directory -ErrorAction SilentlyContinue)) {
                $msvc = Join-Path $ed.FullName 'VC\Tools\MSVC'
                if (-not (Test-Path -LiteralPath $msvc)) { continue }
                $ver = @(Get-ChildItem -LiteralPath $msvc -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)
                $result += [pscustomobject]@{
                    Vs       = $vs.Name
                    Edition  = $ed.Name
                    Path     = $ed.FullName
                    Toolsets = ($ver -join ', ')
                    Has142   = ($ver | Where-Object { $_ -like '14.2*' }).Count -gt 0
                    Has143   = ($ver | Where-Object { $_ -like '14.3*' -or $_ -like '14.4*' }).Count -gt 0
                }
            }
        }
    }
    return $result
}

function Get-WindowsSdkUcrt {
    $out = @()
    $bases = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\Include'),
        (Join-Path $env:ProgramFiles 'Windows Kits\10\Include')
    )
    foreach ($base in $bases) {
        if (-not (Test-Path -LiteralPath $base)) { continue }
        foreach ($d in (Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue)) {
            if (Test-Path -LiteralPath (Join-Path $d.FullName 'ucrt\ctype.h')) { $out += $d.Name }
        }
    }
    return $out
}

$cmake = Find-CMake
Push-Location $root
try {
    # ---------------------------------------------------------------- [0/3]
    Write-Host "[0/3] environment" -ForegroundColor Cyan
    Write-Host "      cmake : $cmake"
    $vss = @(Get-VsToolchains)
    if ($vss.Count -eq 0) {
        Write-Host "      Visual Studio : NOT FOUND" -ForegroundColor Red
    }
    else {
        foreach ($v in $vss) {
            Write-Host ("      Visual Studio {0} {1}  toolsets: {2}" -f $v.Vs, $v.Edition, $v.Toolsets)
        }
    }
    $sdks = @(Get-WindowsSdkUcrt)
    if ($sdks.Count -eq 0) {
        Write-Host "      Windows SDK (ucrt) : NOT FOUND" -ForegroundColor Red
        Write-Host ""
        Write-Host "  !! Windows SDK C-runtime headers (ctype.h, stdio.h...) not found." -ForegroundColor Yellow
        Write-Host "     Symptom: error C1083: cannot open include file: 'ctype.h'" -ForegroundColor Yellow
        Write-Host "     Fix: Visual Studio Installer -> Modify -> under" -ForegroundColor Yellow
        Write-Host "          'Desktop development with C++' check 'Windows 10/11 SDK'." -ForegroundColor Yellow
        Write-Host ""
    }
    else {
        Write-Host ("      Windows SDK (ucrt) : " + ($sdks -join ', '))
    }

    if (-not (Test-Path -LiteralPath $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir | Out-Null
    }

    # ------------------------------------------------------------------
    # A CMake build directory is NOT relocatable: CMakeCache.txt remembers
    # absolute paths, the exact compiler path and the Windows SDK version of
    # the machine that created it.  If the folder was built on another
    # machine (or moved to another path) and then brought back, reusing that
    # cache makes the compiler look for headers that do not exist here
    # ("C1083: cannot open include file: 'ctype.h'").  Detect that and wipe.
    # ------------------------------------------------------------------
    $cachePath = Join-Path $BuildDir 'CMakeCache.txt'
    $rootAbs = [System.IO.Path]::GetFullPath($root).TrimEnd('\')
    $buildAbs = [System.IO.Path]::GetFullPath((Join-Path $root $BuildDir)).TrimEnd('\')

    function Get-CacheValue([string]$text, [string]$name) {
        $m = [regex]::Match($text, '(?m)^' + [regex]::Escape($name) + ':[^=]*=(.*)$')
        if ($m.Success) { return $m.Groups[1].Value.Trim() }
        return $null
    }
    function Format-ComparePath([string]$p) {
        if ([string]::IsNullOrWhiteSpace($p)) { return '' }
        return ([System.IO.Path]::GetFullPath($p) -replace '/', '\').TrimEnd('\').ToLowerInvariant()
    }

    if (Test-Path -LiteralPath $cachePath) {
        $ctext = [System.IO.File]::ReadAllText($cachePath)
        $reason = $null

        $cacheHome = Get-CacheValue $ctext 'CMAKE_HOME_DIRECTORY'
        $cdir = Get-CacheValue $ctext 'CMAKE_CACHEFILE_DIR'
        $cxx = Get-CacheValue $ctext 'CMAKE_CXX_COMPILER'
        $sdkv = Get-CacheValue $ctext 'CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION'

        if ($cacheHome -and (Format-ComparePath $cacheHome) -ne (Format-ComparePath $rootAbs)) {
            $reason = "cached source dir is different: $cacheHome"
        }
        elseif ($cdir -and (Format-ComparePath $cdir) -ne (Format-ComparePath $buildAbs)) {
            $reason = "cached build dir is different: $cdir"
        }
        elseif ($cxx -and -not (Test-Path -LiteralPath $cxx)) {
            $reason = "cached compiler does not exist on this machine: $cxx"
        }
        elseif ($sdkv) {
            $sdkInc = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\Include\$sdkv\ucrt"
            if (-not (Test-Path -LiteralPath $sdkInc)) {
                $reason = "cached Windows SDK $sdkv is not installed on this machine"
            }
        }

        if ($reason) {
            Write-Host ""
            Write-Host "  !! The build directory was created elsewhere and cannot be reused:" -ForegroundColor Yellow
            Write-Host "     $reason" -ForegroundColor Yellow
            Write-Host "     (a CMake build tree is machine/path specific - this is the usual" -ForegroundColor Yellow
            Write-Host "      reason for 'C1083: cannot open include file: ctype.h')" -ForegroundColor Yellow
            Write-Host "     Deleting $BuildDir and configuring from scratch ..." -ForegroundColor Yellow

            Get-ChildItem -LiteralPath $buildAbs -Force -ErrorAction SilentlyContinue |
                Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

            $left = @(Get-ChildItem -LiteralPath $buildAbs -Force -ErrorAction SilentlyContinue)
            if ($left.Count -gt 0) {
                Write-Host ""
                Write-Host "  !! Cannot delete $BuildDir - files are still in use." -ForegroundColor Red
                Write-Host "     Close Visual Studio and any running SPH.exe, then run build.bat again." -ForegroundColor Red
                Write-Host ("     (locked: " + ($left[0].Name) + ")") -ForegroundColor Red
                throw "build directory is locked by another process"
            }
        }
    }

    # ---------------------------------------------------------------- [1/3]
    # Reuse the generator/arch/toolset already stored in CMakeCache.txt.
    # A bare `cmake ..` may have created the cache without an explicit
    # platform, and passing -A x64 then aborts with
    # "generator platform does not match the platform used previously".
    $cache = Join-Path $BuildDir 'CMakeCache.txt'
    $configureArgs = @('-S', '.', '-B', $BuildDir)
    if ($Cxx20) { $configureArgs += '-DSPH_USE_CXX20=ON' }

    if (Test-Path -LiteralPath $cache) {
        Write-Host "[1/3] configure: cmake -S . -B $BuildDir (reusing existing cache)" -ForegroundColor Cyan
        & $cmake @configureArgs
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)" }
    }
    else {
        if ($PSBoundParameters.ContainsKey('Generator')) {
            $candidates = @([pscustomobject]@{ G = $Generator; T = $Toolset })
        }
        else {
            $candidates = @(
                [pscustomobject]@{ G = 'Visual Studio 16 2019'; T = 'v142' },
                [pscustomobject]@{ G = 'Visual Studio 17 2022'; T = 'v143' },
                [pscustomobject]@{ G = 'Visual Studio 16 2019'; T = '' },
                [pscustomobject]@{ G = 'Visual Studio 17 2022'; T = '' }
            )
        }

        $ok = $false
        foreach ($c in $candidates) {
            $args2 = @('-S', '.', '-B', $BuildDir, '-G', $c.G, '-A', $Arch)
            if ($c.T -ne '') { $args2 += @('-T', $c.T) }
            if ($Cxx20) { $args2 += '-DSPH_USE_CXX20=ON' }
            $label = $c.G
            if ($c.T -ne '') { $label = "$label / $($c.T)" }
            Write-Host "[1/3] configure ($label) ..." -ForegroundColor Cyan
            & $cmake @args2
            if ($LASTEXITCODE -eq 0) { $ok = $true; break }
            # fresh directory only: drop the half-written cache and retry
            foreach ($garbage in @('CMakeCache.txt', 'CMakeFiles')) {
                $gp = Join-Path $BuildDir $garbage
                if (Test-Path -LiteralPath $gp) { Remove-Item -LiteralPath $gp -Recurse -Force -ErrorAction SilentlyContinue }
            }
            Write-Host "      -> not usable, trying next toolchain" -ForegroundColor Yellow
        }
        if (-not $ok) {
            throw 'CMake configure failed for every Visual Studio toolchain. See the [0/3] environment list above.'
        }
    }

    # ---------------------------------------------------------------- [2/3]
    $finalize = Join-Path $root 'tools\finalize-generated.ps1'
    $sln = Join-Path $BuildDir 'SPH.sln'
    if (Test-Path -LiteralPath $finalize) {
        Write-Host "[2/3] finalize: drop ALL_BUILD / ZERO_CHECK, relativize paths" -ForegroundColor Cyan
        & $finalize -Sln $sln
        if ($LASTEXITCODE -ne 0) { throw "finalize failed (exit $LASTEXITCODE)" }
    }

    # ---------------------------------------------------------------- [3/3]
    if ($Configuration -eq 'All') {
        $configs = @('Release', 'Debug')
    }
    else {
        $configs = @($Configuration)
    }

    foreach ($cfg in $configs) {
        Write-Host "[3/3] build: --config $cfg --target SPH" -ForegroundColor Cyan
        & $cmake --build $BuildDir --config $cfg --target SPH --parallel
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "  !! Build failed ($cfg). Please send the error above plus the" -ForegroundColor Yellow
            Write-Host "     [0/3] environment block to whoever maintains this package." -ForegroundColor Yellow
            Write-Host "     Common causes: missing Windows SDK (ctype.h), missing v142" -ForegroundColor Yellow
            Write-Host "     toolset, missing 'Desktop development with C++' workload." -ForegroundColor Yellow
            throw "build failed (exit $LASTEXITCODE)"
        }
    }

    Write-Host ''
    Write-Host "================== PATHS (relative) ==================" -ForegroundColor Green
    Write-Host " project root : ."
    Write-Host " build script : CMakeLists.txt"
    Write-Host " source       : 1.cpp"
    Write-Host " build dir    : $BuildDir"
    Write-Host " solution     : $BuildDir\SPH.sln"
    foreach ($cfg in $configs) {
        Write-Host " executable   : $BuildDir\$cfg\SPH.exe"
    }
    Write-Host " work dir     : .   (input.dat / resultN.dat)"
    Write-Host "======================================================" -ForegroundColor Green
}
finally {
    Pop-Location
}
