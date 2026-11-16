# SPH on CUDA

A GPU (CUDA) port of the three-dimensional SPH fluid-structure interaction solver
from `1.cpp`.  The solver is templated on the scalar type: the same code runs in
double (default) or single precision, selected at run time.

## Requirements

| | |
|---|---|
| GPU | NVIDIA, `sm_70` or newer (built for the local GPU by default) |
| CUDA | 13.x toolkit (`nvcc`) |
| Host compiler | MSVC 2019/2022/2026 with the C++ workload |
| CMake | >= 3.24 (only needed for the `build.bat` path) |

## Build

```bat
build.bat                  :: finds Visual Studio, configures with CMake+Ninja, builds
build.bat Debug
```

Everything is relative to this directory; no absolute paths in any script.  The
build output is `build\sph_cuda.exe`.

If you prefer plain CMake:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Run

The solver reads `input.dat` and writes `result<N>.dat` from the **current
directory**, so start it from a directory that contains `input.dat`:

```bat
sph_cuda.exe                          :: fp64 (default)
sph_cuda.exe --fp32                   :: single precision
sph_cuda.exe --precision=fp32
set SPH_PRECISION=fp32                :: same, via the environment
sph_cuda.exe --help
```

Environment variables:

| variable | effect |
|---|---|
| `SPH_MAXSTEP` | maximum number of time steps (default 10 000 001) |
| `SPH_PRECISION` | `fp64` / `fp32` when no command-line flag is given |
| `SPH_PROFILE` | per-phase timing (prepare / cell / pass1 / constitution / pass2 / update) |
| `SPH_SOLDDYN=1` | walk the grid in the solid kernels instead of the static pair list |
| `SPH_LITERALFLUID=1` | literal `pass1_fluid` body instead of the factored one |

The last two are diagnostics for A/B comparison; the optimised paths are the
default and are what was validated.

## Precision

Both instantiations are compiled in by default; CMake can trim the build:

| option | default | meaning |
|---|---|---|
| `SPH_ENABLE_FP64` | ON | build the double path |
| `SPH_ENABLE_FP32` | ON | build the single path |
| `SPH_DEFAULT_PRECISION` | `double` | default when no flag is given |

Measured on an RTX 5080, 237 298 particles, 11.1 M neighbour pairs:

| | ms/step | vs 32-thread CPU (211 ms/step) | accuracy vs fp64 |
|---|---|---|---|
| fp64 | **12.42** | **17.0x** | reference precision |
| fp32 | **2.43** | **87x** | solid `p` median 0.01 Pa / worst 0.17 Pa, `x` within 1e-6 m, `vx` within 1e-5 m/s |

fp32 needs compensated `x`/`rho` updates and a displacement form for the solid,
otherwise the updates fall below float resolution and the plate does not move at
all; that is implemented (`Real = float` only) and verified on the GPU.  The
remaining fp32 error is the quantisation of `p = c0^2(rho-1000)` with `rho` near
1000 (0.14 Pa per step).

## Validation

Against the original package (`result0..9.dat` = 500 steps) and against a
3000-step pristine CPU run (which reaches fluid/plate contact at step 566):

* `x0 x1 x2`, `rho`, `mass`, `itype`, `X0`, `rho0`  -  identical at the printed
  precision;
* `p`  -  maxAbs 1e-3 over 500 steps, 1e-2 over 3000 (the CPU's own
  rerun-to-rerun spread is 1e-2);
* `vx`  -  maxAbs 1e-9 over 500 steps, 1e-8 over 3000 (same as the CPU's own
  spread).

Two identical runs produce byte-identical `result*.dat`, in both precisions.
Enabling the static solid pair list (A3) or the factored fluid loop (A5) on its
own reproduces the same 500-step output to the printed precision, and the float
neighbour pre-filter (A4) is bit-identical.

Intermediate arrays compared after one step (binary dumps, not the 6-digit text):
`F` 2.2e-15, `Pia` 2.5e-8, `rho` 3.0e-12, `SForce` 2.8e-13, `vecn` 7.5e-11,
`A` 2.7e-15 relative.  `compute-sanitizer` (`memcheck`, `initcheck`, `racecheck`,
`synccheck`) reports no errors.

Compare two result sets with the tool from the original package:

```bat
tools\CompareResults.exe <workdir> <reference-dir>
```

## Diagnostics

| variable | effect |
|---|---|
| `SPH_BENCH=1` | time every kernel separately, then exit |
| `SPH_VERIFY=1` | validate the cell list after it is built |

Debug dumps (used for the array-by-array validation) are available through
`SPH_DUMPSTEP`, `SPH_DUMPA` and `SPH_LOADA`; see `docs/DESIGN.md` sec.9.

## Layout

```
CMakeLists.txt        build definition (no absolute paths, precision switches)
build.bat             one-click build
src/sph.h             constants, SoA container, device math, prototypes
src/sph_kernels.cu    all CUDA kernels and launch wrappers
src/sph_host.cu       host driver (I/O, initialisation, step loop, output)
docs/DESIGN.md        algorithm, porting notes, validation, performance analysis
tools/                result comparison helper (from the original package)
```

`input.dat` (20 MB, 237 298 particles x 14 columns) is included so the package
runs out of the box; the 500-step reference results stay in the original
package because they are ~200 MB.

## Design notes

See [`docs/DESIGN.md`](docs/DESIGN.md) for the algorithm, the gather-based
architecture, the list of reference behaviours that had to be reproduced
exactly, the validation method and the measured performance analysis.
