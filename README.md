# Smoothed Particle Hydrodynamics (SPH)

**A 3D SPH fluid–structure interaction solver: a reference CPU implementation and a validated CUDA port.**

[中文](README.zh-CN.md) · English

---

This repository contains a weakly compressible SPH solver with Riemann-solver
pair interactions, a total-Lagrangian solid with Kelvin–Voigt friction, and a
mirror-Riemann fluid–structure coupling.  It comes in two forms:

* **`cpu/`** — the original single-translation-unit C++/OpenMP reference solver
  (`1.cpp`, double precision).  This is the correctness baseline.
* **`cuda/`** — a CUDA port of the same physics, templated on the scalar type so
  that one binary runs in **fp64** (default) or **fp32**, selected at run time.

Both read the same `input.dat` (237,298 particles: 101,250 fluid, 60,320 solid
and 75,728 wall particles; 11.1 M neighbour pairs at t = 0) and write
`result<N>.dat` every 50 steps, with a fixed `dt = 1e-5`.

## Performance

Measured on an NVIDIA RTX 5080 (`sm_120`), CUDA 13.3, 237,298 particles:

| implementation | precision | ms/step | vs 32-thread CPU | accuracy |
|---|---|---:|---:|---|
| `cpu/` | fp64 | 211 | 1x | reference |
| `cuda/` | fp64 | **12.42** | **17.0x** | matches the reference to the printed precision |
| `cuda/` | fp32 | **2.43** | **87x** | solid `p` median 0.01 Pa, `x` within 1e-6 m |

The CPU baseline is a 16-core Ryzen 9 9950X with 32 OpenMP threads.  Two runs
with identical settings produce **byte-identical** `result*.dat`, in both
precisions.

## How it works

The port keeps the reference scheme unchanged and reorganises it for the GPU:

* every pair interaction is evaluated in **gather** form (one thread per
  particle), which makes the per-step work a handful of kernels with no
  atomics;
* a uniform cell list provides the neighbour candidates, with the exact distance
  test used only for survivors of a float pre-filter (compiled out in the fp32
  build, where the exact test is already float);
* solid/solid pairs are **precomputed once** in reference configuration (the
  total-Lagrangian formulation makes them constant), turning the two solid
  kernels into flat CSR loops;
* the result writer formats with `std::to_chars` on worker threads.

See [`cuda/docs/DESIGN.md`](cuda/docs/DESIGN.md) for the full derivation,
including the reference behaviours that had to be reproduced exactly.

## Validation

Against the reference `result0..9.dat` (500 steps):

* `x0 x1 x2`, `rho`, `mass`, `itype`, `X0`, `rho0` — **no differences** at the
  printed precision;
* `p` — max abs **1e-3**;  `vx` — max abs **1e-9**.

Against a pristine CPU run to 3000 steps (past fluid/plate contact at step 566):
`p` max abs **1e-2**, `vx` **1e-9** (the CPU's own run-to-run spread is the same
size).  After one step, intermediate arrays agree to 2.2e-15 … 2.5e-8; the
CUDA build is also clean under `compute-sanitizer` (`memcheck`, `initcheck`,
`racecheck`, `synccheck`).

## Quick start

### CUDA version

Requirements: NVIDIA GPU (`sm_70` or newer), CUDA 13.x, MSVC 2019/2022/2026 with
the C++ workload, CMake ≥ 3.24.

```bat
cd cuda
build.bat
```

Then run from the **repository root**, because the solver reads `input.dat` from
the current directory:

```bat
set SPH_MAXSTEP=500
cuda\build\sph_cuda.exe          :: fp64
cuda\build\sph_cuda.exe --fp32   :: single precision
```

This writes `result0.dat` … `result9.dat` (one file every 50 steps) into the
current directory.  `cuda\build\sph_cuda.exe --help` lists the other options.

### CPU reference

Requirements: Visual Studio 2019 (v142) or newer with the C++ workload, and
CMake ≥ 3.20.

```bat
cd cpu
build.bat
cd ..
set SPH_MAXSTEP=500
cpu\build\Release\SPH.exe
```

The CPU run is much slower (about 105 s for 500 steps at 32 threads).  Use
`SPH_THREADS` to limit the thread count and `SPH_MAXSTEP` to stop early.

## Repository layout

```
input.dat                shared input deck (237,298 particles, ~21 MB)
cpu/                     reference solver: 1.cpp, CMake project, build scripts
cuda/                    CUDA port: src/, CMakeLists.txt, build.bat
  docs/DESIGN.md         design, porting notes, validation, performance analysis
docs/
  CUDA_PORT_ANALYSIS.md  source analysis and porting plan (Chinese)
```

## Documentation

| document | contents |
|---|---|
| [`cuda/docs/DESIGN.md`](cuda/docs/DESIGN.md) | algorithm mapping, precision template, validation, measured performance, remaining work |
| [`docs/CUDA_PORT_ANALYSIS.md`](docs/CUDA_PORT_ANALYSIS.md) | analysis of `1.cpp` and the initial porting plan (Chinese) |
| [`cuda/README.md`](cuda/README.md) | CUDA build, run options, diagnostics |
| [`cpu/README.md`](cpu/README.md) | CPU project, build matrix, troubleshooting |
| [`cpu/QUICKSTART.txt`](cpu/QUICKSTART.txt) | one-page CPU quick start |

## Notes

* The 500-step reference results (`result0..9.dat`, ~200 MB) are **not** stored
  in the repository.  Regenerate them with the CPU solver and compare with
  `cpu\tools\CompareResults.exe <workdir> <reference-dir>`.
* The build scripts target Windows/MSVC; the CUDA sources themselves are
  ordinary CUDA C++.
* `input.dat` is a 14-column text deck; the solver reads and writes it relative
  to the working directory.

## License

MIT — see [LICENSE](LICENSE).
