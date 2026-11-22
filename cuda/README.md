# SPH on CUDA

A GPU (CUDA) port of the three-dimensional SPH fluid-structure interaction solver
from `1.cpp`.  The solver is templated on the scalar type: the same code runs in
double (default) or single precision, selected at run time.

> **2026-11-19 revision — one binary, any deck.**  The particle counts and the
> neighbour grid now come from the deck at start-up; decks are stored in a
> self-describing `.sphbin` container (the 14-column text deck is still
> accepted); results are binary `.sphbin` files.  Details, format and the
> new-case measurements: [`docs/BINARY_IO.md`](docs/BINARY_IO.md).  The
> fixed-case numbers further down describe the previous build.


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

The first argument is the deck; the name is arbitrary.  A `.sphbin` container is
read directly, a text deck is parsed as before.  Results are written to the
**current directory** as `result<N>.sphbin` every 500 steps:

```bat
sph_cuda.exe cases\dam_1203k.sphbin            :: fp64 (default)
sph_cuda.exe cases\dam_1203k.sphbin --fp32     :: single precision
sph_cuda.exe input.dat                          :: legacy 14-column text deck
sph_cuda.exe --precision=fp32 <deck>
set SPH_PRECISION=fp32                          :: same, via the environment
sph_cuda.exe --resume=dam.chk <deck>            :: continue a checkpoint
sph_cuda.exe --help
```

Convert a text deck once (the converter also counts `itype` and checks the
fluid | solid | wall block order):

```bat
python tools\sphbin.py make input_1203k.dat input_1203k.sphbin
python tools\sphbin.py info input_1203k.sphbin
```

### Checkpoint / restart

A long run can be stopped and continued bit-identically: `--ckpt=<file>` (or
`SPH_CKPT`) writes the cross-step state, `--resume=<file>` (or `SPH_CKPT_IN`)
continues from it, and `SPH_MAXSTEP` counts the steps of the current segment.
Frames keep the index a straight run would have written.

```bat
set SPH_MAXSTEP=200000 & set SPH_CKPT=dam.chk & set SPH_CKPT_EVERY=5000
sph_cuda.exe cases\dam_1203k.sphbin --fp32

set SPH_MAXSTEP=100000
sph_cuda.exe cases\dam_1203k.sphbin --fp32 --resume=dam.chk
```

A `result*.sphbin` is **not** a restart state (4 values of the 20 carried per
particle, and three of them are not even current: see
[`docs/CHECKPOINT.md`](docs/CHECKPOINT.md)).  The checkpoint is 160 B/particle
in fp64 and 108 B in fp32 — 183.6 / 124.0 MiB for 1.2 M particles.  The deck,
the precision and the solver switches must match; the loader refuses anything
else instead of continuing into different physics.

Environment variables:

| variable | effect |
|---|---|
| `SPH_MAXSTEP` | maximum number of time steps (default 10 000 001), for this invocation |
| `SPH_PRECISION` | `fp64` / `fp32` when no command-line flag is given |
| `SPH_OUTCADENCE` | steps per result file (default 500) |
| `SPH_OUT_F32=1` | write results as float32 (half the size) |
| `SPH_CKPT` | write a checkpoint at the end of the run |
| `SPH_CKPT_EVERY` | also write one every N steps (0 = only at the end) |
| `SPH_CKPT_IN` | resume from this checkpoint |
| `SPH_STEP0` | absolute step number of a fresh run (result naming only) |
| `SPH_SOLDBOX` | solid wall clamp: `z,y0,y1` bounds, or `off` (default: the deck's geometry -- 3.7M case `0.01,0.00,3.31`, 1.2M case `0.01,0.00,1.31`) |
| `SPH_PROFILE` | per-phase timing (prepare / cell / pass1 / constitution / pass2 / update) |
| `SPH_STEPTIME=1` | print `ms/step` every 500 steps (with `SPH_PROFILE`: per phase) |
| `SPH_SORTEDTHREADS=0` | turn off the cell-sorted thread mapping (see below) |
| `SPH_SOLDDYN=1` | walk the grid in the solid kernels instead of the static pair list |
| `SPH_LITERALFLUID=1` | literal `pass1_fluid` body instead of the factored one |
| `SPH_FLUIDLIST=0` | fused fp64 `pass1_fluid` instead of the two-phase list version |

The last three are diagnostics for A/B comparison; the optimised paths are the
default and are what was validated.

`SPH_SORTEDTHREADS` (S1): the fluid kernels are launched over the cell-sorted
order, so the 32 threads of a warp own spatially adjacent particles.  Particle
ids are frozen at t = 0, so without this the warp's threads scatter as soon as
the flow mixes and `pass1_fluid` costs 2.8x more for exactly the same pair count
(5.40 vs 1.90 ms on the 1.2 M deck).  The per-particle summation order is
unchanged, so the output is byte-identical; it is on by default in fp32 and off
in fp64 (where the two-phase list is faster).  It is slightly slower for the
first few hundred steps, while the particles are still lattice ordered.

## Precision

Both instantiations are compiled in by default; CMake can trim the build:

| option | default | meaning |
|---|---|---|
| `SPH_ENABLE_FP64` | ON | build the double path |
| `SPH_ENABLE_FP32` | ON | build the single path |
| `SPH_DEFAULT_PRECISION` | `double` | default when no flag is given |

Measured on an RTX 5080, 237 298 particles, 11.1 M neighbour pairs (previous
fixed-case build; the current build is runtime-sized and reports the new-case
numbers in [`docs/BINARY_IO.md`](docs/BINARY_IO.md)):

| | ms/step | vs 32-thread CPU (163 ms/step) | accuracy vs fp64 |
|---|---|---|---|
| fp64 | **5.01** | **33x** | reference precision |
| fp32 | **1.64** | **100x** | solid `p` median 0.01 Pa / worst 0.17 Pa, `x` within 1e-6 m, `vx` within 1e-5 m/s |

Start-up builds the static solid pair list once: 4 844 544 pairs, ELL width 92,
402 MB in fp64 (212 MB in fp32), 0.17 s.  A rigid translation of the solid does
not change it — `tools/check_rigid_translation.py` asserts exactly that, after
the list was found to be sensitive to the solid's current position (review
finding C1; DESIGN sec. 12.5).

fp32 needs compensated `x`/`rho` updates and a displacement form for the solid,
otherwise the updates fall below float resolution and the plate does not move at
all; that is implemented (`Real = float` only) and verified on the GPU.  The
remaining fp32 error is the quantisation of `p = c0^2(rho-1000)` with `rho` near
1000 (0.14 Pa per step).

## Validation

> The numbers below were obtained with the previous fixed-case build and its
> text output.  The current build uses the new-case boundary conditions (the
> solid wall clamp is re-enabled by default behind `SPH_SOLDBOX`, `z > 1.0`
> reflection) and binary I/O; the new-case
> comparison against the CPU reference is in
> [`docs/BINARY_IO.md`](docs/BINARY_IO.md).

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
src/sph.h             runtime dims, SoA container, device math, prototypes
src/sph_kernels.cu    all CUDA kernels and launch wrappers
src/sph_host.cu       host driver (binary/text loading, dynamic allocation, step loop, binary output, checkpoints)
docs/DESIGN.md        algorithm, porting notes, validation, performance analysis
docs/BINARY_IO.md     binary container, runtime sizing, new-case measurements
docs/CHECKPOINT.md    restart state, SPHCHK format, what a resume refuses
tools/sphbin.py       text deck -> .sphbin converter, deck / checkpoint inspector
tools/cmp_bin_text.py compare a .sphbin result against a legacy text result
tools/                result comparison helper (from the original package)
```

`input.dat` (20 MB, 237 298 particles x 14 columns) is included as a text deck;
the binary decks are generated from it with `tools\sphbin.py` and are not
committed.  The 500-step reference results stay in the original package because
they are ~200 MB.

## Design notes

See [`docs/DESIGN.md`](docs/DESIGN.md) for the algorithm, the gather-based
architecture, the list of reference behaviours that had to be reproduced
exactly, the validation method and the measured performance analysis.
