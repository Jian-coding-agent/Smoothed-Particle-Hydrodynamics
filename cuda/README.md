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
sph_cuda.exe --help
```

Convert a text deck once (the converter also counts `itype` and checks the
fluid | solid | wall block order):

```bat
python tools\sphbin.py make input_1203k.dat input_1203k.sphbin
python tools\sphbin.py info input_1203k.sphbin
```

Environment variables:

| variable | effect |
|---|---|
| `SPH_MAXSTEP` | maximum number of time steps (default 10 000 001) |
| `SPH_PRECISION` | `fp64` / `fp32` when no command-line flag is given |
| `SPH_OUT_F32=1` | write results as float32 (half the size) |
| `SPH_PROFILE` | per-phase timing (prepare / cell / pass1 / constitution / pass2 / update) |
| `SPH_SOLDDYN=1` | walk the grid in the solid kernels instead of the static pair list |
| `SPH_REFLECT=flip` | flip `v_z` unconditionally above `z = 1` (1.cpp's form); the default only reflects outward motion (M8) |
| `SPH_LITERALFLUID=1` | literal `pass1_fluid` body instead of the factored one |
| `SPH_FLUIDLIST=0` | fused fp64 `pass1_fluid` instead of the two-phase list version |
| `SPH_A3_FULL=1` / `=0` | force the fat (stored `sG/sB/sD`) / lean (index-only) solid pair list; unset = auto |
| `SPH_A3_LIMIT_MB` | auto threshold: go lean when the fat list would exceed this (default 4096 MB) |

The last three are diagnostics for A/B comparison; the optimised paths are the
default and are what was validated.

The solid pair list has two layouts (A6, added for 10^7-particle decks).  The
fat one stores the three per-pair weight vectors, 88 B per pair; the lean one
stores only the partner index (4 B/pair) and rebuilds the weights from the
static solid pack every step, with exactly the arithmetic that filled the arrays
— both produce **byte-identical** `result*.sphbin`.  On the 1 203 508-particle
deck the list drops from 1 254 MB (6.95 KB per solid particle) to 66 MB
(0.35 KB), total VRAM from 2.93 GB to 1.77 GB, and the step time goes from
11.3 ms to 13.4 ms (+19 %, the pack is a gather while the stored list is
slot-major and coalesced).  Auto mode keeps the fat list while it fits under
`SPH_A3_LIMIT_MB`, so the current cases keep full speed and only the very large
decks pay the surcharge.

## Multiple GPUs (D1)

`SPH_RANKS=K` runs the same deck on K devices in one process (one host thread
per rank).  The deck is cut into an `nx*ny*nz` block grid whose compute weight is
balanced, every rank also keeps copies of its neighbours' boundary particles
("ghosts"), and the fields those copies need are exchanged **once per step**: E1
carries x, vx, rho and SurFD, and the layer-1 ghost solids run their own
constitution instead of receiving its result (P11).

```bat
set SPH_RANKS=8
sph_cuda.exe input.sphbin
```

| variable | effect |
|---|---|
| `SPH_RANKS=K` | number of ranks; unset or `1` = the ordinary single-GPU path |
| `SPH_RANKS=auto` | rank the devices (bandwidth probe, free memory, display flag) and choose K from the partition model |
| `SPH_RANKS_GPUS=0,1,...` | which devices, in rank order (default `0..K-1`) |
| `SPH_D1_SKIN=<m>` | ghost-shell skin past 3 h (default `1.5*hsml`); the shell is rebuilt when a particle has moved half of it |
| `SPH_D1_REPART=N` | also rebuild the decomposition every N steps |
| `SPH_D1_REGRID=1` | re-search the block grid at every rebuild |
| `SPH_D1_SPLIT=nx,ny,nz` | force one factorisation of K (A/B switch) |
| `SPH_D1PROF=1` | per-phase and per-exchange-point timings |
| `SPH_BARRIER=spin\|block` | rank-barrier implementation (default adaptive) |
| `SPH_D1_W_SOLID` / `SPH_D1_W_GHOST` | partition weights of a solid particle (default 0.45) and of a ghost (default 0.06) |
| `SPH_D1_XCH_UNITS` / `SPH_D1_AUTO_SLACK` | exchange floor (default 4000 load units) and how far from the best prediction the auto-K choice may sit (default 10%) |
| `SPH_D1_DBG=1` | extra stderr: displacements and device memory after a rebuild |

The ghost set is only complete at the instant it is selected, so the skin is not
optional book-keeping: `step_d1()` measures `max |x - x_build|` over the owned
particles of every rank, and when it passes half the skin the ranks refresh the
halo together.  The maximum is reduced every step and rides in the tail of the
exchange block (P17), so the check is exact and costs no readback; the trigger
sits one `MaxV*dt` below half the skin, because the step that detects a crossing
still runs on the old halo.  A skin-triggered refresh keeps the ownership and
only re-selects the ghosts (P16); the cut is re-done by `SPH_D1_REPART` or
`SPH_D1_REGRID`.
`SPH_D1_SKIN=0` restores the old behaviour (ghosts chosen once, never
revalidated) and prints a warning, because a long run then silently drops
interactions: at K=8 on `input.dat` that is 2.0e-6 m and 1.7 Pa away from the
single-GPU answer after 3001 steps.

**K > 1 is byte-identical to K = 1** (V2): every rank bins into the same global
neighbour grid and orders a cell by global particle id, so the summation order is
the one the single-GPU run uses.  A halo or exchange bug can no longer hide under
1e-16 noise.  Measured: K = 4 and K = 8 over 3001 steps, with automatic rebuilds,
with `SPH_D1_REPART=1000`, and through `SPH_RANKS=auto` -- zero differing bytes.

Two constraints: the decomposition is fp64 only (`SPH_RANKS` + `--fp32` is
refused), and a deck whose walls have a non-zero velocity is refused too -- the
exchange carries the fluid and solid groups only, so a moving wall would not
reach its ghosts.

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

> The comparison against `1.cpp` needs one caveat since M8: the port now reflects
> only *outward* motion at `z = 1` by default, while the reference flips `v_z`
> unconditionally.  On `input.dat` the fluid stays below 0.45 m, so the two agree
> exactly; a deck that reaches `z = 1` must be compared with `SPH_REFLECT=flip`
> until the physics owner settles the case definition.

> The numbers below were obtained with the previous fixed-case build and its
> text output.  The current build uses the new-case boundary conditions (no
> test-only solid clamp, `z > 1.0` reflection) and binary I/O; the new-case
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
src/sph_host.cu       host driver (binary/text loading, dynamic allocation, step loop, binary output)
docs/DESIGN.md        algorithm, porting notes, validation, performance analysis
docs/BINARY_IO.md     binary container, runtime sizing, new-case measurements
tools/sphbin.py       text deck -> .sphbin converter / inspector
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
