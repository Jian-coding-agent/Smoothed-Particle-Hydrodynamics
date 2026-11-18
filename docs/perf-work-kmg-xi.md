# Performance work: CPU and GPU computation details on KMG-XI

| | |
|---|---|
| Date | 2026-11-19 |
| Machine | KMG-XI: RTX 5090 (`sm_120`, 170 SMs, 31.8 GB), i9-14900K (8 P-cores + 16 E-cores, 32 logical), 128 GB DDR5, Windows 11 build 26200, NVIDIA driver 610.88 / CUDA UMD 13.3 |
| Case | `input.dat`, 237 298 particles (101 250 fluid, 60 320 solid, 75 728 wall), `dt = 1e-5`, 11.10 M neighbour pairs at t=0 |
| Build | `98fcf58`; `sph_cuda.exe` 5 400 064 B, SHA256 `F9249C35…E4ADCBE`; `SPH.exe` 1 325 056 B (unchanged since `355b24d`) |
| Companion | [`docs/perf-work-hrb-jian.md`](perf-work-hrb-jian.md) — the same change on the RTX 5080, and the CPU reference in its §10; [`docs/perf-work-tys-jian.md`](perf-work-tys-jian.md) — the same change on an RTX 4090 / Linux, and a second CPU baseline in its §7; [`docs/edit-2026-11-19-12-19.md`](edit-2026-11-19-12-19.md) — correctness side |

This document records **how the two implementations actually compute a step**,
side by side, and what that costs on this machine.  The two programs are not the
same algorithm with a different back end: the CPU reference walks a pair list and
reduces per-thread buffers, the GPU walks particles and gathers.  Almost every
number below follows from that one difference.

All GPU times are milliseconds per step from the `[time]` line, which excludes
result-file writing.  CPU times are wall clock of the process unless marked
"compute".  §7 spells out why the conventions must not be mixed.

## 1. Headline

500 steps, same deck, same machine, back to back:

| | step (ms) | 500-step compute | start-up | 10 result files | end-to-end |
|---|---:|---:|---:|---:|---:|
| GPU fp64 | **3.265** | 1.632 s | 0.307 s | 0.582 s | **2.521 s** (5.04 ms/step) |
| GPU fp32 | **1.157** | 0.578 s | 0.271 s | 0.522 s | **1.371 s** (2.74 ms/step) |
| CPU, 16 threads pinned to the P-cores | **138.4** | 69.2 s | 1.08 s | ~14 s | **84.5 s** (169.1 ms/step) |
| CPU, 16 threads unbound | 166 | 83.0 s | 1.08 s | ~28 s | **112.5 s** (225 ms/step) |
| CPU, 32 threads | 195.9 | 98.0 s | 1.08 s | ~25 s | **124.4 s** (249 ms/step) |

fp64 across five runs: 3.260 / 3.265 / 3.265 / 3.265 / 3.268 ms — a 0.2 % spread.
fp32: 1.152 / 1.157 / 1.157 / 1.160 / 1.163.

Ratios, compute against compute and wall against wall:

| | fp64 | fp32 |
|---|---:|---:|
| GPU step vs CPU compute, both at their best setting | **42.4x** | **119.6x** |
| end-to-end, everything included | **33.5x** | **61.7x** |
| GPU step vs the 163 ms/step wall baseline published in `perf-work-hrb-jian.md` §10 (different machine, same convention as the repository headlines) | 49.9x | 140.9x |

Cross-machine, the GPU is unchanged in this work:

| | fp64 step | fp32 step |
|---|---:|---:|
| RTX 4090 / sm_89 / CUDA 12.2 (`perf-work-tys-jian.md`) | 4.14 | 1.28 |
| RTX 5080 / sm_120 / CUDA 13.3 (`perf-work-hrb-jian.md`) | 5.006 | 1.634 |
| **RTX 5090 / sm_120 / CUDA 13.3 (here)** | **3.265** | **1.157** |

The 5090 is 1.53x the 5080 and 1.27x the 4090 in fp64 on an unchanged build.

## 2. How the CPU reference computes a step

### 2.1 Data layout

Everything is a fixed-size global array of `double`, one entry per particle,
component-major (`x[3][num]`, `vx[3][num]`, `X0[3][num]`, …); the four 3x3
tensors (`A`, `F`, `DFdt`, `Pia`) are nine separate arrays each.  69 doubles per
particle = 552 B, about 131 MB of live state.  Nothing is allocated inside the
step loop and nothing is dynamically sized: the particle count is compiled in
(`num = 237298`), which is why `input.dat` is fixed-width.  `input.dat` is read
with no error check at all — a missing file leaves every array zeroed, puts all
237 298 particles in one cell, and the run still exits 0.

### 2.2 The pair list is rebuilt every step

`direct_find()` re-bins all 237 298 particles into a uniform `27 x 27 x 44` grid
(cell edge `3*hsml = 0.03 m`) with a counting sort, then walks the 27 surrounding
cells of every particle and emits every accepted pair once with `i < j`.  That is
~11.1 M pairs re-derived from scratch on every step, and it is why the reference
carries no topology at all — including for the solid, whose interaction set is
actually constant (the GPU exploits this, §3.3).

### 2.3 One step, five phases

```
1. direct_find()          rebuild the grid, enumerate ~11.1 M pairs
2. continue_density()
   2.0 serial              SurFD := SurF;  p := c0^2 (rho-1000), clamped at 0
   2.1 pass 1 over pairs    fluid/fluid, fluid/(solid|wall), solid/solid
                            -> drhodt, dvxdt, viscos, SurF, F, DFdt
   2.2 reduce               fold `nth` private buffers back into the globals
   2.3 constitution         per solid particle: J, rho = rho0/J, SD, PD, Pia
   2.4 pass 2 over pairs    solid/solid -> PIJW; fluid/solid -> SForce
   2.5 reduce               fold again
3. integrate               explicit Euler, gravity, 30 m/s speed cap
4. every 50 steps          write result<N>.dat, 237 298 lines x 14 columns
```

### 2.4 Why it needs 2 GB of scratch

Each pass updates **both** ends of every pair, so a parallel loop has to either
use atomics or keep one accumulator set per thread.  The reference does the
latter: `g_omp_buf` is `nth x 35 x num` doubles ([cpu/1.cpp:125](../cpu/1.cpp#L125)),
i.e. **1.01 GB at 16 threads and 2.03 GB at 32**.  Each thread zeroes only the
ranges it can write ([cpu/1.cpp:683](../cpu/1.cpp#L683)), scatters into its own
slice with `schedule(dynamic, 16384)`, and then two static-scheduled passes over
`[0, ntotal)` and `[ntotal, ntotal+SolNum)` sum the `nth` copies
([cpu/1.cpp:865](../cpu/1.cpp#L865), [cpu/1.cpp:898](../cpu/1.cpp#L898), and
again for pass 2 at [1029](../cpu/1.cpp#L1029)/[1105](../cpu/1.cpp#L1105)).

Two consequences that matter later:

1. **The reduction is not free** — about 22 % of the CPU runtime
   ([`DESIGN.md` §3.1](../cuda/docs/DESIGN.md)).
2. **The scratch grows with the thread count**, so the zeroing and reduction
   traffic grows with it too.  Threads buy less than they cost, which is exactly
   what §5 measures.

## 3. How the CUDA port computes a step

### 3.1 Gather, not scatter

The port never walks the pair list.  Every term in this solver is symmetric in
`(i, j)` once you notice `e_j = -e_i`, `dwdx` is odd in its first argument and
`U_L - U_R` is the same seen from either side, so each particle can accumulate
**its own half** of an interaction.  One thread per particle, each thread walks
its own 27 cells and gathers:

| quantity | gather form |
|---|---|
| `drhodt` | `-2 rho_p (m_q/rho_q)(U_L - U*) dwdr(r)` |
| `dvxdt` | `-2 m_q (P*/(rho_p rho_q)) gradW(x_p - x_q)` |
| `viscos` | `2 m_q eta S (v_p - v_q) / (rho_q (r^2 + 0.01 h^2))` |
| `F`, `DFdt` | `(m_q/rho0_q)(x_q - x_p) (x) (A_p g0_pq)` and `(m_q/rho0_q)(v_q - v_p) (x) g0_pq` |
| `PIJW` | `(m_q/(rho_p rho_q))[Pia_p (A_p g0_pq) - Pia_q (A_q g0_qp)]` |

The only asymmetric input is `eta`, which needs `SurFD[max(p,q)]` — the port's one
real functional bug, now reproduced deliberately.

**There are no atomics, no reduction kernels and no per-thread buffers.**  That
single decision removes the 2 GB of scratch, the reduction passes, and the
thread-count sensitivity of §2.4.

### 3.2 Kernel map

| kernel | threads | work |
|---|---|---|
| `k_prepare` | `0..NUM` | `SurFD := SurF`, `p`, `irho` |
| `k_cell_hist` / `k_cell_scan` / `k_cell_scatter` | `0..NUM` / 1 block / `0..NUM` | counting sort into the same fixed grid; also produces `cellSolid` / `cellFluid` |
| `k_pass1_fluid(_fast)` | `0..NTOTAL` | fluid forces |
| `k_pass1_solid_static` | `NTOTAL..NALL` | `F`, `DFdt` from the static solid list |
| `k_constitution` | `NTOTAL..NALL` | 3x3 constitutive update, `rho = rho0/J`, `Pia` |
| `k_pass2_fluid` | `0..NTOTAL` | `SForce` from solid neighbours |
| `k_pass2_solid_static` | `NTOTAL..NALL` | `PIJW` from the static list, `SForce` from fluid neighbours |
| `k_update_fluid` / `k_update_solid` / `k_update_pos` | ranges | integration |

Ten kernels plus three `memset`s per step, one `cudaDeviceSynchronize()`.

### 3.3 The three changes that produced this build

* **A3 — static solid pair list.**  The solid/solid interaction set is fixed in
  the reference configuration, so it is enumerated once (4 844 544 pairs) instead
  of every step.  Built from `X0`, not from `x` — the C1 defect, where binning
  from `x` silently dropped pairs once the solid moved ~2 cm and cost 5.3 MPa of
  `trace(Pia)`.
* **P2 — slot-major (ELL) layout.**  Entry `j` of solid particle `k` lives at
  `j*SOLNUM + k`, so a warp walking slot `j` reads 32 consecutive entries instead
  of 32 CSR segments ~89 entries apart.  Width 92, padding 14.6 %, 402 MB in fp64
  and 212 MB in fp32; both precisions store the same pair set.
* **P1 — fp64 `pass1_fluid` in two phases.**  The fused loop runs its FP64 body
  whenever *any* lane of a warp has a hit, while only ~15 % of candidates are
  hits.  The port first records each fluid particle's neighbours in exactly the
  fused traversal's order (slot-major, 192 slots, 77.8 MB) and then runs the same
  body over that list, keeping the FP64 sum order identical.  fp32 keeps the
  fused kernel: it is traversal bound there (body 0.77 ms against 0.56 ms for the
  traversal alone), and the split was measured as a loss.

Both precisions print the same list at start-up, and it is the line to check that
the right binary is running:

```
[A3] solid pair list: 4844544 pairs, 80.3 per solid particle, ELL width 92 (padding 14.6%), 402.2 MB
[init] fp64, 0.186 s, 552 bytes/particle
```

## 4. Where the GPU step time goes

`SPH_PROFILE=1`, 500 steps (seconds per step as printed):

| phase | fp64 | fp32 |
|---|---:|---:|
| prepare | 0.0000 | 0.0000 |
| cell list | 0.0002 | 0.0001 |
| **pass 1** | **0.0024** | **0.0006** |
| constitution | 0.0000 | 0.0000 |
| pass 2 | 0.0006 | 0.0004 |
| update | 0.0000 | 0.0000 |
| **step** | **3.29** | **1.17** |

pass 1 is 73 % of the fp64 step and 51 % of fp32.  Against the reference 5080,
pass 1 falls from 3.53 to 2.4 ms (1.5x) while pass 2 barely moves (0.62 to
0.6 ms, 1.03x) — the solid work that A3/P2 were about is already small, so
everything the faster card buys shows up in the fluid pass.  pass 1 is FP64-pipe
bound and does not scale with SM count like a bandwidth-bound kernel would, which
is why 2x the SMs (170 vs 84) buys 1.53x and not 2x.

## 5. CPU thread scaling: pinning is worth more than threads on this machine

The i9-14900K is 8 P-cores plus 16 E-cores, and every phase of the CPU step ends
at a barrier, so the slowest thread sets the step time.  Steady-state step cost,
measured over steps 51-100 (a 51-step and a 100-step run differ by 49 steps and
the same number of result files, so the file cost cancels):

| threads | affinity | ms/step (compute) |
|---:|---|---:|
| 8 | none | 276.6 |
| 16 | none | 166 |
| **16** | **logical CPUs 0-15 (the P-cores)** | **138.4** |
| 24 | none | 179.6 |
| 32 | none | 195.9 |

Two findings, and the second is the important one:

1. **Adding the 16 E-cores makes the step 18 % slower** (166 → 195.9).  Extra
   threads are not free here: the 2 GB of §2.4 grows with the thread count, and
   an E-core thread finishes its share of a barrier-synchronised loop well after
   a P-core thread does.
2. **Pinning 16 threads to the 16 P-core logical CPUs beats every unpinned
   setting by 1.20x** (166 → 138.4), and 1.42x against the 32-thread default.
   The win is larger than the compute-only figure suggests, because the writer
   threads stop competing with the compute threads for the E-cores: the 500-step
   wall clock drops from 112.5 s to **84.5 s** (1.33x).

`OMP_PROC_BIND` = `true`/`spread` and `OMP_PLACES=cores` all landed within 1 % of
the unbound figure, so the OpenMP runtime's own placement policies do not find
the P-cores on this machine; the affinity mask has to be set explicitly.

There is no thermal throttling: over a 500-step run the 50-step blocks take
11.35, 11.32, 11.33, 11.34, 11.30, 11.29, 11.33, 11.27 s after a faster first
block (9.58 s) — flat to 0.7 %.  The machine runs with VBS/HVCI enabled and on the
Balanced power plan, i.e. the shipped state, and that was left alone.

## 6. The CPU side, across the three machines

The reference solver is the same source everywhere (a prebuilt `SPH.exe` on the two
Windows machines, a GCC build on Ubuntu), so the differences below are hardware,
OS and compiler, not code:

| machine | CPU | OS / compiler | threads | compute ms/step | 500-step wall |
|---|---|---|---:|---:|---:|
| `perf-work-hrb-jian` | Ryzen 9 9950X (16 cores) | Windows / MSVC | 32 | 131 | 81.4 s (163 ms/step) |
| `perf-work-tys-jian` | i9-14900KF (8P+16E) | Ubuntu / GCC 12.3 | 16, pinned | 84.8 | 48.3 s |
| `perf-work-tys-jian` | same | same | 32 | 77.3 | 44.6 s |
| **KMG-XI (here)** | **i9-14900K (8P+16E)** | **Windows / MSVC** | **16, pinned** | **138.4** | **84.5 s (169.1 ms/step)** |
| KMG-XI (here) | same | same | 32 | 195.9 | 124.4 s |

Three things fall out of that table:

* **This machine and the 9950X are the same speed once pinned** — 84.5 s against
  81.4 s, 4 % apart, on completely different silicon.  Any impression that the
  CPU here is slow comes entirely from thread placement (§5).
* **Windows/MSVC is the expensive half.**  The *identical* CPU die type (14900K
  against 14900KF) runs this solver 1.63x faster under GCC 12.3 on Ubuntu
  (84.8 against 138.4 ms/step).  `perf-work-tys-jian.md` §7 attributes a 1.47x
  factor to `-O3` versus `/fp:precise`; measured across the two Windows machines
  and the Linux one, the end-to-end compiler effect here is 1.6x.
* **The E-cores behave differently between the two 14900K runs.**  On Linux,
  32 threads still win slightly (77.3 against 84.8) because the neighbour search
  scales well; on Windows, 32 threads lose (195.9 against 138.4).  Same silicon,
  opposite conclusion — which is why the thread count has to be recorded next to
  every CPU number, not just the CPU model.

The file writer costs **~1.4 s per result file** here (the 500-step wall of
84.5 s is 69.2 s of compute + 1.08 s of start-up + ~14 s of output), matching the
1.47 s recorded on the 9950X and 540-546 ms on Linux.  The GPU writes the same
20.5 MB in **58 ms**, 24x faster, because it formats in bulk with `to_chars`
instead of `ostream <<` per field.  Measuring the writer on an *unpinned* run
inflates it to ~2.7 s, because the write-back then competes with 16 compute
threads for the E-cores; that is a measurement artefact, not the writer's cost.

## 7. Measurement conventions

Three different "ms/step" appear in these documents and they are not
interchangeable:

| name | covers | how it is obtained |
|---|---|---|
| **GPU step** | ~10 kernels + 3 memsets + 1 sync, no result file | the `[time]` line / N |
| **GPU end-to-end** | start-up + N steps + N/50 result files | process wall clock / N |
| **CPU compute** | one step, file excluded | `T(100) - T(51)` / 49, which cancels the file |
| **CPU wall** | one step including its share of the files | process wall clock / N |

The repository headlines compare CPU wall against GPU step (33x in
`perf-work-hrb-jian.md`, from 163 ms against 5.01 ms).  That is the established
convention here and §1 reproduces it in its third row, but both honest pairings
are also given: compute against compute (42.4x) and wall against wall (33.6x).

One number needed correcting while preparing this: the shipped `README.txt` quotes
the CPU reference as `80.69 s / 161 ms/step`, and at the time this measurement
started the repository still carried `105.6 s / 211 ms/step` as the baseline.  The
161 figure is the right one — it matches the 163 ms/step that
`perf-work-hrb-jian.md` §10 now records as the reproducible 9950X number, and the
211 figure has since been retracted there.  Nothing in this document depends on
the old value.

## 8. Correctness on this machine

The port was re-validated on KMG-XI rather than assumed to be
architecture-independent.

| check | result |
|---|---|
| fp64 run-to-run, 14 columns x 10 files | **all zero** (bit-identical) |
| fp64 vs CPU `double` reference (32-thread run) | `x0 x1 x2 / rho / mass / itype / X00 X01 X02 / rho0` **0 differing values**; `p` maxAbs **1e-3**; `vx` maxAbs **1e-9** |
| fp64 vs the previous build `355b24d` | `x/rho/mass/itype/X0/rho0` identical; `p` differs on **8 particles** out of 237 298 x 10, maxAbs 1e-3 — the summation-order change P1 and ELL are expected to cause |
| fp32 vs fp64 | `x` <= 1e-6 m, `rho` <= 0.01 kg/m3, `p` maxAbs 0.2 Pa (limit 1 Pa), `vx` <= 1e-5 m/s |
| result file sizes vs `baseline-fp64` | **10 / 10 identical** |

That last row is weaker than it looks — identical file sizes are consistent with
identical content but do not prove it.  **The official
`CompareResults . baseline-fp64` run was not possible here:** the 196 MB of
baseline files arrive as OneDrive cloud placeholders and cannot be hydrated;
reads block indefinitely and then fail with `The cloud sync provider failed to
validate the downloaded data`.  The CPU cross-check above exercises the same
acceptance criterion, but the byte-level comparison against the RTX 5080
baseline is still outstanding.

Also not done: `cuda/tools/check_rigid_translation.py`, because this machine has
no Python installation (only the Microsoft Store stub).  The C1 fix is unverified
here beyond the fact that the shipped list is the expected 4 844 544 entries.

## 9. Reproducing

```bat
:: GPU, both precisions
set SPH_MAXSTEP=500
sph_cuda.exe --fp64
sph_cuda.exe --fp32
set SPH_PROFILE=1
sph_cuda.exe --fp64

:: CPU: 16 threads, and pin them to the P-cores (logical CPUs 0-15)
set OMP_NUM_THREADS=16
set SPH_MAXSTEP=500
start /affinity FFFF SPH.exe

:: compare (the CPU run must be its own directory)
tools\CompareResults.exe <gpu-dir> <cpu-dir>
```

The measurements above separate the components as follows: `SPH_MAXSTEP=0` gives
start-up alone (no step, no file); `SPH_MAXSTEP=51` against `SPH_MAXSTEP=100`
gives a step time with the file cost cancelled, because both write the same
number of files; and the difference between an unpinned and a pinned run of the
same 500 steps isolates the placement effect from the work.
