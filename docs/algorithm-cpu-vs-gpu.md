# The same SPH solver twice: CPU reference vs CUDA port

Why the two implementations compute a step differently, what each one actually
spends its time on, and where the 20–40x comes from.

| | |
|---|---|
| Case | `input.dat`: 237 298 particles (101 250 fluid, 60 320 solid, 75 728 wall), `dt = 1e-5`, 11.10 M neighbour pairs at t = 0 |
| Code | `cpu/1.cpp` (reference) and `cuda/src` (port), as of `98fcf58` + the doc commits |
| Machines | RTX 5080 / Ryzen 9 9950X, RTX 4090 / i9-14900KF, RTX 5090 / i9-14900K |
| Measurements | [`perf-work-hrb-jian.md`](perf-work-hrb-jian.md) (5080), [`perf-work-tys-jian.md`](perf-work-tys-jian.md) (4090), [`perf-work-kmg-xi.md`](perf-work-kmg-xi.md) (5090) |
| Design detail | [`cuda/docs/DESIGN.md`](../cuda/docs/DESIGN.md) |

---

## 1. The scheme both programs implement

Weakly compressible SPH with a Riemann-solver pair interaction, a
total-Lagrangian solid with Kelvin–Voigt friction, and a mirror-Riemann coupling
at the fluid/solid interface.  Per step, identically in both:

1. find the neighbour pairs (everything within `3*hsml = 0.03 m`);
2. **pass 1**, per pair: continuity and momentum for the fluid (`drhodt`,
   `dvxdt`, `viscos`, `SurF`), and the deformation gradient `F` plus its rate
   `DFdt` for solid/solid pairs;
3. solid constitution, per solid: `J = det F`, `rho = rho0/J`, the stress `Pia`
   (elastic + Kelvin–Voigt damping);
4. **pass 2**, per pair involving a solid: `PIJW` (solid/solid) and `SForce`
   (fluid/solid, both sides);
5. integrate: fluid `rho`, `v`, solid `v` (with a boundary clamp), then `x +=
   dt*v`.

The pair terms are symmetric in the sense the port exploits: the contribution of
a pair to the two particles is the same number with opposite sign, and the only
asymmetric ingredient (`eta`, from the viscosity switch) is a function of a
*state* array indexed by `max(p,q)`, so either side can read it.

---

## 2. The CPU reference: principles and implementation

### 2.1 Data layout

One fixed-size global array per scalar, `double`, component-major:

```
x[3][num]  vx[3][num]  X0[3][num]  rho[num]  p[num]  mass[num]  itype[num] ...
A[9][num]  F[9][num]  DFdt[9][num]  Pia[9][num]        (nine arrays each)
```

69 doubles per particle = 552 B → 131 MB of state.  237 298 particles is a
compile-time constant (`num`), which is why the deck is fixed-width.

### 2.2 Neighbour search

A fixed uniform grid over `[-0.10,0.70]x[-0.10,0.70]x[-0.10,1.20]`, cell edge
`3*hsml = 0.03`, 27x27x44 = 32 076 cells (7.4 particles per cell).  Each step:

```
count per cell -> prefix sum -> scatter particle ids into `sorted` (atomics)
-> insertion-sort each cell by particle id  (deterministic order)
-> enumerate the 14 half-neighbour cells per cell, in parallel, into pair_i/pair_j
```

The pair list is materialised as two `int` arrays; with a capacity of 14 M
entries that is **112 MB written and then read every step**, on top of the
11.1 M distance tests.  (The first version of this function rebuilt an octree of
200 k nodes per step and measured 1.16 s/step; the uniform grid replaced it.)

### 2.3 One step, and where the time goes

`direct_find()` then `continue_density()` then the integration loops.

`continue_density()` is a **pair-centric scatter**: 32 OpenMP threads walk
chunks of the pair list, and each visit updates *both* particles.  Because both
sides of a pair are touched by an arbitrary thread, every accumulator written by
the pair loop is replicated per thread in case two threads own the same pair
neighbourhood:

```
35 arrays x 237 298 particles x threads of double
= 35 x 237 298 x 8 B = 66 MB per thread = 2.03 GB at 32 threads
```

The step then zeroes the ranges each thread can write (563 MB, after an
earlier optimisation that stopped zeroing the whole 2.03 GB) and sums the
buffers back into the globals (a second 563 MB of traffic).  Measured phase split
on an i9-14900KF, 500 steps, 16 threads pinned to the P-cores:

| phase | s / 500 steps | ms/step | share |
|---|---:|---:|---:|
| `direct_find` (search + pair list) | 9.41 | 18.8 | 22 % |
| `continue_density` (physics, both passes + reductions) | 32.60 | 65.2 | 77 % |
| integration | 0.37 | 0.74 | 1 % |
| **compute total** | **42.38** | **84.8** | |

(The full step also pays ~540 ms per result file, because every field goes
through `ostream <<`.)

### 2.4 Why it is written this way

It is the correctness oracle: a single translation unit, double precision,
plain serial semantics, no symmetry assumptions that could hide a bug.  The
per-thread buffers keep every pair's two contributions in one place and make the
result independent of which thread owns which pair — at the cost of the memory
above.

---

## 3. The CUDA port: principles and implementation

### 3.1 The one structural idea: gather, not scatter

Because a pair's two contributions are equal and opposite, the port inverts the
loop: **one thread per particle**, and that thread walks its own neighbourhood
and adds in its own half of every pair.  Then

* no accumulator is shared between threads → **no per-thread buffers, no
  reductions, no atomics on the physics**;
* each particle writes its own 3-vector or scalar once, at the end of the
  kernel, from registers;
* the two sides of a pair are computed by two different threads, each reading
  the other's state — which is what a GPU wants: many independent readers.

This is not an optimisation of the CPU code; it is a different loop structure.
Everything after it follows.

### 3.2 Precision and layout

One templated implementation (`SphPtrsT<Real>`, `Real = double | float`) with
both instantiated in the same binary; `--fp64` / `--fp32` selects at run time.
Tensors are still component-major (`M[9][num]`), but `Pia` is stored
array-of-structs (12 doubles per solid, 16-byte aligned) so a neighbour's 3x3
stress arrives in five 16-byte loads instead of nine scattered ones.

fp32 is not a straight recompile: `x`, `rho` and the solid displacement use
compensated sums (a `dt*v` of 1e-9 m is below the ulp of a 0.5 m coordinate), and
the solid accumulates `F - I` from displacements instead of `F` from absolute
positions, so the float mantissa is spent on the strain rather than on the
coordinate.

### 3.3 Neighbour search

Same grid, same cell walk, one thread per particle, but only *candidates* are
generated; membership is decided by one exact test:

```
27 cells -> float distance estimate (A4) rejects r2 >= r2max*(1+1e-4)
         -> exact test with __dadd_rn/__dmul_rn inside
```

The rounded intrinsics exist because the lattice puts thousands of pairs exactly
at `r = 3*hsml`; without them the pair set depends on whether the compiler
contracts `a*a + b*b` into an FMA, and the fp64 count shifts by 1 080 pairs.

The cell list is rebuilt every step (3 small kernels + an insertion sort per
cell), 0.13 ms.

### 3.4 Kernel map, and what each costs

One thread per particle, so a particle's whole state stays in registers.

| kernel | what it does | fp64 ms (5080) |
|---|---|---:|
| `prepare` | `SurFD := SurF`, `p := c0^2(rho-1000)` | 0.01 |
| cell hist / scan / scatter / sort | neighbour grid | 0.13 |
| `pass1_fluid` | fluid pair terms (two-phase, sec. 3.6) | 3.53 |
| `pass1_solid` | `F`, `DFdt` from the static list (sec. 3.5) | 0.34 |
| `constitution` | `rho`, `Pia` per solid, in double for both precisions | 0.05 |
| `pass2_fluid` | fluid side of `SForce` | 0.13 |
| `pass2_solid` | `PIJW` and the solid side of `SForce` | 0.62 |
| update x3 | `rho`/`v` for fluid, `v` for solid, `x` for all | 0.03 |
| **sum** | | **4.84** |
| **measured step** | | **5.00** |

State is 131 MB (fp64) plus 402 MB of static solid pairs plus 78 MB of fluid
neighbour lists: ~630 MB, against 2.03 GB of CPU scratch.

### 3.5 Solid/solid pairs: a constant pair set (A3)

The solid is total-Lagrangian: the kernel weight is evaluated at the **reference**
positions `X0` and `A` is a constant.  The solid/solid pair set and every
per-pair weight are therefore constant for the whole run, so they are computed
**once**, at start-up, and stored:

```
pass1:  F    += (x_q - x_p) (x) B        DFdt += (v_q - v_p) (x) G
pass2:  u    += (rho0_q/rho_q) G          t += (rho0_q/rho_q) Pia_q D
```

with `G = w0 gradW`, `B = w0 A_p gradW`, `D = w0 A_q (-gradW)` precomputed.  No
distance, no square root, no kernel evaluation per step: 4.84 M pairs in
0.34 ms.  The list is slot-major (ELL: entry `j` of particle `k` at
`j*SOLNUM + k`, nine separate component arrays) so a warp's loads are
contiguous.

The pair set must be a function of `X0` alone, which is why the cell list it is
searched in is binned from `X0`, not from the current `x` (see
[`edit-2026-11-19-12-19.md`](edit-2026-11-19-12-19.md) sec. 3).

### 3.6 Fluids are the exception: a recorded neighbour list (P1)

Fluid moves, so no static list.  But the fused kernel was dominated by
divergence: a warp walks 32 different neighbourhoods, and only ~15 % of the
candidates it visits are hits, so the ~60-instruction FP64 body runs whenever
*any* lane has a hit.  Instead, fp64 builds the neighbour lists first
(`fnbr[j*NTOTAL + p]`, 192 slots, 78 MB) and then runs the same body over a
dense list.  `pass1_fluid` 8.77 → 3.54 ms.  fp32 keeps the fused kernel, where
the body is cheap and the traversal dominates.

### 3.7 The arithmetic is factored (A5)

Four identities collapse the per-visit work: the kernel derivative polynomial in
4 multiplies, `gradW.d` without component products, `UL-UR` without the absolute
velocities, and one division instead of three.  About 60 FP64 ops per visit
instead of 90.

---

## 4. The difference in principle

| | CPU reference | CUDA port |
|---|---|---|
| loop structure | **pair-centric scatter**: thread walks pairs, updates both sides | **particle-centric gather**: thread walks its own neighbours, updates itself |
| pair list | materialised every step, 112 MB of `int`, both directions | never materialised for solids (static, 4.84 M, computed once); recorded per particle for fp64 fluid |
| accumulators | 35 arrays x num x threads = **2.03 GB**, zeroed and reduced every step | **registers**; one write per particle |
| synchronisation | 2 barrier-protected reduction passes per step | none; kernels are separated by data dependency only |
| thread count | 32 | 237 298 (one per particle) |
| work per step | 11.1 M pair visits, each doing both sides | 10.5 M fluid visits + 4.84 M solid pairs, each doing one side |
| solid pair work | distance test + kernel + gradient every step | table lookup: the geometry is constant |
| precision | double only | fp64 and fp32, same source |
| memory footprint | 131 MB state + 112 MB pair list + 2.03 GB scratch | 630 MB, no scratch |

Two of these are *algorithmic* (the gather reformulation and the constant solid
pair set) and two are *machine* consequences (thread count, no barriers).  The
rest of the port is bookkeeping to make the gather structure efficient:
coalesced layouts, the float pre-filter, factored arithmetic.

---

## 5. Why the GPU is 20–40x faster

### 5.1 It is not a faster scalar machine

On these GeForce parts FP64 runs at about **1/64** of the FP32 rate.  A 5080's
FP64 peak is the same order of magnitude as a 16-core Zen 5 running scalar
double precision, so the speed-up cannot be FP64 throughput.  The measured work
rates say where it is:

| | what is timed | pair terms | time | rate |
|---|---|---:|---:|---:|
| 9950X, 32 threads | whole compute step | 11.1 M | 131 ms | 85 M/s |
| 14900KF, 16 threads | `continue_density` only | 11.1 M | 65.2 ms | 170 M/s |
| 5080, fp64 | `pass1_fluid` body only | 10.5 M | 2.67 ms | **3.93 G/s** |
| 5080, fp64 | `pass1_solid` only (static list) | 4.84 M | 0.34 ms | 14.2 G/s |

The GPU's fluid body alone runs at 23x the CPU's whole physics phase per pair
term, and 46x the CPU's whole-step rate.  The speed-up is in the *structure*,
not in the clock.

### 5.2 The three mechanisms, in order of size

**(a) The CPU never uses its cores; the GPU can.**  The CPU gains only **7.1x**
from 16 cores (934 ms/step at one thread vs 131 at 16) and **nothing** from
32 threads (131.0) — it is saturated on memory bandwidth, and on hybrid Intel
parts 32 threads are *slower* than 16 pinned P-cores (195.9 vs 138.4 ms/step on
the 14900K).  The GPU runs 237 298 threads and hides the latency of every
neighbour load behind other warps rather than behind caches.  If the CPU scaled
perfectly, its step would be 29 ms; it is 131.

**(b) The scratch buffers cost more than the physics.**  Per step the CPU zeroes
563 MB of accumulator ranges, writes into them at scattered indices from 11.1 M
pair visits, then reads 563 MB back to reduce.  That is roughly 1–3 GB of
traffic per step, i.e. 20–60 ms of the 131.  The GPU has no such buffers: the
accumulators live in registers, and the only per-step traffic is the state it
actually needs (`p`, `rho`, `vx`, ...) plus the neighbour structure.  The review
put the reduction passes alone at ~22 % of the CPU step, which matches.

**(c) Some of the work disappears entirely.**  Solid/solid pairs are a constant
table (A3): the CPU recomputes distance, kernel, gradient and both gradient
correction matrices for all 4.84 M solid pairs every step; the GPU reads 52 B
per pair from a precomputed list.  That kernel is now bandwidth bound (754 of 1008 GB/s peak
on the 4090) — it has nothing left to optimise.  A5's factoring cuts the fluid
pair body from ~90 to ~60 FP64 operations (measured 11.18 → 8.78 ms on the fused
kernel), and A4's float pre-filter drops the exact distance tests from 69.6 M to
10.7 M per step.

### 5.3 The ratio is mostly a statement about the CPU

Same GPU code, three machines:

| machine | CPU compute, ms/step | GPU fp64, ms/step | ratio |
|---|---:|---:|---:|
| RTX 5080 + 9950X, 32 threads | 131 | 5.006 | **26x** |
| RTX 4090 + i9-14900KF, 16 pinned | 84.8 | 4.14 | **20.5x** |
| RTX 5090 + i9-14900K, 16 pinned | 138.4 | 3.265 | **42.4x** |

The GPU varies by 1.53x across three architectures; the CPU ratio varies by
2.1x.  Quoting a single speed-up without the CPU configuration is therefore
meaningless, which is why the repository headlines state "vs the 32-thread CPU"
and which machine.

---

## 6. Performance across machines

### 6.1 GPU (same build, three architectures)

| | fp64 step | fp32 step | vs 5080 |
|---|---:|---:|---:|
| RTX 4090 / sm_89 / CUDA 12.2 / GCC 11 | 4.14 | 1.28 | 1.21x |
| RTX 5080 / sm_120 / CUDA 13.3 / MSVC | 5.006 | 1.634 | 1.00 |
| RTX 5090 / sm_120 / CUDA 13.3 / MSVC | **3.265** | **1.157** | 1.53x |

fp32 is 2.8–3.2x faster than fp64 on the GPU even though the hardware's fp32
rate is ~64x the fp64 rate: at 1.6 ms the fp32 step is limited by the neighbour
traversal (0.56 ms of it is pure traversal), not by arithmetic.

### 6.2 CPU (three configurations of two CPUs)

| machine | threads | compute, ms/step | 500-step wall | note |
|---|---:|---:|---:|---|
| 9950X (16C/32T), Windows | 32 | 131 | 163 | no gain over 16 threads |
| 9950X, Windows | 16 | 132 | 164 | |
| 9950X, Windows | 8 | 196 | 228 | |
| 9950X, Windows | 1 | 934 | — | 7.1x at 16 threads, 22 % efficiency |
| i9-14900KF, Linux | 16 (P-cores) | 84.8 | 97 | 77 % of it in `continue_density` |
| i9-14900KF, Linux | 32 (all) | 77.3 | 89 | |
| i9-14900K, Windows | 16 (P-cores) | 138.4 | 169 | pinning worth 1.20x compute |
| i9-14900K, Windows | 32 (all) | 195.9 | 249 | E-cores *slow it down* |

### 6.3 End-to-end, not just the step

At the shipped 50-step output cadence, output is a real fraction of the wall
clock on both sides:

| | compute, ms/step | output per file | output share of wall |
|---|---:|---:|---:|
| GPU fp64, 5080 | 5.0 | 56 ms | 18 % |
| GPU fp32, 5080 | 1.6 | 51 ms | 37 % |
| GPU fp32, 4090 | 1.28 | 66 ms | 47 % |
| CPU, 9950X | 131 | 1.47 s | 18 % |

---

## 7. The CPU's bottlenecks, concretely

| bottleneck | evidence | cost |
|---|---|---|
| **Per-thread accumulation buffers** | 35 arrays x num x threads = 2.03 GB at 32 threads; 563 MB zeroed and 563 MB reduced per step, plus scattered read-modify-write from every pair visit | the review attributes ~22 % of the step to the reductions alone; the buffer traffic is the reason the step does not scale |
| **No scaling past ~16 threads** | 934 ms (1 T) → 196 (8 T) → 132 (16 T) → 131 (32 T) | 7.1x from 16 cores: 22 % parallel efficiency |
| **Hybrid CPUs make it worse** | 14900K: 138.4 ms pinned to P-cores vs 195.9 at 32 threads; the E-cores invert sign between the Windows and Linux runs | up to 1.4x lost by "using all the cores" |
| **Pair list materialised every step** | 11.1 M pairs → 112 MB of `int` written and read back, plus the per-cell count / prefix / scatter / sort | 18.8 ms/step of the 84.8 on the 14900KF (it was 1.16 s with the original octree) |
| **Recomputed geometry** | solid/solid distance, kernel, gradient and `A` products recomputed for 4.84 M pairs every step, although they are constant | removed entirely by A3 on the GPU |
| **Redundant kernel evaluations** | `dwdx()` called ~30x per fluid/fluid pair, each with a division and the full polynomial | A5's factoring took the GPU from 96 to 60 FP64 ops per visit |
| **Serial phases** | the solid constitution is per-particle but runs between the two parallel passes; `SurFD`/`p` are updated serially | 60 320 particles of ~300 double operations, plus two barriers per step |
| **Writer** | 540 ms per 20.5 MB file, every field through `ostream <<` | 12 % of the CPU wall clock |
| **No input validation** | a missing `input.dat` leaves every array zeroed: all particles in one cell, O(n²) search, negative pair count, exit code 0 | looks like a 200x slowdown instead of an error |

---

## 8. What to take from this

* The port is faster because it **reformulated the loop** (gather), **removed a
  constant-cost table's worth of work** (static solid pairs) and **stopped moving
  2 GB of scratch per step** — in that order of importance.  "GPU = more cores"
  explains the least.
* Anything measured against the CPU must state the CPU configuration; the same
  GPU is 20.5x or 42.4x depending on the host.
* The CPU tree is kept as a validation oracle, deliberately unoptimised.  Its own
  acceleration options (gather formulation, hoisting `1/dr`, static solid list,
  `-O3`) are listed in [`perf-work-hrb-jian.md`](perf-work-hrb-jian.md) sec. 10
  and are not applied, because the reference must stay reproducible: even the
  pair set depends on the compiler's FMA policy.
* On the GPU the remaining cost is no longer arithmetic: after P1/P2 the largest
  single block is the result writer (18 % of wall in fp64, up to 47 % in fp32),
  then the per-step launch overhead (~0.17 ms on Windows/WDDM, ~0 on Linux).
