# SPH on CUDA  -  design and porting notes

A GPU port of the three-dimensional smoothed-particle-hydrodynamics fluid-structure
interaction solver that lives in `1.cpp` of the original package.  The solver is
templated on the scalar type, so the same code runs in double (default) or single
precision.

Everything below is measured on the machine used for the port:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, `sm_120`, 84 SMs, 16 GB, driver 610.88 |
| CUDA | 13.3 (nvcc 13.3.73) |
| Host | MSVC 14.51 (Visual Studio 2026 / 18.7), CMake 4.2.1, Windows 11 26200 |
| CPU (reference baseline) | Ryzen 9 9950X, 16C/32T, 253 GB RAM |

---

## 1. Summary

| | |
|---|---|
| Particles | 237 298 = 101 250 fluid (`itype=2`) + 60 320 solid (`itype=-20`) + 75 728 wall (`itype=-2`) |
| Neighbour pairs | 11 102 707 at t=0 (~94 per particle), 3x3x3 cells of edge 0.03 m over a fixed box |
| Reference (CPU, 32 threads) | 500 steps in 81.4 s -> **163 ms/step** (see sec. 6.4) |
| Port, fp64 | **5.01 ms/step** -> 33x vs the 32-thread CPU |
| Port, fp32 | **1.64 ms/step** -> 100x vs the 32-thread CPU |
| Validation | intermediate arrays agree to 1e-15...1e-9 relative at step 1; 500- and 3000-step output within the CPU's own run-to-run noise |

---

## 2. The reference solver (`1.cpp`)

### 2.1 Algorithm

Weakly compressible SPH whose pair interactions are evaluated through an
**approximate Riemann solver** (Riemann-SPH style), plus a total-Lagrangian SPH
solid and a mirror-Riemann fluid-structure coupling.

* Kernel: **quintic spline**, `h = 1.5*hsml`, support `2h = 3*hsml = 0.03 m`:

  `W(q) = 21/(256 pi h^3) * (1+2q)(2-q)^4`, `q = r/h in [0,2]`

* Equation of state (fluid, **linear**): `p = c0^2(rho - 1000)`, `c0 = 48 m/s`,
  clamped to `p = 0` for `rho < 1000`.
* Continuity: `drho_i/dt = -2 rho_i (m_j/rho_j) (U_L - U*) * dW/dr`
* Momentum: `dv_i/dt = -2 m_j (P*/(rho_i rho_j)) gradW`
* Riemann state:

  `U* = 1/2(U_L+U_R) + 1/2(p_L-p_R)/(1/2(rho_L+rho_R)c0)`

  `P* = 1/2(p_L+p_R) + 1/2(1/2(rho_L+rho_R))c0(U_L-U_R)*min(3*max(U_L-U_R,0)/c0, 1)`

* Viscosity: Morris-type Laplacian, `2 m_j eta S (v_i-v_j)/(rho_j(r^2+0.01h^2))`,
  `S = gradW*r`; `eta` switches between `2e-5` and `1e-3` (50x) depending on `|SurFD|`.
* Solid: total Lagrangian.  `F` from the reference configuration with a fixed
  gradient-correction matrix `A` (computed once by `GraCorrec`), `J = det F`,
  `rho = rho0/J`, and

  `P = mu(F - F^-^T) + lambda*log10(J)*F^-^T + P_D`

  where `P_D` is a Kelvin-Voigt-like friction term.  Note **log10**, not ln.
* Fluid/solid and fluid/wall pairs use the solid's surface normal `vecn`
  (computed once at t=0) to build a mirror velocity and then the same Riemann
  solver; the resulting `SForce` is applied to both sides.
* Time integration: explicit Euler, fixed `dt = 1e-5`, velocity capped at 30 m/s.

### 2.2 Data layout and sizes

Component-major SoA: `double x[3][num]`, `vx[3][num]`, `X0[3][num]`, `vecn[3][num]`,
`PIJW[3][num]`, `SForce[3][num]`, `dvxdt[3][num]`, `viscos[3][num]`, plus scalars
(`rho, rho0, mass, p, itype, SurF, SurFD, drhodt`) and four 3x3 tensors stored as
nine separate arrays (`A, F, DFdt, Pia`).

69 doubles per particle = 552 B -> 131 MB of state, plus the OpenMP accumulation
buffer `nth x 35 x num` doubles = **2.03 GB at 32 threads**.

### 2.3 One time step

```
1. direct_find()        rebuild the uniform cell list, enumerate the pair list
2. continue_density()
   2.0 serial:  SurFD := SurF;  p := c0^2(rho-1000)  (clamped at 0)
   2.1 pass1 over all pairs:
         fluid/fluid        -> drhodt, dvxdt, viscos, SurF
         fluid/(solid|wall) -> same, using the neighbour's normal
         solid/solid        -> F, DFdt
   2.2 reduce the 32 thread-private buffers back to the globals
   2.3 solid constitution (per solid): DFdtB, J, rho = rho0/J, SD, PD, F^-^-1, Pia
   2.4 pass2 over pairs that contain a solid:
         solid/solid -> PIJW ; fluid/solid -> SForce on both sides
   2.5 reduce again
3. integrate: fluid rho and v (with gravity and a speed cap), solid v (with the
   boundary clamp), then x += dt*v for every particle
4. every 50 steps write result<N>.dat (237 298 lines x 14 columns)
```

### 2.4 Initialisation (once)

* `init_grid()`  -  fixed box `[-0.10,0.70]x[-0.10,0.70]x[-0.10,1.20]`, cell edge
  `3*hsml = 0.03` -> 27x27x44 = 32 076 cells (7.4 particles per cell).
* `NorVector()`  -  one neighbour pass builds `vecn` (surface normals) for the
  solid/wall particles.
* `GraCorrec()`  -  one neighbour pass over solid/solid pairs builds the gradient
  correction matrix `a`, then inverts it into `A`.

### 2.5 Behaviours that had to be reproduced exactly

These are the places where a "reasonable" rewrite silently changes the physics.

| # | Source | Behaviour |
|---|---|---|
| 1 | `eta` | two consecutive `if`s, the second overwrites the first -> **both** sides of a pair use `SurFD[j]` (j = the higher-indexed particle) |
| 2 | `SurF` | assigned every step from the pair sum (not accumulated); it is the "surface indicator" that drives `eta` |
| 3 | `PRii` in pass2 | contains `PIJW[i]`, which is never written for fluid particles -> always 0 |
| 4 | `AveV`, `DisGrad*`, `GLStrain*` | computed but never read (`GLStrain()` is commented out) -> dropped |
| 5 | `dr` epsilons | pass1 adds `1e-13`, pass2 none, `NorVector` adds `1e-12` |
| 6 | pair ordering | pairs are emitted with i<j, but every pair term is symmetric, so a gather is equivalent |
| 7 | type tests | fluid `itype>0`, solid `itype<-10`, wall `itype=-2` (not a solid) |
| 8 | output `p` | fluid: `p[i]`; otherwise `Pia11+Pia22+Pia33` (wall -> 0) |
| 9 | `paii` | `0.25*rho0*1000*hsml`  -  the 1000 is a literal, not `rho0` |
| 10 | solid density | overwritten by `rho0/J`; `abs(J)<1e-6` -> 1000, but `log10(J)` is still evaluated |

---

## 3. Port design

### 3.1 Gather instead of the pair loop

The CPU code walks the pair list once per pass and updates **both** particles of
each pair.  On a GPU that needs either ~500 M atomics per step or 32 private
buffers plus a reduction  -  and the reduction alone costs 22 % of the CPU runtime.

Every pair term in this solver is symmetric in (i,j) once you notice that
`e_j = -e_i`, `dwdx` is odd in its first argument and `UL-UR` is the same seen
from either side.  So each particle can accumulate **its own half** of every
interaction, and no atomics or reductions are needed at all:

| quantity | gather form used here |
|---|---|
| `drhodt` | `-2rho_p(m_q/rho_q)(U_L-U_momentum)dwdr(dr)` |
| `dvxdt` | `-2m_q(P*/(rho_prho_q))gradW(x_p-x_q)` |
| `viscos` | `2m_qetaS(v_p-v_q)/(rho_q(r^2+0.01h^2))` |
| `SurF` | `S*m_q/rho_q` |
| `F`, `DFdt` | `(m_q/rho0_q)(x_q-x_p)(x)(A_p*g0_pq)` and `(m_q/rho0_q)(v_q-v_p)(x)g0_pq` |
| `PIJW` | `(m_q/(rho_prho_q))[Pia_p*(A_p g0_pq) - Pia_q*(A_q g0_qp)]` |
| `SForce` | each side uses its own coefficient and the other side's normal |

The only asymmetric quantity is `eta` (item 1 above), which needs
`SurFD[max(p,q)]`  -  this was the one real functional bug of the first version
(sec.7.1).

### 3.2 Kernel map (one thread per particle)

| kernel | threads | work |
|---|---|---|
| `k_prepare` | 0...NUM | `SurFD := SurF`, `p`, `irho = 1/rho` |
| `k_cell_hist` / `k_cell_scan` / `k_cell_scatter` | 0...NUM / 1 block / 0...NUM | counting sort of the particles into the fixed grid; `cellSolid` / `cellFluid` counts come out of the histogram |
| `k_copy_xf` | 0...NUM | float copy of `x` for the A4 pre-filter (once; `k_update_pos` keeps it current) |
| `k_pass1_fluid_fast` | 0...NTOTAL | fluid forces, factored inner loop (A5; `k_pass1_fluid` is the literal version) |
| `k_pass1_solid_static` | NTOTAL...NALL | `F`, `DFdt` from the static solid pair list (A3; `k_pass1_solid` walks the grid) |
| `k_constitution` | NTOTAL...NALL | 3x3 constitutive update, `rho = rho0/J`, `Pia` |
| `k_pass2_fluid` | 0...NTOTAL | `SForce` from solid neighbours |
| `k_pass2_solid_static` | NTOTAL...NALL | `PIJW` from the static list, `SForce` from fluid neighbours (A3) |
| `k_update_fluid` / `k_update_solid` / `k_update_pos` | ranges | integration |
| `k_nor_sumvecn`, `k_nor_normalize`, `k_gra_correc`, `k_gra_invert`, `k_build_solid_pack` | once | initialisation |
| `k_solid_pair_count`, `k_solid_pair_fill` | once | A3 pair list, two passes over the reference configuration |

### 3.3 Neighbour search

Same fixed grid, same domain, same cell size, same clamping as `direct_find`.
Each particle scans the 27 cells around its own cell and filters `r^2 < (3hsml)^2`.
The membership test uses `__dadd_rn`/`__dmul_rn` instead of ordinary `+`/`*`, so
the compiler cannot fuse them into an FMA: the lattice puts a large number of
pairs exactly on `r = 3hsml` and without this the fp64 pair count differed from
`direct_find()` by ~0.01 %.

`sph_neigh2()` is the ILP variant: two neighbours per loop iteration, same
neighbour set and same accumulation order (sec.6.3).

Since A4 (sec.11) each candidate is first rejected by a **float** estimate of
`r^2` against `r2max*(1+1e-4)`; only the survivors take the exact test above,
which alone decides membership.  In the fp32 build the pre-filter is compiled out
entirely (`if constexpr`), because there the exact test is already fp32.

### 3.4 Memory

| allocation | size (fp64) | size (fp32) |
|---|---|---|
| state (69 Reals/particle) | 131 MB | 65 MB |
| `solidPack`  -  `{X0[3], A[9], rho0, mass}` per solid, built once | 6.8 MB | 3.4 MB |
| `Pia`  -  12 Reals per solid (9 used + 3 pad), AoS, 16-B aligned | 5.8 MB | 2.9 MB |
| A3 pair list: 4 844 544 pairs, ELL width 92 x (9 Reals + 1 int) | 402 MB | 212 MB |
| `xf`  -  float copy of the positions (A4) | 2.8 MB | 2.8 MB |
| cell list + sorted indices | 1.1 MB | 1.1 MB |
| P1 fluid neighbour list (fp64 only): 192 slots x 101 250 | 78 MB | - |
| **total** | **~630 MB** (versus 2.03 GB of OpenMP buffers on the CPU) | **~285 MB** |

The CPU's 2 GB per-thread accumulation buffer disappears completely because
each particle's accumulators are private to its own thread.

The A3 list is the one allocation that grew; 402 MB of 15.9 GB is not a
constraint here, and the fp32 build needs 212 MB of it.  The list can be halved
by dropping `sB` (see sec.12.4).

### 3.5 Output

`result<N>.dat` keeps the reference column layout (14 columns, tab separated,
`%.6g`).  All values are converted to `double` before formatting, so the fp32 and
fp64 builds produce files that the existing comparison tool can read unchanged.

Since A2 the text is produced by `std::to_chars(..., general, 6)` instead of
`snprintf("%.6g")` and in up to 8 contiguous chunks formatted on worker threads,
written in order.  The result is byte-identical to the old single-threaded
`snprintf` output (verified on all ten files, both precisions) and about five
times faster: 0.34 s -> 0.07 s per file.

---

## 4. Precision template

The solver is a template on `Real`; both instantiations are compiled into the
same executable and selected at start-up.

```bat
sph_cuda.exe                 :: fp64 (default)
sph_cuda.exe --fp32          :: fp32
set SPH_PRECISION=fp32       :: same, via the environment
```

CMake side: `SPH_ENABLE_FP64`, `SPH_ENABLE_FP32` (both ON by default; switching
one off halves the build time) and `SPH_DEFAULT_PRECISION`.

Two implementation details matter:

1. **Every floating-point literal is written `Real(x)`.**  Otherwise `0.5*(a+b)`
   would silently promote parts of the fp32 path to double, and FP64 on this GPU
   is 1/67 of FP32  -  one promotion eats the whole speed-up.  `cuobjdump -sass`
   on an fp32-only build confirms there is no double arithmetic left.
2. **Neighbour-list rounding stays explicit** (`sph_add_rn`/`sph_mul_rn`), which
   for fp32 maps to `__fadd_rn`/`__fmul_rn`.

---

## 5. Validation

### 5.1 Method

Three independent checks, all against the **pristine** reference package:

1. **Array-by-array**: both implementations dump `F`, `DFdt`, `rho`, `Pia`,
   `SForce`, `PIJW`, `drhodt`, `dvxdt`, `viscos`, `SurF`, `vecn`, `a`, `A` after
   one step; compare in binary (not through the 6-digit text output).
2. **Trajectory**: compare `result0..9.dat` with `tools/CompareResults`.
3. **Sanitisers**: `compute-sanitizer` `memcheck`, `initcheck`, `racecheck`,
   `synccheck`.

### 5.2 Reference case (500 steps)

| column | differing values | maxAbs | CPU's own rerun-to-rerun noise |
|---|---|---|---|
| `x0 x1 x2` | **0** | **0** | 0 |
| `rho`, `mass`, `itype`, `X00..X02`, `rho0` | **0** | **0** | 0 |
| `p` | 81 155 | **1e-3** | 0.01 |
| `vx0 / vx1 / vx2` | 245 / 1 372 / 13 | **1e-9** | 1e-9 |

Every column sits inside the CPU's own reproducibility, and `p` is one order of
magnitude tighter.  Bulk quantities over the same 500 steps: kinetic energy
0.02 %, mean pressure 0.05 %, maximum pressure identical to five digits, mean
plate displacement 0.0000 %.

### 5.3 Intermediate arrays (1 step)

| array | maxAbs | note |
|---|---|---|
| `F11` | 2.2e-15 | solid deformation gradient |
| `DFdt11` | 2.0e-17 | |
| `rho` (solid) | 3.0e-12 | |
| `Pia11` | 2.5e-8 | stress |
| `SForce0` | 2.8e-13 | fluid/solid coupling (second case) |
| `PIJW0` | 1.3e-9 | |
| `vecn0` | 7.5e-11 | surface normal |
| `a11 / A11` | 2.7e-15 | gradient correction |
| `drhodt` | 1.6e-11 | |
| `viscos0` | 4.9e-15 | |
| `SurF` | 4.9e-15 | |

These are GPU-versus-**CPU** deviations, measured before A3/A4/A5 (which move the
GPU away from the CPU by a further 1e-15/1e-12-relative; see sec.11.0 for the
optimisation-versus-optimisation numbers).

### 5.4 A second case is required

The reference case has **zero fluid/solid pairs** (the plate does not touch the
fluid; all coupling goes through walls).  That means `SForce`, `vecn` and `PRii`
never execute, and their comparison is vacuous.  A second case was therefore
built by relabelling all 75 728 wall particles as solid (`itype=-2 -> -20`),
which produces a large number of fluid/solid pairs (`SForce` ~3.7e2).  Both
implementations then agree to 1e-13 on `SForce` and to 1e-6 on the 200-step
output.  A third case (particles given a 1e6x initial velocity and a density of
500) exercises the speed cap and the density floor.

### 5.4b Post-contact validation, 3000 steps

An external review found that the 500-step reference set ends *before* the fluid
reaches the plate: the first fluid/solid pair appears at **step 566**, so nothing
in `result0..9.dat` exercises the coupling in the real geometry.

The comparison was therefore extended to 3000 steps against the pristine CPU
build (486 s CPU, 39.6 s GPU).  Result, over all 60 output files:

| column | differing values | maxAbs |
|---|---|---|
| `x0 x1 x2`, `rho`, `mass`, `itype`, `X0`, `rho0` | **0** | **0** |
| `p` | 486 257 | **0.01** |
| `vx0 / vx1 / vx2` | 3 766 / 6 482 / 420 | **1e-9 / 1e-9 / 1e-8** |

The per-file spread does not grow after contact (still ~8 300 lines and
maxAbs <= 0.01 at file 59, same as at file 1), so the coupling path
(`pass2_fluid`, the fluid branch of `pass2_solid`, the fluid/solid branch of
`pass1_fluid`) tracks the CPU as well as the uncoupled phases do.  This run was
made with A3+A4+A5 all enabled, i.e. the shipped configuration.

### 5.5 Not exercised by any case

| branch | why |
|---|---|
| `abs(Jacb) < 1e-6 -> rho = 1000` | `log10(Jacb)` is still evaluated, so both implementations produce NaN; only the NaN *locations* could be compared |
| `abs(Da) < 1e-6 -> A = I` | needs a nearly singular reference configuration |
| `min(3*dU/c0, 1)` limiter | `dU ~ 1e-2`, `c0 = 48` -> the limiter never activates |

---

## 6. Performance

### 6.1 Per-kernel profile (`SPH_BENCH=1`, 20 iterations each, fp64)

The table below is the state *before* P1/P2 (sec. 13); it is what identified
`pass1_fluid` as the bottleneck.  For the current build see sec. 6.4 and 13.

Shipped configuration (A3+A4+A5), RTX 5080:

| kernel | ms/iter | share |
|---|---|---|
| `pass1_fluid` | 8.78 | 71 % |
| `pass2_solid` | 1.15 | 9 % |
| `pass1_solid` | 1.05 | 8 % |
| cell list | 0.128 | 1 % |
| `pass2_fluid` | 0.128 | 1 % |
| constitution | 0.052 | |
| prepare + three update kernels | 0.034 | |
| **sum** | **11.32** | (12.42 ms/step measured; the rest is launch overhead) |

For comparison, the same build with the pre-optimisation paths forced on
(`SPH_SOLDDYN=1 SPH_LITERALFLUID=1`): `pass1_fluid` 11.18, `pass1_solid` 4.99,
`pass2_solid` 5.94 -> sum 17.4 ms.  fp32 (all optimisations on).  Run-to-run variation of the bench numbers is about 1 %:

| kernel | ms/iter |
|---|---|
| `pass1_fluid` | 0.760 |
| `pass2_solid` | 0.636 |
| `pass1_solid` | 0.552 |
| cell list | 0.127 |
| `pass2_fluid` | 0.105 |
| constitution | 0.062 |
| prepare + updates | 0.022 |
| **sum** | **2.26** (2.43 ms/step measured) |

fp32 is now spread thin: no single kernel dominates, so only the launch
overhead (0.17 ms, 7 %) and `pass1_fluid` are worth attacking.

### 6.2 How the bottleneck was identified

`ncu` is installed (`C:\Program Files\NVIDIA Corporation\Nsight Compute ...`) but
refuses to run: `ERR_NVGPUCTRPERM`, hardware counters need administrator rights
and this account cannot write the driver's `RmProfilingAdminOnly` key.  So the
analysis was done with controlled probes instead:

* a probe kernel that performs **the same neighbour traversal and the same 13
  loads but no arithmetic** -> 3.66 ms (fp64), 3.65 ms with the loads replaced by
  coalesced addresses.  Scattered reads cost nothing here: within a warp the
  particles are adjacent in cell order, so their neighbour lists overlap and the
  L1/L2 hit rate is high.
* therefore `pass1_fluid` = 3.7 ms of traversal + **10.6 ms of arithmetic**
  (both figures are the pre-A3/A4/A5 baseline; today the same probe reports
  2.15 ms and `pass1_fluid` is 8.78 ms).
* `cuobjdump -sass` on the innermost loop then showed why the arithmetic is slow:
  **17 `CALL.REL.NOINC` instructions**  -  every FP64 division and square root on
  this GPU is a software subroutine.

### 6.3 What worked, what did not

| change | effect |
|---|---|
| Hoist `1/dr`, `1/(rho_i+rho_j)`, `1/(dr^2+0.01h^2)`, `1/rho` (new `irho` array), compile-time `1/h` | 17 -> 4 division calls per visit, **56 -> 37 ms/step** |
| Share the kernel polynomial between `dwdr` and the three `dwdx` components | -24 FP64 ops per visit, **37 -> 31 ms** |
| `__launch_bounds__(256, 4/5/6)` occupancy sweep | 14.5 / 14.3 / 14.3 ms  -  **no effect** |
| Solid static data pack (`X0`, `A`, `rho0`, `mass`), 14 scattered loads -> 1-2 lines | `pass1_solid` 6.7 -> 6.3 ms |
| Per-cell solid flag -> skip the empty `pass2_fluid` traversal | 2.28 -> 0.44 ms |
| `Pia` AoS + `double2` vectorised loads (loads 64 -> 42) | **no gain**  -  the solid kernels are not load bound either |
| `rsqrt` instead of `1/sqrt` | **slower** (14.3 -> 15.3 ms) |
| ILP: 2 neighbours per loop iteration (`sph_neigh2`) | fp64 unchanged, **fp32 2.96 -> 2.48 ms** |
| A3 static solid pair list (sec.12) | `pass1_solid` 4.96 -> **1.05**, `pass2_solid` 5.91 -> **1.14 ms** |
| A5 factored `pass1_fluid` loop (sec.11.1) | 11.16 -> **8.78 ms** |
| A4 float pre-filter in the neighbour test (sec.11.2) | traversal probe 4.17 -> **1.76 ms**, `pass1_fluid` 9.51 -> **8.78 ms** |
| A4 also applied to the fp32 build | **slower** (+12 %, the exact test there is already fp32) -> guarded by `if constexpr` |
| A2 `std::to_chars` + chunked formatting on worker threads | 0.34 -> **0.07 s per result file**, byte-identical text |
| A6a merge the 3 x-cells of a row into one range (bit-identical) | **slower**: fp64 `pass1_fluid` 8.78 -> 9.79 ms, traversal probe 1.76 -> 1.82; reverted |

The last two lines are the interesting ones: the fp64 kernels are limited by
**FP64 pipe throughput**, not latency and not memory, which is why adding warps,
reducing loads and adding instruction-level parallelism all fail to move them.
fp32 is latency limited, so ILP pays off there.

### 6.4 Current numbers

After the external review, its two patches (sec. 7.4-7.7), the review's three
acceleration items A3/A4/A5 (sec. 11, 12) and the post-review work of sec. 13
(fp64 fluid neighbour lists, slot-major solid lists, the C1-C3 fixes).

| | ms/step | vs 32-thread CPU (163 ms/step) |
|---|---|---|
| fp64 | **5.01** | **33x** |
| fp32 | **1.64** | **100x** |

(The CPU figure was quoted as 211 ms/step in earlier versions of this section.
Re-measured twice: 81.4 s and 80.7 s for 500 steps, i.e. 163 ms/step, with the
start-up and the writer separated out in `docs/perf-work-hrb-jian.md` sec. 10 —
compute alone is 131 ms/step, so the like-for-like ratio to the GPU is 26x for
fp64 and 80x for fp32.)

The same trajectory, per change, fp64 500 steps:

| build | ms/step |
|---|---|
| as reviewed (patches 01/02 + E7 + V2) | 25.19 |
| + A3 static solid pair list | 15.77 |
| + A5 factored `pass1_fluid` loop | 13.26 |
| + A4 float neighbour pre-filter (reviewed state) | 12.42 |
| + P1 two-phase `pass1_fluid` (fp64 only) and P2 slot-major solid list | **5.01** |

fp32 went 2.43 -> 1.64 with P2 alone; P1 is deliberately not used there (sec.
13.2).  Measured in the shipped binary against its own pre-optimisation paths:
fp32 `pass1_fluid` 0.818 (literal) -> 0.768 (A5), `pass1_solid` 0.698 -> 0.166
(P2), `pass2_solid` 0.751 -> 0.296 (P2), and A4 is compiled out (applying it
there was measured as a 12 % *loss*).

fp32 accuracy, measured over 500 steps against the fp64 build at result9:

| quantity | before the fp32 fix | after |
|---|---|---|
| `x` maxAbs | 1.3e-5 m | **1e-6 m** |
| solid `p` median / max | 72.9 Pa / 1291 Pa | **0.01 Pa / 0.17 Pa** |
| fluid `p` maxAbs | 8.5 Pa | **0.2 Pa** |
| `vx` maxAbs | 2.4e-3 m/s | **1e-5 m/s** |
| `rho` maxAbs | 0.09 kg/m^3 | **0.01 kg/m^3** |
| bulk: kinetic energy / mean pressure | 0.000 % / 0.18 % | 0.000 % / ~0.2 % |

An earlier version of this document blamed the fp32 solid error on the
conditioning of `lambda*log10(J)*F^-T`.  That is only the smaller part: the
dominant cause was that **the state updates themselves fell below float
resolution** (see sec. 7.4).  With that fixed, the remaining fp32 error is at the
float quantisation floor: `p = c0^2(rho-1000)` with `rho` near 1000 has an ulp of
6.1e-5, i.e. 0.14 Pa per pressure step.

---

## 7. Bugs found during the port (and fixed)

### 7.1 `eta` used the neighbour instead of the higher-indexed particle

The first version differed from the reference by up to 80 Pa on `p` after 500
steps.  The cause was not chaos (an earlier and wrong conclusion): in `1.cpp`

```c
if(fabs(SurFD[i])<2.5){eta = 0.001;}else{eta = 0.00002;}
if(fabs(SurFD[j])<2.5){eta = 0.001;}else{eta = 0.00002;}   // overwrites the first
```

so **both** sides of a pair use `SurFD[j]`, where `j` is the higher-indexed
particle.  The gather used the neighbour's value, which is only correct when the
current particle happens to be the `i` side; on the `j` side the artificial
viscosity was off by a factor of 50.

It is invisible at step 0 because `SurFD` is 0 there (it is a copy of `SurF`,
which has not been computed yet), while after the first step **96 802 of
101 250** fluid particles have `|SurF| >= 2.5`.  Fix: `SurFD[(p>q) ? p : q]`.

### 7.2 `solidPack` read out of range

The static pack only covers `[NTOTAL, NALL)`, but a particle with `itype<-10` can
live outside that range (the second test case does exactly that).  The index ran
past the end of the pack and produced NaNs.  Fix: an `inpack` test that falls
back to the SoA arrays  -  `A` is 0 for those particles in `1.cpp` too.

### 7.3 Debug dump still assumed the old `Pia` layout

After the AoS change, the intermediate dump read `g.Pia + 8*NUM`, i.e. far past
the 12*SOLNUM buffer, and `cudaMemcpy` failed with "invalid argument".  Fix: a
dedicated `dump_pia()` that scatters the AoS blocks into nine `NUM`-long arrays.

### 7.4 fp32: state updates below float resolution (review E1, high)

With `dt = 1e-5` the position increment `dt*v` is 1e-9 to 1e-5 m, while one float
ulp of a coordinate between 0.25 and 1 m is 3e-8 to 6e-8 m.  So
`s.x[i] + dt*s.vx[i]` rounded straight back to `s.x[i]` for every particle slower
than ~2 mm/s: those particles never moved, however long the run.  `rho` had the
same problem, and `F` - built from differences of quantised positions - was then
noise at the 1e-5..1e-3 strain level the solid actually sees.

Measured on the input state: the float update changed nothing for 100 % of the
moving solid x coordinates, 97.9 % of solid y and 54.5 % of solid z.  After 500
steps the solid kinetic energy came out 4x too high and the solid `p` was off by
8 % median (1291 Pa worst).

Fix (external patch 02, verified here on the GPU), applied only when
`Real = float`:

1. `x` and `rho` advance with a compensated (Kahan) sum, carrying the rounding
   error of each step in `xc` / `rhoc`;
2. solid particles integrate the displacement `u = x - X0` (formed in double at
   load time), and `x = X0 + u` is rebuilt each step for the neighbour search;
3. `k_pass1_solid` accumulates `F - I` from `u_j - u_i` (identical tensor, because
   `sum_j w (X0_j - X0_i) tensor (A g0) = I` by construction of `A`);
4. `k_constitution` evaluates in double for both precisions and adds `I` there.

Measured after the fix: solid `p` median error 72.9 -> **0.01 Pa**, worst
1291 -> **0.17 Pa**, `x` 1.3e-5 -> **1e-6 m**, `vx` 2.4e-3 -> **1e-5 m/s**.
Cost: fp32 2.48 -> **2.80 ms/step** (+13 %); fp64 unchanged apart from run-to-run
variance.

### 7.5 `--precision=fp32` selected fp64 (review E2)

The parser tested `v[3] == '3'`, but in `"fp32"` index 3 is `'2'`, so the long
form silently ran fp64 while `--fp32` worked.  Both parsers are now one function
that accepts `fp32/f32/float/single/32` and `fp64/f64/double/64` in any case and
rejects anything else with a message.

### 7.6 `build.bat` wiped the build tree on every run (review E7)

It searched `CMakeCache.txt` for `CMAKE_MAKE_PROGRAM:FILEPATH=` with backslashes,
but CMake stores the path with forward slashes, so the test never matched and
every invocation deleted `build\` and recompiled everything.  Now both spellings
are searched: a second `build.bat` prints `ninja: no work to do`.

### 7.7 Diagnostic fixes (review E3-E6)

* `SPH_LOADA` now runs before the solid pack is built (it used to leave `A` and
  `solidPack` disagreeing);
* the `pairs=` line rebuilds the cell list first (it used to count on a list built
  before the previous position update, which was off by one at `T=3`);
* `SPH_NBRLIST` checks the particle index and copies exactly `hn` entries;
* the duplicated `loads x3 (scattered)` benchmark line is gone, `SPH_DUMPSTEP`
  under `SPH_PROFILE` prints a note instead of silently doing nothing, and
  `SPH_VERIFY` bins in `Real` rather than double;
* a `static_assert` documents the `SPH_NCELL <= 1024*32` assumption of
  `k_cell_scan`.


### 7.8 Deterministic order inside a cell (review V2)

`sorted[]` used to be filled by an atomic scatter, so the order in which
neighbours were summed depended on the arrival order and two identical runs
produced different bits (measured: 5e-15 m/s in `vx`, 1e-11 relative in `Pia`
after 100 steps).  A 20-line `k_cell_sort` (one thread per cell, insertion sort by
particle index, cells hold at most 42 particles) fixes it: `sph_cuda.exe --fp64`
twice in a row now produces byte-identical `result*.dat`, for both precisions.

It also made the step **17 % faster** (30.4 -> 25.3 ms), because ascending index
order is much better for the L1/L2 hit rate than the scatter order.

### 7.9 Static solid pair list: the first attempt was abandoned, the second works

The first attempt at the review's A3 produced the expected 6x on both solid
kernels but the trajectory blew up (`p` error 8e-4 at step 50, NaN by step 100,
in both the fluid and the solid columns), and it also made `--fp32` 900x slower
(2152 ms/step).  It was reverted and recorded here as unexplained.

Both symptoms had the same root cause, and it is worth writing down because the
signature was misleading: an auxiliary kernel of that attempt read `Pia` with
the **old component-major layout** (`Pia[c*NUM + p]`) after `Pia` had become
AoS.  That is an out-of-bounds read of up to 8xNUM doubles past a
12xSOLNUM buffer, i.e. arbitrary garbage into a stress tensor.  The fp64 build
survived it for 50 steps by luck; the fp32 build -- whose neighbour list holds
95 000 extra knife-edge pairs, so a different execution order -- produced NaN
almost immediately.  Once a solid particle's `rho` is NaN, `sph_cell()` clamps it
to cell 0, all 60 320 solid particles land in one cell, and every fluid particle
near that cell scans 60 320 candidates per pass: that is the 2152 ms/step, not a
CSR indexing bug.

The second attempt has no auxiliary array: everything a solid kernel needs is
either in the CSR entry or inlined from the AoS `Pia` block, and it was verified
with A3 and A4 separately disabled (`SPH_SOLDDYN`, `SPH_LITERALFLUID`) so that
every kernel could be compared against its own predecessor.  Details in sec.12.

## 8. Files

```
CMakeLists.txt              build (no absolute paths), precision switches
build.bat                   one-click build: vswhere -> vcvars64 -> CMake+Ninja
src/sph.h                   constants, SphPtrsT<Real>, device math, prototypes
src/sph_kernels.cu          all kernels + launch wrappers + explicit instantiation
src/sph_host.cu             host driver: I/O, init, step loop, output, dispatch
README.md                   quick start
docs/DESIGN.md              this document
```

---

## 9. Reproducing

```bat
build.bat Release                     :: -> build\sph_cuda.exe

:: reference case (needs input.dat in the working directory)
set SPH_MAXSTEP=500
sph_cuda.exe --fp64                   :: writes result0..9.dat
sph_cuda.exe --fp32

:: compare against the reference package
tools\CompareResults.exe <workdir> <reference-dir>
```

Diagnostics: `SPH_PROFILE=1` (per-phase timing), `SPH_BENCH=1` (per-kernel timing,
including the pre-optimisation kernels side by side), `SPH_VERIFY=1` (validate
the cell list), `SPH_DUMPSTEP=1` / `SPH_DUMPA=1` (dump intermediates for
array-by-array comparison), `SPH_LOADA=<dir>` (use an externally supplied `A`
matrix), `SPH_NBRLIST=<p>` (print one neighbour list),
`SPH_SOLDDYN=1` (walk the grid in the solid kernels instead of the static list)
and `SPH_LITERALFLUID=1` (literal `pass1_fluid` body instead of the A5 one).
The last two exist so every optimisation can be A/B'd inside one binary.

---

## 10. What is left on the table

In the review's numbering, A1 (fp32) A3 A4 A5 are done, and the remaining items
are now the interesting ones because nothing else is big:

| item | estimate | note |
|---|---|---|
| A7 fewer launches / CUDA graph | up to 1.1 ms of the 12.42 | 13 launches cost ~85 us each here; a captured graph removes the host round trips.  This is now the largest single remaining block. |
| A6 half-size cells (5^3 stencil) | fp32 mostly | 425 candidates per particle instead of 687 |
| A8 periodic reordering | - | protects long runs, needs an experiment first |
| `pass1_fluid` arithmetic | ~1 ms | what is left is one sqrt + one division per neighbour (both are software routines on this GPU) |

**A6 row merging does not pay on this GPU** (tried, measured, reverted): merging
the three x-cells of a row into one contiguous `sorted[]` range, which is
bit-identical and removes 18 of 27 loop headers, made fp64 `pass1_fluid`
8.78 -> 9.79 ms and the pure-traversal probe 1.76 -> 1.82 ms; the extra register
pressure per loop costs more than the loop headers.  See sec.6.3.

A2 (the result writer) is done as well; it does not change ms/step, but it was
worth doing because at the 50-step cadence it dominated the wall clock of the
fp32 build.  `write_result` now formats with `std::to_chars` (same text as
`%.6g`, verified byte for byte on both precisions) in up to 8 contiguous chunks
on worker threads, written in order: 0.34 s -> **0.07 s per file**.  Wall clock
for the 500-step case: fp64 6.95 s of which 6.20 s is stepping, fp32 **1.89 s**
of which 1.21 s is stepping (before: fp32 needed ~4.8 s).

---

## 11. The review's A4 and A5

### 11.0 What the three accelerations changed, numerically

The three items are independent switches, so their effect can be measured
directly: `SPH_DUMPSTEP=1 SPH_MAXSTEP=1` with everything on against
`SPH_SOLDDYN=1 SPH_LITERALFLUID=1` (legacy solid kernels, literal fluid loop,
A4 still on).  Step 1, fp64, over all 237 298 particles:

| array | ndiff | maxAbs | maxRel |
|---|---|---|---|
| `F` (solid deformation gradient) | 60 318 | 1.1e-16 ... **8.9e-16** | 6.7e-16 |
| `DFdt` | 45 425 | **2.1e-17** | 4.1e-15 |
| `Pia` (solid stress) | 60 318 | **1.1e-8** | 1.9e-12 |
| `PIJW` (solid momentum source) | 60 320 | **6.5e-10** | 1.5e-12 |
| `drhodt` | 100 509 | 8.6e-12 | 6.2e-15 |
| `dvxdt` | 101 163 | 3.1e-13 | 2.9e-14 |
| `viscos` | 94 566 | 2.7e-15 | 7.3e-16 |
| `SurF` | 60 224 | 2.7e-15 | 8.9e-16 |
| `x` | 5 | 1.1e-16 | 1.9e-16 |
| `SForce` | **0** | 0 | 0 |

`SForce` is identical because this case has no fluid/solid pair until step 566;
the 3000-step run (sec.5.4b) is what covers the coupling path.  Everything else
agrees at the 1e-15/1e-12-relative level, i.e. the same size as the
neighbour-order noise the kernel already had (sec.7.8) -- and 500 steps later the
difference is still below the 6-digit output precision (sec.12.3).

### 11.1 A5: factored `pass1_fluid` inner loop

`k_pass1_fluid_fast` keeps the physics of `k_pass1_fluid` (which is retained
verbatim, selected with `SPH_LITERALFLUID=1`) but uses four identities from the
review:

1. `(2-q)^4 - 2(1+2q)(2-q)^3 = -5 q (2-q)^3`, so the kernel value is four
   multiplies instead of a ten-term polynomial;
2. only `UL - UR` is ever needed: `-(dv.d)/dr` for fluid/fluid, `-2(vi.n)` for a
   wall, so the two separate dot products disappear;
3. `gradW.d = (dW/dr) * r2 / dr`, so `S` needs no per-component products;
4. `1/dr`, `1/(dr^2+0.01h^2)` and `1/rhoSum` come from a single division
   `R = 1/(dr*den*rhoSum)` followed by three multiplies -- two of the three
   software division calls per neighbour are gone.

Measured: fp64 11.18 -> **8.78 ms** (`pass1_fluid`), fp32 0.814 -> **0.763 ms**.
That is -2.4 ms of the 12.42 step and matches the review's "about -3 to -4.5 ms
of the 10.6 ms of arithmetic" in order of magnitude.

Verification of the tolerance claim: `CompareResults` against the pristine
package gives the *same* summary with and without A5 (`p` maxAbs 1e-3, `vx` 1e-9,
`x/rho/mass` identical), and a 500-step A5-on vs A5-off comparison at 6
significant digits shows no difference at all -- the deviation stays below the
text output precision.

### 11.2 A4: float pre-filter in front of the exact distance test

`sph_neigh` / `sph_neigh2` now evaluate `r^2` in fp32 from a float copy of the
positions (`xf`, kept current by `k_update_pos`, so it costs no extra launch) and
skip the candidate when `r2f >= r2max * 1.0001f`.  Surviving candidates take
today's exact `sph_add_rn`/`sph_mul_rn` test unchanged, so membership is still
decided in `Real`.

Why the accepted set cannot change: the float estimate differs from the exact
`r2` by at most ~`3.5*ulp_float(|x|max)/(3*hsml)` relative to `r2max`, which for
this domain (`|x| <= 1.2 m`, `ulp_float = 1.2e-7`) is 1.4e-5 -- seven times
inside the 1e-4 margin.  Verified empirically: 500-step output with A4 is
bit-identical to A4-off for both precisions, and the 3000-step post-contact
comparison is unchanged.

Measured: `pass1_fluid` 9.51 -> **8.78 ms**, `pass2_fluid` 0.33 -> 0.13,
`pass2_solid` 1.23 -> 1.15, step 12.92 -> **12.42 ms**.  The pure-traversal probe
(`count_pairs`) drops 4.17 -> **1.77 ms**, which is where the effect really shows:
exact tests fall from 69.6 M to ~10.7 M per step, as the review measured.

A detail worth keeping: **the pre-filter is compiled out for `Real = float`**
(`if constexpr` in `sph_reject_far`).  In the fp32 build the exact test is
already fp32, so the pre-filter adds three loads and six ops per candidate and
saves nothing; measured, it cost 12 % of `pass1_fluid` (0.74 -> 0.83 ms) before
the guard was added.

---

## 12. The review's A3: static solid pair list

### 12.1 Why it works

Both solid kernels evaluate the kernel gradient at the **reference** positions
`X0`, and `A` is a constant, so in this total-Lagrangian form the solid/solid
pair set and every per-pair weight are constants of the run.  (The reference
`1.cpp` is not exactly this: it accumulates `F`/`DFdt` over the *current*
pair list while evaluating the weight at the reference distance, so a pair whose
reference distance is below `3*hsml` but whose current distance is not is dropped
there and kept here.  On this lattice the largest interior reference distance is
`sqrt(8)*0.01 = 0.0283 m` against a cutoff of `0.03 m`, so the two formulations
agree while no pair stretches by more than about 6 %; the plate moves 0.1 mm
here.  A3 is arguably the more correct of the two -- see review finding R2.)
The list is built once after `GraCorrec` (`build_solid_lists()`), slot-major
(ELL) over the 60 320 solid particles: entry `j` of particle `k` lives at slot
`e = j*SOLNUM + k`, `0 <= j < sCnt[k]`.

Per directed pair `(p, q)`, with `g0 = dwdr(|X0_p-X0_q|/h) * (X0_p-X0_q)/|X0_p-X0_q|`
and `w0_q = m_q/rho0_q`:

| stored | meaning | used by |
|---|---|---|
| `sG` (3) | `w0_q * g0` | `DFdt`, and `PIJW` via `vol_q*g0 = (rho0_q/rho_q) sG` |
| `sB` (3) | `w0_q * (A_p g0)` | `F` |
| `sD` (3) | `w0_q * (A_q (-g0))` | `PIJW` |
| `sQ` (1 int) | partner index | - |

The nine tensor components sit in nine separate arrays (component `c` at
`c*sw + e`), so every warp load is contiguous 32 slots -- that is what P2
changed (sec. 13.3).  The loop bodies become

```
pass1:  F    += (x_q - x_p) (x) sB        DFdt += (v_q - v_p) (x) sG
pass2:  u    += (rho0_q/rho_q) sG         t    += (rho0_q/rho_q) Pia_q sD
        PIJW_p = (Pia_p (A_p u) - t) / rho_p
```

18 and 13 multiply-adds per pair, no square root, no division, no distance test,
no cell walk.  The `q`-independent factors are taken out of the pass-2 sum, which
is why 9 stored values are enough for both passes.

Size: **4 844 544 pairs** (80.3 per solid particle), which is exactly the number
of reference pairs with `|X0_p - X0_q| < 3*hsml` -- the same number the review's
brute-force enumeration found.  ELL width 92 (padding 14.6 %), 402 MB in fp64 and
212 MB in fp32.  Both precisions now store the same set: the review's C2 filter
(`Dr < 3*hsml*(1 - eps)`) removes the knife-edge pairs whose weight is ~1e-40
(fp64) or ~1e-20 (fp32) and which cannot change a bit of any accumulator.  `eps`
has to be representable in `Real`, so it is 1e-9 in fp64 and 1e-6 in fp32 -- at
1e-9 the fp32 expression is equal to `3*hsml` and the filter would do nothing.

### 12.2 Results

| kernel | before | static list | factor |
|---|---|---|---|
| `pass1_solid` | 4.96 ms | **0.34 ms** | 15x |
| `pass2_solid` | 5.91 ms | **0.62 ms** | 9.5x |

(`before` measured in the same binary with `SPH_SOLDDYN=1`, which is why it is
4.96/5.91 rather than the 6.3/7.4 of sec.6.1: A4 applies to the dynamic path
too.)  In fp32 the same change takes `pass1_solid` 0.698 -> 0.166 ms and
`pass2_solid` 0.751 -> 0.296 ms.

### 12.3 Correctness

* 500 steps against the CPU reference: `x/rho/mass/itype/X0/rho0` identical at
  the printed precision, `p` maxAbs 6.8e-8 Pa, `vx` 1e-9 m/s (fp64).
* The pair set is invariant under a rigid translation of the solid, and the
  stresses are too -- that is the C1 regression test,
  `cuda/tools/check_rigid_translation.py` (sec. 12.5).
* C3: the fill pass returns its per-particle count and the host aborts unless it
  matches the count pass exactly.  A mismatch would silently shift every later
  particle's slot range, so this is checked, not assumed.
* Reproducibility is kept: two identical runs are byte-identical in both
  precisions (the list is built in a fixed order -- one thread per solid
  particle, cells visited in the same order as `sph_neigh` -- so the summation
  order is deterministic).
* `compute-sanitizer` memcheck (fp64 and fp32) and racecheck: 0 errors.

### 12.4 Where the memory could be halved

`sB` is only needed by pass 1, and `F_rc = sum_q w Dx_r (A_p g0)_c` can be
rewritten as `F = M A_p^T` with a 9-element per-particle accumulator
`M = sum_q Dx (x) sG`.  That removes `sB` (-134 MB in fp64) and 9 of the 18
multiply-adds per pair, at the cost of one 3x3x3 product per particle -- about
0.1 ms.  Not done here because it changes the fp32 displacement formulation that
patch 02 depends on and would have to be re-validated.

### 12.5 The review's C1-C3: what was actually wrong, and the regression test

**C1 (real defect).** The list was searched in a cell list binned from the
*current* positions, although it is derived from `X0`.  A partner `q` was found
only if `cell(x_q)` lay within +-1 cell of `cell(X0_p)`, so once the solid had
moved about one cell the search silently lost pairs -- and the list is built
once, so a restart from a deformed state carried the error for the whole run.

Measured here by shifting the solid rigidly (a translation changes no reference
distance, so it must change nothing), 60 320 solids, one step, fp64:

| offset | old list | trace(`Pia`) max error | now |
|---|---|---|---|
| 0 | 5 371 593 | - | 4 844 544 |
| 2 mm | 5 372 031 | 4.8e-8 Pa | 4 844 544 |
| 15 mm | 5 367 871 | 5.6e-8 Pa | 4 844 544 |
| **20 mm** | 5 163 935 | **5.34e+6 Pa** | 4 844 544 |
| 25 mm | 5 068 175 | 5.34e+6 Pa | 4 844 544 |
| 25 mm, `SPH_SOLDDYN=1` (control) | - | 5.9e-8 Pa | 5.9e-8 Pa |

It is a cliff, not a slow degradation: nothing at 15 mm, 5.3 MPa at 20 mm --
four orders of magnitude above the p99 stress (4.6 kPa).  The fix is two lines:
build the cell list from a copy of the solver state whose `x` points at `X0`,
build the list, then rebuild the live cell list from `x`.

`cuda/tools/check_rigid_translation.py` runs exactly this test (one step at
0 offset and one at +25 mm, compares the list size and `trace(Pia)`) and exits
non-zero on a regression.  Against the pre-fix binary it reports 5 371 593 vs
5 068 175 pairs and 5.34e+6 Pa.

**C2 (bandwidth).** The knife-edge pairs are now filtered geometrically, see
12.1: width 115 -> 92 and fp64 list 503 -> 402 MB, with identical output.

**C3 (silent corruption).** `k_solid_pair_fill` now writes its count back and the
host aborts with the first offending particle if it differs from
`k_solid_pair_count`'s.  Cost: one 240 KB copy at start-up.


---

## 13. Post-review work: P1, P2 and the small findings

Everything in this section was measured on the RTX 5080 (sm_120, CUDA 13.3,
Windows) with the 500-step case; the review's own numbers are from a 4090 and a
different tree, so only the deltas are compared.

### 13.1 fp64 `pass1_fluid` in two phases (P1)

The fused kernel runs a ~60-instruction FP64 body whenever *any* lane of the warp
has a hit, while only about 15 % of the candidates are hits.  P1 records each
fluid particle's neighbours first, in exactly the order the fused traversal
visits them, then runs the identical body over that list:

| fp64 | ms |
|---|---|
| fused (`SPH_FLUIDLIST=0`) | 8.77 |
| list build + body over list | 0.87 + 2.67 = **3.54** |
| full step, with P2 | 11.44 -> **5.01** |

The list is slot-major -- neighbour `j` of particle `p` at `fnbr[j*NTOTAL + p]`,
so a warp walks 32 consecutive ids -- and 192 slots per particle (77.8 MB).
Measured occupancy: mean 103.9, min 42, max 115 neighbours, so no particle
overflows; if one did, `fcnt[p]` is set to -1 and that particle alone falls back
to the fused traversal, which is why the width is a memory knob and never a
correctness limit.

fp32 does **not** use it: there the kernel is traversal bound (its fused body is
0.77 ms, the traversal alone 0.56 ms) and the same split was measured as a loss.

Both instantiations share one templated kernel (`k_pass1_fluid_body<Real,
USE_LIST>`) so the arithmetic is literally the same source; `SPH_FLUIDLIST=0`
selects the fused one at run time for A/B comparison.

### 13.2 Slot-major static solid list (P2)

CSR put entry `a` of particle `k` at `sStart[k]+...`, so a warp read 32 segments
that were ~89 entries apart.  Storing entry `j` of particle `k` at `j*SOLNUM + k`
-- and splitting the nine tensor components into nine arrays -- makes every warp
load contiguous.  Per-particle order is unchanged, so nothing else moved:

| kernel | fp64 CSR -> ELL | fp32 CSR -> ELL |
|---|---|---|
| `pass1_solid` | 1.05 -> **0.34** | 0.698 -> **0.166** |
| `pass2_solid` | 1.14 -> **0.62** | 0.751 -> **0.296** |

With C2 applied as well the width is 92 (padding 14.6 %) and fp64 memory 402 MB.

### 13.3 The small findings

| id | what was done |
|---|---|
| C4 | `load_input()` now validates the type ranges (`fluid > 0`, `solid < -10`, `wall in [-10,0]`), rejects positions outside the fixed grid, and warns about trailing records.  A permuted deck would break the `eta = f(SurFD[max(p,q)])` rule, so it is refused rather than run. |
| C5 | `step()` ends with `CUDA_CHECK(cudaGetLastError())`; a launch-configuration failure no longer looks like a kernel that did nothing. |
| C6 | the `--fmad=true` comment said the opposite of what the flag does; corrected. |
| C7 | **not done.** `k_cell_sort` is still an insertion sort per cell.  It is O(n^2) only when the clamp collects many particles into one boundary cell (sec. 7.9); for this deck the worst cell holds a few hundred.  A stable radix sort on `(cell, index)` would give the same order in O(n), but it is a change to the deterministic-order guarantee, so it is left for a run that actually hits the cliff. |
| C8 | `k_gra_invert` counts its `A = I` fallbacks; the host warns, and refuses fp32 (where the `F - I` displacement form needs `a*A = I`) if the count is not zero. |
| R1 | GCC/Clang builds now pass `-O3 -ffp-contract=off` (MSVC already used `/fp:precise`), so the reference pair count no longer depends on the compiler's FMA policy. |
| R2 | sec. 2.5 and this section state the condition under which A3 matches the reference's current-configuration pair list, instead of claiming the reference's pair set is constant. |
| R3 | README: the printed `p` is the value used during the step, i.e. it lags `rho`, `x` and `vx` by one update, so `p != c0^2 (rho - 1000)` in `result*.dat`. |

### 13.4 What is still on the table

* **P3, overlap the result writer.**  After P1/P2 a result file costs about
  55 ms, which at the 50-step cadence is ~22 % of the wall clock in fp64 and
  ~40 % in fp32.  Fetching the four static columns once and copying the rest
  asynchronously into pinned memory on a side stream is the next win.
* **P4, launch overhead.**  One `cudaDeviceSynchronize()` per step remains (it
  is what `[time]` measures), and the eight kernels are not captured in a graph.
  On Windows/WDDM DESIGN measured ~1.1 ms/step of launch overhead, which is now
  a visible fraction of 5 ms.
* **P6, one evaluation per fluid pair.**  Halves the FP64 body of P1 but breaks
  bit reproducibility through the scatter; a deterministic alternative (per-pair
  buffer + reverse CSR) costs ~0.7 ms of traffic.
* **C7**, above.

---

## 14. The second review: the domain decomposition (C9-C15, P8, P9)

[`review/Claude/REVIEW-2026-11-19-multi-gpu.md`](../../review/Claude/REVIEW-2026-11-19-multi-gpu.md)
reviewed the decomposition (D1), the lean solid list (A6) and the binary I/O.  It
found that the decomposition is correct only while particles stay near where they
were when the ghosts were chosen, and that repartition -- the remedy -- was itself
wrong.  Both are fixed here, together with the smaller findings, and the exchange
records are slimmer.  All numbers below were re-measured on 8 x RTX 5090 D
(`sm_120`), CUDA 13.5, fp64, one rank per card, `input.dat` (237 298 particles)
unless the row says otherwise.

### 14.1 C9: the ghost set was chosen once and never revalidated

`build_ranks()` selects the ghosts of a rank as the particles inside its block
grown by 3 h, which is complete only at that instant.  There was no skin and no
check, and `SPH_D1_REPART` defaults to 0, so the default decomposed run silently
lost interactions as soon as the fluid started to move: the review measured the
first missing fluid pair at step ~1750 (K = 8) and 4.4e-3 m / 1.9e4 Pa of
deviation from one GPU at step 6000.

**Fix.**  Three parts, all in `step_d1()` and `build_ranks()`:

1. the ghost shell is `3h + s` with `s = SPH_D1_SKIN` (default 1.5 h);
2. every step each rank reduces `max |x - x_build|` over its owned particles (a
   Chebyshev distance, matching the per-axis box test) into a parity-buffered
   host slot, and reads all ranks' values back after E1's barrier -- the values
   are published *before* the barrier and read after it, so the K ranks agree on
   the maximum without spending a barrier of their own;
3. when the maximum passes `s/2` they rebuild the decomposition together, reuse
   the repartition path, and reset `x_build`.

The argument is the standard Verlet one: a particle that was not selected was
more than `3h + s` from the block on some axis, and if both ends of a pair move at
most `s/2` on every axis, they are still more than 3 h apart.

`SPH_D1_SKIN=0` restores the old behaviour for A/B runs and prints a warning.

### 14.2 C10: repartition selected ghosts and grids from t = 0

`repartition()` cut the new boxes from the current positions but called
`build_ranks(J, J.deckRows, ...)`, i.e. the deck as read at start-up.  So the new
ghost set and every rank's neighbour grid (`finish_deck()` derives it from the row
extremes) came from t = 0 coordinates: a rank's particles could lie outside its
own grid (the C7 cliff), and turning repartition on made a correct run wrong
(5.0e-3 m / 643 Pa at step 6000 in the review).

**Fix.**  Rank 0 builds a copy of the deck with `x`, `vx` and `rho` overwritten
from the live state and passes that to `build_ranks()`.

### 14.3 The smaller findings

| id | what was done |
|---|---|
| C11 | `xch_free()` frees the 20 index/offset arrays `build_send_lists()` allocates, and zeroes the counts; device memory is now flat across rebuilds (1781 / 2171 MB per rank over 19 events). |
| C12 | the A6 lean/fat choice is taken by all ranks together: `sph_a3_fat_hook` publishes each rank's fat size, meets at the rank barrier, and hands back the maximum.  A rank that owns no solid still joins the barrier (with a fat size of 0) -- otherwise it walks past the barrier the others are waiting at and the run desyncs.  With `SPH_A3_LIMIT_MB=100`, three runs in a row now all choose lean; before, six identical runs split 3/3. |
| C13 | the decomposed driver refuses a deck whose walls have a non-zero velocity: E1 carries the fluid and solid groups only, so a moving wall would never reach its ghosts.  Every wall in this deck is static, and a static wall cannot start moving (no kernel writes `SForce` or `PIJW` for it). |
| C14 | solid ghosts are selected by current *or* reference position.  The A3 pairs are evaluated in the reference configuration, so a partner can be inside the reference shell of a rank's solids while being outside the shell of its current block.  The reference box of a rank's own solids is static, so this adds only a few particles; `make_part()` counts them the same way. |
| C15 | `sph_matvec()` strides by `numP` (the tensors are ghost-strided); `init_reference_state()` clears `numP`-strided arrays; the profiler's gather column is labelled as a launch time; `gs0` is now read (it bounds the ghost-fluid block of the p rebuild). |

### 14.4 P8 and P9: the exchange carries only what is read

E3 shipped the fluid `PIJW` after `pass2_fluid`, and no kernel ever writes `PIJW`
for a fluid particle -- it is three zeros per ghost fluid per step.  E1 shipped
`p`, `PIJW` and (for solids) `Pia`, none of which the receiving side needs:
`p` and `irho` are functions of `rho`, and `Pia` is overwritten by E2 before
`pass2_solid` reads it.

The records are now 8 doubles for a fluid (x, vx, rho, SurFD) and 7 for a solid
(x, vx, rho), against 12 and 24; E3 is gone; `launch_xch_fixup()` rebuilds `p`
and `irho` for the ghost block with the expressions `k_prepare()` uses, so the
bits are unchanged.

### 14.5 Measurements

Validation (all against the same binary's single-GPU run; `max|dx|` is the
worst position difference, `max|dp|` the worst pressure difference):

| run (3001 steps) | max abs dx | max abs dp |
|---|---:|---:|
| K = 8, 501 steps, skin trigger | 3.3e-16 m | 2.4e-7 Pa |
| K = 8, **skin trigger**, 3001 steps | **1.2e-15 m** | 3.1e-7 Pa |
| K = 8, `SPH_D1_SKIN=0` (old behaviour), 3001 steps | **1.9e-6 m** | 1.6 Pa |
| K = 8, `SPH_D1_REPART=1000` (C10 fixed), 3001 steps | 1.7e-15 m | 2.3e-7 Pa |

Everything after V2 (sec. 14.5b) is checked with `fc /b` instead -- the
differences above are only quoted for the runs that predate it.

The skin costs what it costs: 68 411 -> 102 602 ghost slots (+50 %) on this deck
before P11's second layer, and six rebuilds in 3001 steps at 175 ms each (260 ms
once the layer-1 ghosts run their own constitution), i.e. ~0.3-0.5 ms/step.  The
readback of the displacement is scheduled adaptively (10 % of the skin may go
unseen), so it costs ~0.01 ms/step rather than the 0.6 ms a per-step synchronous
8-byte read costs on Windows/WDDM.

Exchange and barrier structure, per step:

| | before | after |
|---|---|---|
| exchange points | 3 (E1, E2, E3) | **1** (E1) |
| records | fluid 12, solid 24, E2 13, E3 3 doubles | fluid 8, solid 7 |
| E1 bytes at K = 8 (this deck) | 8.2 MB | 3.4 MB |
| rank barriers | 6 | **1** |
| `Pia` + `solidPack` slots | one per index >= ntotal | one per solid (P13) |

Step-time comparisons were taken on this 8 x 5090 D box at different times and
the machine is shared, so only same-session pairs are meaningful:

| K = 8, 501 steps | ms/step |
|---|---:|
| old exchange (E1 12/24, E3 on) | 2.396 |
| new exchange (E1 8/7, no E3), no skin | **2.134** (-11 %) |
| new exchange with the 1.5 h skin | 2.388 |
| final (skin + P10 + P11 + P14), 3001 steps | ~2.3-3.1 depending on machine state |

### 14.5b V2: K > 1 is byte-identical to K = 1

The decomposition used to differ from the single-GPU run at the 1e-16 level,
because each rank derived its own neighbour grid and ordered a cell by *local*
index -- and a local index says nothing about the deck order once ghosts are
appended.  Two changes make the summation order the one the single-GPU run uses:

* every rank bins into the **global grid** derived once from the whole deck (and
  the grid no longer drifts when a repartition re-derives it);
* `k_cell_sort` orders a cell by **global id** (`s.gid`); on one rank `gid` is
  null and the local index *is* the global one.

The payoff is not cosmetic: a halo or exchange bug can no longer hide under
rounding noise.  Everything below was verified this way -- `fc /b` on the result
files, not a tolerance:

| configuration | result |
|---|---|
| K = 4 and K = 8, 501 and 3001 steps | byte-identical to K = 1 |
| K = 8, 3001 steps, six automatic rebuilds | byte-identical |
| K = 8, 3001 steps, `SPH_D1_REPART=1000` | byte-identical |
| `SPH_RANKS=auto`, 501 steps | byte-identical |
| **negative control**: K = 8, `SPH_D1_SKIN=0`, 3001 steps | 2.0e-6 m / 1.7 Pa |

The negative control matters: with the skin disabled the decomposed run still
decays, which is exactly what V2 is for -- the check is supposed to fail when the
halo is wrong.

One caveat: the V2/P11/P13 refactor shifted the last bit of *some* solid stresses
relative to the previous build (<= 1 ulp on 18 % of the solids, deterministic,
fluids unaffected).  The dumped `F`, `DFdt`, `A` and `rho` arrays are identical,
so this is instruction scheduling of the constitution kernel, not a change in the
algorithm; the 501-step drift is 2.2e-16 m, the same rounding floor as before.
Byte-identity with the *previous build* is therefore not claimed -- byte-identity
of K > 1 with K = 1 is, and it is what the validation above uses.

### 14.5c P10, P11, P13, P14

| id | what was done | effect |
|---|---|---|
| P10 | the send block is double buffered and consecutive exchange points alternate halves, so the trailing barrier is gone (a rank can only rewrite a half after the *next* exchange's barrier, which no rank reaches before finishing its relay) | two barriers per exchange point -> one; six per step -> one |
| P11 | the ghost solids come in two layers: layer 1 is what the box selects and **runs its own constitution** (pass1_solid + constitution over slots `[0, solnum + ngs1)`), layer 2 is the layer-1 solids' A3 partners, which only ever get read | **E2 is gone**: one exchange point per step.  Costs: +1790-3 300 ghost solids per rank on this deck and ~25 % more solid work in the halo |
| P13 | `Pia`/`solidPack` hold one slot per *solid* instead of one per index >= ntotal | 92 MiB on the 1203k deck, 1.47 GiB on the 19.3 M one, ~13 MB per rank here |
| P14 | two result staging buffers and a writer thread on rank 0; a stage is reused two results later, and rank 0 waits for its writer before the barrier that releases it | the 55 ms file write no longer stalls all ranks |

P11 is the one that changes the structure of the step: it is the reason the
constitution has to be bit-exact on a ghost, and it is why the partner set of a
layer-1 ghost has to be complete -- the A3 partner scan uses the same global grid
and the same global-id order for owner and ghost, so the two sums are identical
by construction.  The build checks that with the byte-identity table above.

### 14.5d L1/L2/L3: choosing K, and what a ghost costs

`SPH_RANKS=auto` ranks the devices (a 64 MB device-to-device bandwidth probe, free
memory, and the display watchdog flag, which is how a card that drives the desktop
gets demoted) and then chooses K from the *actual* partition: for every K it runs
the same search `build_job` runs and predicts

    T(K) = max_r(own weight) + w_ghost * ghosts/K + XCH_UNITS

i.e. the slowest rank's compute plus its share of the ghosts plus one exchange
point's floor.  On this machine (8 x 5090 D, all of them carrying a display
flag) it predicts the table in `[D1] K prediction` and picks K = 8, and the
result is byte-identical to K = 1.  The smallest K within `SPH_D1_AUTO_SLACK`
(default 10 %) wins, so a batch of cases keeps more cards free.

L2/L3 changed the objective: a solid particle now weighs 0.45 (the measured
0.41 with the fat list / 0.97 with the lean one, so the default sits between and
is a knob), and a ghost is charged 0.06 rather than a full particle -- it is a
cell-list slot plus its share of the exchange, not a particle that computes.  The
review's point was that charging a ghost at the owned rate turned the ghost term
into an exchange penalty with an arbitrary scale.

### 14.5e M8: the reflection at z = 1

`k_update_fluid()` flipped `v_z` whenever `z > 1.0`, which also flips a particle
that is already moving down and can leave it oscillating just above the plane.
The standard form reflects only outward motion and is now the default; the
literal 1.cpp behaviour is `SPH_REFLECT=flip`.  The fluid in `input.dat` stays
below 0.45 m, so the two settings produce identical results on it -- verified,
not assumed -- and the choice is a knob rather than a silent change.

### 14.6 What is still open

* **P12, overlap the exchange with the interior work.**  Not done.  It used to
  overlap E2 (0.2-0.6 ms); with P11 there is one exchange point left, and every
  kernel of the step reads ghost `x`, so the remaining overlap needs the
  interior/boundary split the review describes (interior particles have no ghost
  neighbour and can run on a second stream while the exchange is in flight).
  That is a redesign of the step scheduling, not a local change, and it should be
  measured against the byte-identity harness rather than bolted on.
* **L2 calibration on the target card.**  The weights are the review's RTX 4090
  numbers; the 1/64-rate fp64 of the 5090 D may move them, and `SPH_BENCH` is
  the way to re-measure.
* **C7 / the cell-list cliff** still exists in the single-GPU path (a fixed grid
  and an insertion sort); the decomposed path now inherits the same grid by
  design, because V2 requires it.


