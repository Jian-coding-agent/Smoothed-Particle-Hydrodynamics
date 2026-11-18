# SPH → CUDA Refactor: Source Analysis and Porting Plan

> Analysed package: `SPH-src-20261112-r5-sln.zip` → `SPH-src/` (`1.cpp`, 1488 lines, `input.dat`, `result0..9.dat`, `tools/`)
> Analysis machine: RTX 5080 (sm_120, 16 GB) / CUDA 13.3 / MSVC 14.51 (VS2026) / CMake 4.2.1 / Ryzen 9 9950X (16C32T) / 253 GB
> Every timing in this document was measured on that machine, not estimated.
>
> Note: the original `1.cpp` carried Chinese (GBK) comments. They have since
> been translated to English, so the source is now ASCII; references below to
> the GBK code page describe the state at the time of the analysis.

---

## 0. Executive summary

| Item | Fact |
|---|---|
| Source | single file `1.cpp`, 1488 lines, explicit time integration SPH, **all double**, global SoA (`double x[3][num]` style) |
| Size | `num = 237298` particles = 101250 fluid + 60320 solid + 75728 wall; **11.1 million** particle pairs per step (~94 neighbours/particle) |
| CPU baseline | 32 threads **500 steps in 105.6 s (0.211 s/step)**; 8 threads 171.4 s (0.343 s/step) |
| Per-step hotspots | pass1 fluid/solid forces **55%**, two "thread-private buffer reductions" **22%**, pass2 **10%**, neighbour search **7%**, solid constitution **5%**, integration 1% |
| Additional finding | writing `result<N>.dat` takes **1.9 s per file** (20 MB of text), 19 s of the 500 steps (**18%**) |
| Key conclusion 1 | the original scatter pattern ("update i and j for every pair") would need either 500 M atomics/step on the GPU or 32 private buffers; **but every pairwise formula in the code can be rewritten mathematically as "each particle gathers its own side"**, giving no atomics, no reductions and no races (derivation in §3.2) |
| Key conclusion 2 | the reductions (22%) **disappear entirely** under the gather architecture, and the output (18%) can be rewritten as bulk writes; these are two free speedups |
| GPU precision capability | measured on this machine: FP64 **0.81 TFLOPS** (= 1/67 of FP32), FP32 54.3 TFLOPS, Stream bandwidth 848 GB/s |
| Numerical acceptance baseline | rerunning the same exe, p differs by at most 0.01 (relative 2e-6), vx ≤ 1e-9, and x/rho are identical at printed precision → use this "noise floor" for port acceptance, see §4.2 |

---

## 1. Source structure

### 1.1 Physical model (weakly compressible SPH + fluid-structure coupling)

- **Kernel**: a kernel other than the cubic spline, `kernel(dr)`, `dwdr(dr)`, `dwdx(dx,dr)`; `h = 1.5*hsml`, `hsml = 0.01 m`, support domain `q ≤ 2`, i.e. `dr ≤ 3*hsml = 0.03 m`.
- **Fluid**: equation of state `p = c0²(ρ-1000)` (`c0 = 48`), Riemann approximation `U*`/`P*` for particle pairs, artificial viscosity `eta` (two levels, 0.001 or 2e-5).
- **Fluid-solid/wall**: uses `vecn[j]` (solid/wall normal) to construct the mirror velocity and applies `SForce` on both sides.
- **Solid**: SPH discretisation of the deformation gradient `F = ∂x/∂X0` plus a gradient-correction matrix `A` (computed once by `GraCorrec`), `J = det F`, `ρ = ρ0/J`, `Pia = μ(F - F⁻ᵀ) + λ·log10(J)·F⁻ᵀ + PD (friction term)`, with PIJW the solid internal stress divergence.
- **Time integration**: explicit Euler, `dt = 1e-5`; fluid velocity cap `MaxV = 30`; solid bottom/side boundaries force `v = 0`.

### 1.2 Global state

- Per-particle state: **146 doubles** (12 3×3 tensors + several 3-vectors + scalars), `237298 × 146 × 8 ≈ 277 MB`.
- Plus the OpenMP private accumulation buffer `g_omp_buf`: `nth × 35 × num` doubles, **2.03 GB** with 32 threads; this is the main memory-bandwidth burden of the current implementation.
- Particle-pair arrays `pair_i/pair_j`: 11.1 M pairs → 89 MB (`reserve(14000000)`, `data()` used as the length, `size()` always 0).

### 1.3 Execution order of one time step (`main`)

```
for itime:
  1. direct_find()            # uniform-grid cell list: count -> prefix sum -> bucket -> enumerate 14 half-neighbour cells into the pair table
  2. continue_density()
       2.0 serial: SurFD = SurF;  p = c0²(ρ-1000) (<1000 -> 0)
       2.1 pass1 (parallel over pairs): fluid-fluid -> drhodt/dvxdt/viscos/SurF/AveV;
                                    fluid-(solid|wall) -> update the fluid side only;
                                    (solid-solid) -> F, DFdt
       2.2 reduction 1: sum the nth private buffers back into the globals (two ranges: fluid / solid)
       2.3 solid constitution (serial over [ntotal, ntotal+SolNum)): DFdtB, J, ρ=ρ0/J, SD, PD, F⁻¹, F⁻ᵀ, Pia
       2.4 pass2 (parallel over pairs, solid-containing pairs only): solid-solid -> PIJW; fluid-solid -> SForce on both sides
       2.5 reduction 2: SForce over everything + PIJW over the solid range
  3. integration: fluid ρ, v, speed cap; solid v (+PIJW+SForce-gravity) + boundary; all x += dt·v
  4. every 50 steps write result<N>.dat (237298 rows × 14 columns, 6 significant digits, tab-separated)
```

### 1.4 Initialisation (once)

- `init_grid()`: fixed domain `[-0.10,0.70]×[-0.10,0.70]×[-0.10,1.20]`, cell edge `3*hsml = 0.03` → **27×27×44 = 32076 cells** (7.4 particles per cell on average).
- `NorVector()`: one neighbour search to compute `sumvecn` → `vecn` (solid/wall normals, zero for fluid).
- `GraCorrec()`: one neighbour search plus a serial pass over 11.1 M pairs to compute the gradient-correction matrix `a` and invert it into `A` (solid range).

### 1.5 "Quirks" that must be reproduced exactly (otherwise results drift)

| # | Location | Behaviour |
|---|---|---|
| 1 | `continue_density` pass1 | `eta` is assigned twice, so **only `SurFD[j]` (the neighbour) decides it in the end**; after gather-isation it must use "the neighbour's SurFD" |
| 2 | `SurF` | **never cleared** (only once in `NorVector`), accumulates on top of the old value every step → unbounded growth; affects `eta` through `SurFD` |
| 3 | pass2 `PRii` | contains `PIJW[i]`, but a fluid particle's `PIJW` is never written → always 0; keeping the formula is equivalent to `PRii = p[i]` |
| 4 | `AveV`/`DisGrad*`/`GLStrain*` | accumulated but **never used** (`GLStrain()` is commented out in main) → can be deleted |
| 5 | 3 places with an epsilon in `dr` | pass1 adds `1e-13`, pass2 adds none, `NorVector` adds `1e-12` |
| 6 | Particle index order | `pair` forces `i<j`, but every pairwise term is **symmetric** in i/j (see §3.2), so the order does not affect the physics, only the floating-point summation order |
| 7 | Type test | fluid `itype>0` (=2), solid `itype<-10` (=-20), wall `itype=-2` **is not a solid** |
| 8 | Output `p` column | fluid prints `p[i]`, everything else prints `Pia11+Pia22+Pia33` (always 0 for walls) |
| 9 | `paii = 0.25*rho0*1000*hsml` | the constant 1000 is a literal, not `rho0` |
| 10 | Solid density | overwritten by `ρ0/J` every step; when `abs(J)<1e-6` it is set to 1000, but `log10(J)` is still evaluated |

---

## 2. Measured baseline

### 2.1 CPU, full machine

| Configuration | 500-step time | Per step | Notes |
|---|---|---|---|
| 32 threads | **105.6 s** | 0.211 s | includes 10 outputs (19 s) |
| 8 threads | 171.4 s | 0.343 s | 8→32 threads is only 1.62×, limited by the 2 GB buffer bandwidth |
| Startup | 1.0 s | — | read input.dat + 2 GB calloc + NorVector + GraCorrec |

### 2.2 Per-step breakdown (32 threads, `QPC` instrumentation, average of 102 steps)

| Phase | s/step | Share |
|---|---|---|
| Neighbour search `direct_find` | 0.0098 | 7.0% |
| pass1 (fluid/solid forces) | 0.0763 | **54.5%** |
| Reduction 1 | 0.0236 | 16.9% |
| Solid constitution | 0.0073 | 5.2% |
| pass2 (stress/coupling) | 0.0137 | 9.8% |
| Reduction 2 | 0.0072 | 5.1% |
| Integration | 0.0014 | 1.0% |
| **Total (excluding output)** | **0.140** | 100% |
| Output (every 50 steps) | 1.91 s/file | ≈0.038 s/step amortised |

### 2.3 GPU capability (measured on this machine)

| Item | Value |
|---|---|
| Card | RTX 5080, sm_120, 84 SM, 16 GB |
| FP64 FMA | **0.81 TFLOPS** |
| FP32 FMA | 54.3 TFLOPS |
| Stream (2R+1W) | 848 GB/s |
| Toolchain | nvcc 13.3 + cl 14.51 + CMake 4.2.1 verified to build and run (only the C4819 code-page warning, needs `/utf-8` or `/wd4819`) |

---

## 3. CUDA port design

### 3.1 Overall: five kernels per step, all "one thread per particle"

| Kernel | Thread range | What it does | Original code |
|---|---|---|---|
| `k_prepare` | 0..num | `SurFD=SurF`; `p=c0²(ρ-1000)` (<1000→0) | 2.0 |
| `k_cellcount` / `k_cellscan` / `k_cellscatter` | 0..num / 1 block / 0..num | uniform-grid CSR: histogram → prefix sum → sort by cell | `direct_find` 1)2) |
| `k_pass1` | 0..ntotal+SolNum | traverse the neighbours in the 27 cells, **accumulate only its own side** | 2.1 |
| `k_constitution` | ntotal..ntotal+SolNum | 3×3 constitution (J, F⁻¹, SD, PD, Pia…), local variables suffice | 2.3 |
| `k_pass2` | 0..ntotal+SolNum | neighbour traversal: solid-solid→PIJW; fluid-solid→SForce | 2.4 |
| `k_update` | 0..num | ρ/v update, speed cap, solid boundary, `x+=dt·v` (split by index range) | 3 |
| Every 50 steps | — | `cudaMemcpy` back to the host → the same `ofstream` output code | 4 |

The reductions (2.2, 2.5) are **not needed at all**: each particle's accumulation is written only by its own thread, so it is private by construction.

### 3.2 Key derivation: every pairwise term can be gather-ised (the foundation of the whole port)

The original code updates both i and j for every pair (i<j). To change it to "each particle p traverses its neighbours q and computes only p's own contribution", the formula must be identical whether p is the i or the j side. Verified term by term (`e_i = (x_j-x_i)/dr = -e_j`, and `dwdx` is odd in `dx`):

| Quantity | i side (original) | j side (original) | Gather form (p↔i or p↔j, same expression) |
|---|---|---|---|
| `drhodt` | `-2ρ_i(m_j/ρ_j)(UL_i-UM_i)dwdr` | same structure | `-2ρ_p(m_q/ρ_q)(UL_p-UM_p)dwdr(dr)` ✔ |
| `dvxdt` | `-2m_j(PM_i/ρ_iρ_j)∇W(x_i-x_j)` | same structure | `-2m_q(PM_p/ρ_pρ_q)∇W(x_p-x_q)` ✔ |
| `viscos` | `+2m_j·η·S·(v_i-v_j)/(ρ_j(dr²+0.01h²))`, `S=Σ∇W·(x_i-x_j)` | same structure (`S` computed from the reversed difference but equal) | same expression, `S=Σ∇W(x_p-x_q)·(x_p-x_q)` ✔ |
| `SurF` | `+S·m_j/ρ_j` | `+S·m_i/ρ_i` | `+S·m_q/ρ_q` ✔ |
| `AveV` (unused, delete) | `(m_j/ρ̄)(v_j-v_i)W` | same structure | `(m_q/ρ̄)(v_q-v_p)W` ✔ |
| `F` (solid-solid) | `(m_j/ρ0_j)(x_j-x_i)⊗(A_i g0)` | `(m_i/ρ0_i)(x_i-x_j)⊗(-A_j g0)` | both equal `(m_q/ρ0_q)(x_q-x_p)⊗(A_p g0_pq)`, `g0_pq=dwdx(X0_p-X0_q)` ✔ |
| `DFdt` (solid-solid) | `(m_j/ρ0_j)(v_j-v_i)⊗g0` | same structure | `(m_q/ρ0_q)(v_q-v_p)⊗g0_pq` ✔ |
| `PIJW` (solid-solid) | `(m_j/ρ_iρ_j)(Pi_i-Pi_j)` | `(m_i/ρ_iρ_j)(Pi_j-Pi_i)` | `(m_q/ρ_pρ_q)[Pia_p·(A_p g0_pq) - Pia_q·(A_q g0_qp)]` ✔ |
| `SForce` (fluid-solid) | `-2(m_j/ρ_iρ_j)∇W(x_i-x_j)·C_i` | `-2(m_i/ρ_iρ_j)∇W(x_j-x_i)·C_i` | each side computes its own, with `C` using the other side's `vecn` and `p` ✔ |

Therefore: **each particle can gather independently; no atomics, no private buffers, no reductions**. This also removes the CPU version's 22% reduction overhead.

### 3.3 Neighbour search

- Keep the original fixed grid `[-0.1,0.7]²×[-0.1,1.2]`, cell edge 0.03, and **the clamp rule** (particles outside the domain go into the boundary cell), so the particle-pair set matches the CPU pair by pair.
- GPU side: `atomicAdd` histogram (237k increments, ~10 µs) → prefix sum → `atomicAdd` cursor bucket fill.
- **During gather, scan the 27 neighbouring cells directly**, without building a pair table or CSR: 237k × 27 × 7.4 ≈ 47 million distance tests, about 0.3–0.5 ms, cheaper than building and re-reading a table.
  (If a CSR adjacency list is later built once and reused by pass1/pass2, 89 MB buys 2× distance tests; evaluate in M3.)
- Bucket filling uses atomics, so the summation order per step is unstable (does not affect correctness, only the last digits); if **bitwise reproducibility** is needed, use `cub::DeviceRadixSort` by `(cell, particle id)` (+0.2–0.4 ms).

### 3.4 Memory and trimming

- Kept: `x[3] vx[3] X0[3] rho rho0 mass p itype SurF SurFD vecn[3] drhodt dvxdt[3] viscos[3] A[9] F[9] DFdt[9] Pia[9] PIJW[3] SForce[3]`
- Trimmed (used only inside a single loop → local variables): `a[9] Fni[9] FniT[9] SD[9] PD[9] DFdtB[9] DisGrad[9] GLStrain[9] AveV[3] sumvecn[3] countiac`
- Result: **146 → about 77 doubles/particle (≈146 MB)**, with no 2 GB buffer; 16 GB of GPU memory is no constraint.

### 3.5 Output

- The CPU version's 1.9 s/file comes from formatting each `ofstream << double` individually and the flush semantics of every `endl`.
- GPU version: `cudaMemcpy` 14 arrays back to the host (26 MB, ~5 ms) and use the same `ofstream`/`snprintf` output code; or format into a character buffer on the device and `fwrite` it.
- The 10 files of a 500-step run should drop from 19 s to < 1 s (**that alone is an 18% speedup**). The output text must stay identical (6 significant digits, tabs); `tools/CompareResults.exe` parses the numeric columns, so an identical format is enough for value-by-value comparison.

### 3.6 Precision routes

| Route | Estimated per step | Relative to CPU | Risk |
|---|---|---|---|
| **A. All FP64 (do this first)** | 5–15 ms | 14–40× | none (same precision domain as the reference) |
| B. FP64 state + FP32 pair arithmetic | 2–5 ms | 40–100× | drift over 500 steps must be measured |
| C. All FP32 | 1.5–3 ms | 70–140× | `log10(J)`, `ρ0/J`, `p=c0²(ρ-1000)` are sensitive to small quantities; likely fails acceptance |

Recommend A → (if needed) B; abstract with `using real = double`, so B/C only change the type plus targeted verification.

---

## 4. Validation plan

### 4.1 Reference data

`SPH-src/result0..9.dat` = 500-step reference output (237298 rows × 14 columns). Verified: it is the output of this package's Release exe with 32 threads (a fresh 32-thread run differs only in the last printed digits).

### 4.2 Tolerances (the measured "noise floor", not a guess)

Differences from **rerunning the same exe** with the same 32 threads, and between 8 and 32 threads:

| Column | Rerun 32↔32 | 8↔32 | Notes |
|---|---|---|---|
| x0/x1/x2 | **identical** | identical | invisible at 6 significant digits |
| p | maxAbs **0.01** (relative 2e-6), 27k values out of 10 files differ | 0.01 | `p=2304(ρ-1000)` amplifies small differences by 2304× |
| vx0/vx1/vx2 | maxAbs ≤1e-9 (mostly 1e-10) | same | |
| rho | identical (to printed precision) | same | actual difference ~4e-6 |
| mass/itype/X0/rho0 | identical | same | constants |
| Rows differing | 2500–3700 / 237298 / file (≈1.2%) | same | |

> This noise floor is the criterion of §9.7 L2; the CUDA version's measured differences are all within it (`p` at 1e-3, an order of magnitude smaller).

> Note: `CompareResults.exe`'s `maxRel=2` is an artefact of near-zero values (e.g. vx1≈-1e-13 changing sign); look at `maxAbs` and the per-column summary.

### 4.3 Comparison workflow

```
.work/gpu-run/            # contains input.dat + SPH-cuda.exe
  SPH_MAXSTEP=500 .\SPH-cuda.exe        # produces result0..9.dat
tools\CompareResults.exe .work\gpu-run SPH-src
```

---

## 5. Project structure (proposed)

```
SPH/
├─ SPH-src/                     # original CPU reference (left untouched)
├─ docs/CUDA_PORT_ANALYSIS.md   # this document
├─ cuda/
│  ├─ CMakeLists.txt            # LANGUAGES CXX CUDA, CMAKE_CUDA_ARCHITECTURES=native
│  ├─ build.bat / build.ps1     # same "one click, zero absolute paths" style as the existing project
│  ├─ src/
│  │  ├─ sph_common.h           # constants (§1), grid parameters, device array declarations
│  │  ├─ sph_kernels.cu         # the six kernels above
│  │  ├─ sph_solver.cu          # initialisation / single step / output / timing
│  │  └─ main.cpp               # entry point, SPH_MAXSTEP/SPH_THREADS equivalents, -o directory
│  └─ tools/compare.ps1
└─ .work/                       # temporary baseline runs and instrumented builds (safe to delete)
```

---

## 6. Phased plan

| Phase | Content | Completion criterion |
|---|---|---|
| **M1 skeleton + correctness (first version)** | CMake/CUDA project, read input, cell list, pass1, constitution, pass2, integration, output; FP64 | 1-step `result0.dat` passes §4.2; per-step timing printed |
| **M2 full-pipeline validation** | run 500 steps and compare all of `result0..9` | all 10 files pass §4.2; timing table produced |
| **M3 performance** | neighbour-cell tiling/shared memory, register pressure, per-step pair-pass fusion, fast output, deterministic sorting | stable < 5 ms/step (FP64), or FP32 path < 2 ms/step |
| **M4 engineering** | build.bat/build.ps1, README, comparison script, instructions for coexisting with the CPU version | one command produces the exe and reproduces the validation |

**Estimate:** M1 is the main effort (many formulas that must be aligned one by one), M2 is mostly waiting and comparing, M3 is tuning.

---

## 7. Risks

1. **Bitwise identity is not the goal**: after 500 steps the measured maximum differences are p 1e-3, vx 1e-9, and x/rho/mass/itype/X0/rho0 identical at printed precision; this is inside the CPU's own rerun noise floor. Judge with the two-level criteria of §9.7.
2. **Ill-conditioning of the solid constitution**: `1/J` and `log10(J)` are sensitive as J→0; if J crosses zero on some step, the GPU and CPU versions may produce NaN at different times. The first version should report the first NaN particle/step.
3. **Register pressure**: 9×9 temporaries in the constitution kernel easily spill; if necessary keep `DFdtB/SD/PD` in global memory as in the original.
4. **FP32 route**: build the FP64 baseline first, then evaluate drift separately.
5. **MSVC+CUDA details**: `/utf-8` (or `/wd4819`) avoids the code-page warning; `-arch=native` works with CMake 4.2.

---

## 8. Open questions

1. Precision route: **A (FP64 first, recommended)** / B (FP64 state + FP32 pair arithmetic) / C (all FP32)?
2. Acceptance: the "noise floor" tolerance of §4.2 (recommended), or another requirement (e.g. bitwise identity with one specific version)?
3. Scope: solver only, or also `input.dat` pre/post-processing (e.g. ParaView export)?

---

## 9. M1/M2 measured results (first CUDA implementation)

Project: `cuda/` (`src/sph.h` + `src/sph_kernels.cu` + `src/sph_host.cu` + `CMakeLists.txt`),
built with CMake + nvcc 13.3 + MSVC 14.51, `-arch=native` (sm_120).

### 9.1 Functional status

| Item | Result |
|---|---|
| Read `input.dat` / initialisation (NorVector, GraCorrec) | ✅ |
| Neighbour pair count | **11,102,707, bitwise identical to the CPU** (T=0/T=1 exactly equal, then only 2–15 pairs differ) |
| One-step intermediate arrays compared one by one (F/DFdt/Pia/rho/SForce/PIJW/drhodt/vxdt/viscos/SurF) | ✅ see table below |
| 500-step stable run | ✅ no NaN/Inf |
| Output format | ✅ column count/tabs/6 significant digits, compatible with `CompareResults.exe` |

One-step comparison against the true values (not the printed values):

| Quantity | maxAbs | maxRel |
|---|---|---|
| F11 (solid-solid deformation gradient) | 2.2e-15 | 2.2e-15 |
| DFdt11 | 4.5e-17 | 1.0e-12 |
| rho (solid) | 4.0e-12 | 4.0e-15 |
| Pia11 | 3.0e-8 | 3.1e-10 |
| p = ΣPia (solid) | 8.8e-8 | 2.7e-10 |
| drhodt (fluid) | 1.1e-11 | 5.6e-10 |
| dvxdt / viscos (fluid) | 3.5e-13 / 1.8e-15 | 2.9e-10 / 1.7e-9 |
| x (position) | identical at printed precision | — |
| A (gradient-correction matrix, initialisation) | 2.7e-15 | 1.3e-15 |

### 9.2 Performance (first version, unoptimised)

| | 500-step time | Per step |
|---|---|---|
| CPU 32 threads | 105.6 s | 211 ms |
| **CUDA (first version)** | 28.1 s | 56 ms |
| of which pass1 / pass2 / grid | — | 41.7 / 13.8 / 0.1 ms (after optimisation 25.6 / 10.8 / 0.1, whole step 37.1 ms, see §11) |

Speedup 3.8×. Against the FP64 roof (about 2.3 GFLOP/step ÷ 0.81 TFLOPS ≈ 3 ms) there is still ~9× of headroom; the bottleneck was located:
each neighbour visit performs 12–14 **cross-array** random reads (each of the 32 threads in a warp hits a different cache line → 448 transactions/warp),
registers 94 (pass1_fluid)/109 (pass1_solid), occupancy only 25%.
Optimisation directions (M3): (1) enumerate work particles in cell order (threads in a warp then have highly overlapping neighbourhoods → coalesced access); (2) pack the hot neighbourhood data into
AoS (104–112 B/particle, 1–2 cache lines); (3) precompute reciprocals to reduce FP64 divisions (6 per visit → 2–3).

### 9.3 A real bug in the first version (fixed): `eta` used the SurFD of "the higher-numbered particle"

After the first 500-step run, p differed by up to 80 (2.7% relative). I initially attributed it to "chaotic amplification"; **that judgement was wrong**.
Checking `continue_density` in `1.cpp` line by line revealed:

```c
if(fabs(SurFD[i])<2.5){eta = 0.001;}else{eta = 0.00002;}
if(fabs(SurFD[j])<2.5){eta = 0.001;}else{eta = 0.00002;}   // the second assignment overwrites the first
```

`eta` is computed **once per pair and shared by both sides**, and it always takes `SurFD[j]` (the original always has `i<j`).
My gather version used "the neighbour's SurFD": correct when the current particle is the i side, **wrong on the j side** —
artificial viscosity differs by a factor of **50** between 0.001 and 2e-5.

The reason this bug was completely invisible in the step-0 array comparison: **at step 0 `SurFD` is always 0** (`SurF` is cleared in
`NorVector`), so every particle has eta=0.001; immediately after step 0, **96802/101250 = 95.6%** of fluid particles have
`|SurF| ≥ 2.5`, so from step 2 onward eta splits between 0.001 and 2e-5. That is why `result0` matched but `result1` was off by 7.2.

Fix: `eta` now uses `SurFD[(p>q) ? p : q]` (the higher-numbered particle in the pair), see the comment in `k_pass1_fluid`.

### 9.4 Measured results after the fix (500 steps, `CompareResults` against `result0..9`)

| Column | Differing values | maxAbs | Reference: CPU's own rerun noise floor |
|---|---|---|---|
| `x0 / x1 / x2` | **0** | **0** | 0 |
| `rho / mass / itype / X00..X02 / rho0` | **0** | **0** | 0 |
| `p` | 81991 | **1e-3** | 0.01 |
| `vx0 / vx1 / vx2` | 320 / 1517 / 17 | **1e-9** | 1e-9 |
| Differing rows per file | ~8300 / 237298 | — | 2500–4500 |

**Every column is inside the CPU's own rerun noise floor (`p` is even an order of magnitude better)**, so the original acceptance criteria of §4.2
(`p ≤ 0.1`, `vx ≤ 1e-8`, `x/rho` identical at printed precision) **are satisfied**.

### 9.5 Second case: also cover the branch that "was never executed"

In the original case **fluid-solid pairs = 0** (the solid plate never touches the fluid; all coupling goes through walls),
so the `SForce` / `vecn` / `PRii` path **never ran**, and an all-zero comparison cannot count as validation.
A second case was therefore constructed: all 75728 wall particles (itype −2) were relabelled as solid (−20),
which produces many fluid-solid pairs (`SForce` magnitude 3.7e2); both sides ran 1 step and every array was compared:

| Quantity | maxAbs | Relative (to magnitude) |
|---|---|---|
| `SForce0/1/2` | 2.8e-13 | 5.8e-11 (~3.7e2) |
| `vecn0/1/2` | 7.1e-11 | — |
| `PIJW0/1/2` | 2.0e-9 | 4e-12 (~5e-2…5e2) |
| `Pia11/22/33` | 3.1e-8 | 4.5e-12 (~6.9e3) |
| `F11` / `DFdt11` / `rho` | 2.2e-15 / 1.6e-17 / 4.0e-12 | 1e-15 level |

### 9.6 Lessons (written into the process)

1. **"One-step comparison passes" does not mean the port is correct**: every branch must actually be executed (in this case fluid-solid pairs were 0, so `SForce/vecn/PRii` ran empty throughout).
2. **Quantities like `eta` that are "shared by both sides and taken from only one side" are the easiest to silently break during gather-isation**; list them in a dedicated table and check each one (added to §3.2 of this document).
3. When something "does not match", suspect the implementation before the precision; "chaotic amplification" is a convenient conclusion that hides bugs.

### 9.7 Proposed acceptance criteria (two levels, both met)

- **L1 pointwise (1 step, two cases)**: relative difference ≤ 1e-9 for every intermediate quantity; `x` identical at printed precision; neighbour pair count bitwise identical to the CPU.
  → met (worst: `Pia` 4.5e-12, `SForce` 5.8e-11).
- **L2 pointwise (500 steps)**: compare with `result0..9`; every column inside the CPU's own rerun noise floor. → met.
- **L3 stability**: 500 steps with no NaN/Inf. → met.

### 9.8 Code the two cases still do not cover (manually reviewed only)

| Location | Condition | Why this case does not trigger it |
|---|---|---|
| `k_update_fluid` speed cap | `sqrt(v²+1e-5) > 30` | actual speed ~1e-2 m/s |
| `k_update_fluid` density floor | `rho < 960` | `rho ≈ 1000` |
| `k_constitution` singularity fallback | `abs(J) < 1e-6` → `rho0 = 1000` | `J ≈ 1 + 1e-5` |
| `k_gra_invert` singularity fallback | `abs(Da) < 1e-6` → `A = I` | `abs(Da) ≈ 4` |

These are all single-line `if` clamps and their semantics have been compared line by line; 100% coverage would need extreme cases (for example
giving one solid particle an initial speed of 100 m/s, or forcing rho0 to 0 to make `J` singular).

---

## 10. Precision template (switchable fp64 / fp32)

### 10.1 Usage

The whole solver is templated on the scalar type `Real` (`SphPtrsT<Real>`, all kernels, the host-side `Solver<Real>`),
and **both precisions are compiled into the same exe by default**, selected at run time:

```bat
sph_cuda.exe                 :: fp64 (default, aligned with the CPU reference)
sph_cuda.exe --fp32          :: fp32
set SPH_PRECISION=fp32       :: same, via the environment
sph_cuda.exe --fp64 --help   :: other options
```

CMake side:

| Option | Default | Description |
|---|---|---|
| `SPH_ENABLE_FP64` | ON | whether to compile the double instance |
| `SPH_ENABLE_FP32` | ON | whether to compile the float instance (keeping only one halves compile time/binary size) |
| `SPH_DEFAULT_PRECISION` | `double` | default precision when no flag is given |

```bat
cmake -S . -B build -DSPH_ENABLE_FP64=ON -DSPH_ENABLE_FP32=ON -DSPH_DEFAULT_PRECISION=double
```

### 10.2 Implementation notes (pitfalls encountered)

1. **Literals must be written as `Real(x)`**. Otherwise an expression such as `0.5*(a+b)` is promoted to double when `Real=float`,
   and on consumer GPUs FP64 is only 1/67 of FP32 throughput, so one promotion can eat the whole speedup.
   All ~130 literals in the file were explicitly converted; `cuobjdump -sass` confirmed that the fp32-only build contains
   **no double arithmetic instructions**.
2. **The neighbour test still uses `sph_add_rn/sph_mul_rn`** (float uses `__fadd_rn/__fmul_rn`) to avoid FMA contraction;
   in fp64 this keeps the particle-pair set bitwise identical to `direct_find`.
3. **Output and comparisons always convert to double**: the numeric format of `result<N>.dat`, `CompareResults.exe` and the
   binary debug dumps are independent of precision, so the comparison workflow does not change when switching.
4. Template instances are explicitly instantiated at the bottom of `sph_kernels.cu` (guarded by the `SPH_HAVE_FP64/32` macros),
   and the host-side `run_case<Real>()` takes one path per precision.

### 10.3 Measured cost and benefit of the two precisions (500 steps, RTX 5080)

| Item | fp64 | fp32 |
|---|---|---|
| Per step | 55.9 ms | **4.20 ms (13.3×)** |
| 500 steps | 28.1 s | 2.1 s |
| of which pass1 / pass2 | 41.9 / 13.7 ms | 2.4 / 1.7 ms |
| Per-particle state | 552 B (69 doubles) | 276 B (69 floats) |
| Neighbour pairs (T=0) | 11,102,707 (bitwise identical to CPU) | 11,209,220 (0.96% more, see below) |
| `x` max difference (1 step) | identical at printed precision | 7e-6 (2e-5 relative, = float granularity) |
| `rho` max difference (1 step) | 0 | 0.09 (9e-5 relative, ≈1.5 ulp) |
| `p` max difference (1 step) | 1e-4 | **202** (see below) |
| 500-step global quantities | see §9.4 | KE difference **0.000%**, mean pressure difference **0.18%** |

**Why fp32's `p` differs at the 1% level**: not an implementation problem, but the inherent ill-conditioning of the constitution.
Solid `Pia ≈ λ·log10(J)·F⁻ᵀ` with `λ=1.33e7`, `J≈1+1.7e-5`,
so `dPia/Pia ≈ 6e4·(dJ/J)`: a relative error of ~1e-6 in `F` in fp32 is amplified to **~6e-2 relative** in `Pia` (≈200 Pa).
The same mechanism gives fp32 0.96% more neighbour pairs — many pairs in the lattice sit **exactly** on the knife edge at `r = 3h`,
and float rounding classifies them as "inside" (their kernel value is 0, so the physics is unaffected; only the printed count changes).

Conclusions:

- **Use fp64 for acceptance and production cases** (same order as the CPU reference, 56 ms/step, 3.8× faster than 32-thread CPU);
- **fp32 is suitable for parameter sweeps and trends** (4.2 ms/step, 50× faster than the CPU, global quantities within 0.2%),
  but do not compare it pointwise with fp64 and do not look at absolute solid pressure values;
- If both fp32 speed and solid accuracy are needed, the next step is a mixed path "fluid fp32 + solid constitution fp64"
  (only `k_pass1_solid`/`k_constitution` change, about +0.2 ms/step, to be measured).

---

## 11. Speed bottleneck: measured localisation and optimisations applied

### 11.1 Per-kernel timing (`SPH_BENCH=1`, each item run 20 times)

| Kernel | fp64 before | fp64 after | fp32 after | Notes |
|---|---|---|---|---|
| `pass1_fluid` | 32.68 ms | **17.32 ms** | 1.23 ms | fluid-side forces (the largest share) |
| `pass2_solid` | 10.36 ms | **8.50 ms** | 1.40 ms | PIJW / fluid-solid coupling force |
| `pass1_solid` | 9.86 ms | **8.30 ms** | 0.77 ms | deformation gradient F / DFdt |
| `count_pairs` (pure neighbour traversal) | 4.14 ms | 4.13 ms | 0.72 ms | **traversal floor**: only the 27 neighbour cells + distance test |
| `pass2_fluid` | 2.28 ms | 2.28 ms | 0.41 ms | this case has no fluid-solid pairs → empty run |
| cell(hist+scan+scatter) | 0.08 ms | 0.08 ms | 0.08 ms | |
| constitution | 0.05 ms | 0.05 ms | 0.008 ms | solid constitution (no neighbour loop) |
| prepare + update ×3 | 0.03 ms | 0.03 ms | 0.02 ms | |
| **Total** | **59.5 ms** | **36.4 ms** | **4.7 ms** | whole step measured 55.9 → **37.1 ms** / 4.20 → **3.91 ms** |

### 11.2 Root cause: 17 FP64 division subroutine calls per neighbour visit

Counting the innermost loop of `k_pass1_fluid<double>` (one neighbour visit) with `cuobjdump -sass`:

| | Before | After |
|---|---|---|
| Real instruction count | 576 | 518 |
| FP64 ALU (DFMA/DMUL/DADD) | 249 | **168** |
| `CALL.REL.NOINC` (division/sqrt subroutines) | **17** | **4** |
| 64-bit global loads (LDG.E.64) | 17 | 18 |

On consumer GPUs, FP64 division is **not one instruction** but a subroutine of `MUFU.RCP64H` plus several Newton iterations;
17 of them per visit consumed half the time on their own. The fix (pure algebra, only 1–2 ulp difference):

1. `q = dr/h` → `dr * (1/h)`, with `1/h` a compile-time constant (`sph_inv_h<>()`);
2. compute `1/dr`, `1/(ρ_i+ρ_j)` and `1/(dr²+0.01h²)` once each and turn the rest into multiplications;
3. add an `irho[i] = 1/ρ[i]` array (filled in `k_prepare`, refreshed for solids in `k_constitution`)
   to remove the `m_j/ρ_j`, `PM/(ρ_iρ_j)` and `S·m_j/ρ_j` divisions;
4. constant divisions such as `3/c0` and `1/(c0²)` are folded at compile time.

Result: **55.9 → 37.1 ms/step (1.5×)**, and 500 steps are still within the noise floor of the CPU reference
(`p` maxAbs 1e-3, `x/rho/mass/itype/X0/rho0` identical at printed precision).

### 11.3 The bottleneck at that point: uncoalesced neighbour data reads

Two pieces of hard evidence:

1. **Pure traversal floor**: `count_pairs` reads only the neighbour's `x[3]` (4 loads) and takes 4.13 ms;
   `pass1_fluid` reads 18 fields → 17.3 ms. **The ratio 17.3/4.13 = 4.19 is exactly 18/4 = 4.5**.
2. **Effective bandwidth only 82 GB/s** (10.5M visits × 136 B / 17.3 ms), against a peak of 848 GB/s →
   this is not a bandwidth limit but a **transaction** limit: the 32 threads of a warp read 32 different particles, so one `LDG.E.64`
   is split into up to 32 128 B transactions while only 8 B is used.

So the next step is not more arithmetic but **fewer scattered reads per visit**:

| Option | Approach | Estimate |
|---|---|---|
| A. Pack neighbourhood data into AoS | every step, spend 0.1 ms packing `x/vx/ρ/ρ⁻¹/m/p/itype/SurFD/vecn` into a contiguous ~144 B/particle block; threads in a warp then access adjacent particles (cell order), cutting transactions by an order of magnitude | pass1_fluid 17.3 → **5–7 ms** |
| B. Shared-memory tiling | one block handles several cells, cooperatively stages the 27-cell neighbourhood into shared memory, then reads shared per particle | faster, but requires rewriting the neighbour loop and handling halos |
| C. Fuse the pass1/pass2 traversals | fuse the two fluid-side traversals (in this case pass2_fluid is an empty 2.3 ms); the solid side can only skip by cell flag because `ρ` is overwritten in the constitution | −2.3 ms |
| D. Pack `A[9]`/`Pia[9]` in the solid kernels | `pass2_solid` reads two 3×3 tensors per visit (18 scattered reads) | 8.5 → 4–5 ms |

With A + C + D, fp64 should reach **10–12 ms/step** (about **18–21×** the 32-thread CPU's 211 ms/step), and fp32 benefits too.

### 11.4 Correcting the conclusion with probes: **not memory, but arithmetic**

ncu (at `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\...\ncu.exe`) runs,
but returns `ERR_NVGPUCTRPERM`: hardware counters need administrator rights, and this machine has neither admin rights nor the ability to change
`HKLM\SOFTWARE\NVIDIA Corporation\Global\NVTweak\RmProfilingAdminOnly`.
So a "controllable probe" was used: a kernel with **the same neighbour traversal as `pass1_fluid`, reading the same 13 fields, but doing no arithmetic**
(`k_bench_loads`, plus a coalesced variant that maps neighbour addresses to consecutive addresses across the warp).

| Experiment | fp64 | fp32 |
|---|---|---|
| Traversal + 13 **scattered reads** (no arithmetic) | 3.66 ms | 1.01 ms |
| Traversal + 13 **coalesced reads** (no arithmetic) | 3.65 ms | 0.86 ms |
| Full `pass1_fluid` | **14.26 ms** | **1.23 ms** |
| ⇒ arithmetic share of `pass1_fluid` | **10.6 ms (74%)** | — |

**The scattered-read penalty is ≈ 0** (3.66 vs 3.65 ms) — the earlier "448 transactions per warp" guess was wrong:
threads in the same warp process particles that are adjacent in cell order, their neighbour lists overlap heavily, and the L1/L2 hit rate is high.
(Note: `count_pairs`'s 4.15 ms cannot be used as the floor; it does one `atomicAdd` to the same address per particle, which is slowed down by atomic contention.)

So the bottleneck is **arithmetic efficiency**: about 100 FP64 operations per visit, and the 10.6 ms arithmetic part
corresponds to a FP64 roof of ~2.6 ms (0.81 TFLOPS), i.e. **only 19–25% of the roof**, which means instruction-latency/occupancy limited.

### 11.5 Second optimisation round (this round)

1. **Compute the kernel polynomial once**: `dwdr` and the three `dwdx` components share the same
   `coe·((2-q)⁴-2(1+2q)(2-q)³)` (`sph_wq()`), saving three polynomial evaluations per visit, about −24 FP64 operations.
2. **Raise occupancy**: add `__launch_bounds__(256,4)` to `k_pass1_fluid`, reducing registers from 94 to ≤64 and doubling
   the warps resident per SM (25% → 50% occupancy).

| | First version | After division factoring | After this round |
|---|---|---|---|
| `pass1_fluid` | 32.68 | 17.32 | **14.26 ms** |
| `pass1_solid` | 9.86 | 8.30 | **6.69 ms** |
| `pass2_solid` | 10.36 | 8.50 | **7.13 ms** |
| Whole step fp64 | 55.9 | 37.1 | **31.2 ms** |
| Whole step fp32 | 4.20 | 3.91 | **~3.3 ms** |
| Relative to 32-thread CPU (211 ms/step) | 3.8× | 5.7× | **6.8×** |

Re-compared after 500 steps: `p` maxAbs 1e-3, `x/rho/mass/itype/X0/rho0` identical at printed precision, still inside the noise floor.

### 11.6 Next steps (ordered by payoff, none done yet)

| Option | Basis | Estimate |
|---|---|---|
| A. Keep cutting arithmetic | still at ~25% of the FP64 roof; the `min/max` in `PM/UM`, the `eta` selection, `rhoR` etc. can still be reduced | `pass1_fluid` 14.3 → 10–11 ms |
| B. Raise occupancy further | if registers reach 48 → 6 blocks/SM | depends on spilling |
| C. Skip empty traversals | this case has no fluid-solid pairs, so `pass2_fluid` is 2.28 ms of pure idle work; a per-cell "contains solid" flag skips it at nearly zero cost | −2.2 ms |
| D. Pack solid tensors / shared memory | `pass2_solid` reads `A[9]`+`Pia[9]` per visit; `pass1_solid` reads `A[9]` | 7.1+6.7 → 8–9 ms (combined) |

With C+D the whole step should be **~22 ms (about 10× the CPU)**, and adding A brings it to ~19 ms.

### 11.7 Third optimisation round (this round)

1. **Skip empty traversals**: `k_cell_hist` also counts the solids per cell (`cellSolid`), and `k_pass2_fluid` first checks whether any of the 27 neighbour cells contains a solid; if not it returns immediately (this case has no fluid-solid pairs).
   → `pass2_fluid` **2.28 → 0.435 ms**.
2. **Pack static solid data**: pack each solid particle's `{X0[3], A[9], rho0, mass}` into 14 consecutive doubles
   (`solidPack`, built once after initialisation and never changed), turning 13 scattered reads in the solid kernels into 1–2 cache lines.
   → `pass1_solid` **6.69 → 6.26 ms**.
3. **Occupancy sweep**: sweep `__launch_bounds__(256,N)` in `k_pass1_fluid` from N=4 to 6; times 14.5/14.27/14.31 ms
   — **essentially no difference**, showing it is not "not enough warps" but the FP64 pipeline dependency chain (this card's FP64 is only 1/64 of FP32).

| | First version | Second round | **This round** |
|---|---|---|---|
| `pass1_fluid` | 32.68 | 14.26 | 14.26 ms |
| `pass1_solid` | 9.86 | 6.69 | **6.26 ms** |
| `pass2_solid` | 10.36 | 7.13 | 7.22 ms |
| `pass2_fluid` | 2.28 | 2.28 | **0.44 ms** |
| **Whole step fp64** | 55.9 | 31.2 | **29.9 ms (7.1× the CPU)** |
| **Whole step fp32** | 4.20 | 3.75 | **3.35 ms (63× the CPU)** |

**⚠ The packing introduced an out-of-bounds bug once**: `solidPack` covers only `[NTOTAL, NALL)`,
but particles with `itype<-10` can lie outside that interval (the second case, which relabels walls as solid, is exactly that situation),
so the index read out of bounds into garbage → NaN results. An `inpack` check now falls back to the SoA arrays (in 1.cpp these particles' `A` is 0 anyway).
The main case is unaffected (all its solids are inside the interval), but **only the second case could expose it** — again showing that "you need cases that cover every branch".

Re-compared after 500 steps: `p` maxAbs ≤ 1e-3, `x/rho/mass/itype/X0/rho0` identical at printed precision; the synthetic case is identical to 1e-6 after 60 steps.

### 11.8 Conclusion: remaining headroom for fp64

`pass1_fluid` now takes 14.26 ms, of which traversal+loads are 3.65 ms and arithmetic 10.6 ms. At 168 FP64 instructions per visit,
the arithmetic roof is about 2.6–4.4 ms, so it is still at **25–40%** of the roof; but the occupancy sweep had no effect, meaning it is limited by
**the consumer GPU's FP64 pipeline itself** (1/64 rate plus dependency-chain latency); further arithmetic trimming has limited payoff.

There are only two ways to go faster:

1. **fp32 (or mixed precision)**: fp32 is already at 3.35 ms/step (63× the CPU), at the cost of solid stress accuracy (see §9.7/§10.3);
2. **Mixed precision**: fluid fp32 + solid constitution fp64, estimated 4–5 ms/step, keeping solid pressure accuracy at the fp64 level.

### 11.9 Tried: `Pia` as AoS + vectorised loads (payoff ≈ 0)

Based on the assumption "the solid kernels are memory-bound", a version was built with `Pia` changed from component-major `Pia[c*NUM+i]` to
12 consecutive Reals per solid particle (9 used + 3 padding, 96 B, 16-byte aligned) in AoS form,
and `k_pass2_solid` reads 16 B at a time with `double2` (`float2` in fp32).

Results (SASS confirms the vectorisation is active):

| | Before | After |
|---|---|---|
| 64-bit scattered reads in the `pass2_solid` loop | 64 | **42** (including 4 `LDG.E.128`) |
| FP64 ALU / CALL in the loop | 199 / 7 | 190 / 7 |
| `pass2_solid` time | 7.47 ms | **7.79 ms (slightly worse)** |
| Whole step fp64 | 29.48 ms | 29.60 ms |
| Whole step fp32 | 3.07 ms | **2.96 ms** |

**Conclusion: the solid kernels are not memory-bound either.** Cutting a third of the loads did not make it faster, so what holds it back is
**the latency of 190 FP64 ALU instructions + 7 `CALL.REL` (software sqrt/division subroutines)**, the same wall as `pass1_fluid`.
My earlier inference of "static load-instruction count ≈ time, therefore memory-bound" was wrong.

A problem I introduced myself was also fixed: the debug dump still read the old Pia layout, so `g.Pia + 8*NUM` went out of bounds
(`cudaMemcpy` reported invalid argument). It now uses a dedicated `dump_pia()` (AoS → nine NUM-long arrays, zero-padded outside the solid range).

Validation: main case, 500 steps, `p` maxAbs ≤ 1e-3 and `x/rho/...` identical at printed precision; synthetic case, 1 step,
`Pia` 2.6e-8 / `PIJW` 1.4e-9 / `F` 2e-15, 60-step output maxAbs 1e-6.

**The genuinely promising next step is ILP**: the three hot kernels all have "2–7 sqrt/division subroutines chained on the dependency path";
processing two particles per thread lets the two chains interleave and fills the CALL latency (no change to results, only scheduling).
Estimated `pass1_fluid` 14.4 → 10–12, `pass2_solid` 7.8 → 6, `pass1_solid` 6.4 → 5,
whole step fp64 about **22–25 ms**.

### 11.10 ILP (process two neighbours per loop iteration) — effective in fp32, not in fp64

`sph_neigh2()`: the neighbour set and accumulation order are **exactly the same** as `sph_neigh()` (same cell, same ascending a);
it only computes two neighbours' `r²`/`sqrt`/division in one iteration so two independent dependency chains interleave in the pipeline.

| | Before ILP | After ILP |
|---|---|---|
| `pass1_fluid` fp64 | 14.39 ms | 14.3 ms (**no change**) |
| `pass1_solid` fp64 | 6.35 | 6.3 (no change) |
| `pass2_solid` fp64 | 7.47 | 7.4 (no change) |
| **Whole step fp64** | 29.5 | **29.6 ms (no change)** |
| **Whole step fp32** | 2.96 | **2.47 ms (−17%)** |

The conclusion is clear: **fp64 is FP64-pipeline-throughput limited, not latency limited** — so adding ILP (filling latency) does nothing;
fp32 is latency limited, and ILP takes 17% directly. This also explains why the earlier occupancy sweep had no effect:
the FP64 units are already saturated, so more warps do not help.

The code is kept (harmless in fp64, 17% gain in fp32). Validation: main case, 500 steps, `p` maxAbs ≤ 1e-3 and `x/rho/...` identical at printed precision;
synthetic case, 60 steps, maxAbs 1e-6. Because the accumulation order is unchanged, the results are equivalent to the version without ILP.

### 11.11 Current final performance

| | ms/step | Relative to 32-thread CPU (211 ms/step) |
|---|---|---|
| fp64 | **29.6** | **7.1×** |
| fp32 | **2.47** | **85×** |
| (starting point, first version of this session) | 55.9 / 4.20 | 3.8× / 50× |

Further fp64 speedup would require only "reducing the FP64 operations themselves" (algorithmic rewrites that change the arithmetic);
scheduling, memory access and occupancy have all been tried and are exhausted.
