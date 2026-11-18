# Work log: static solid pair list (C1–C3), P1 and P2

| | |
|---|---|
| Date | 2026-11-19 |
| Base | `d284042` — the tree Claude reviewed in [`review/Claude/REVIEW.md`](../../review/Claude/REVIEW.md) |
| Machine | RTX 5080 (`sm_120`, 84 SMs), CUDA 13.3, Windows 11, MSVC 2022 toolset |
| Case | `input.dat`, 237 298 particles (101 250 fluid, 60 320 solid, 75 728 wall), `dt = 1e-5` |
| Scope | every correctness finding of the review except C7, plus P1 and P2 |

This document is the record of the change; the design rationale lives in
[`cuda/docs/DESIGN.md`](../../cuda/docs/DESIGN.md) (§6.4, §12, §13), and the
performance detail (profiles, per-kernel numbers, what is left) in
[`docs/perf-work-hrb-jian`](../perf-work-hrb-jian/README.md).

---

## 1. What was wrong

### C1 — the static solid list was searched in the wrong configuration (real defect)

`solid_pair_count` / `solid_pair_fill` walk the 27 cells around
`cell(X0_p)`, but the cell list they walked had been binned from the **current**
positions `x`.  A partner `q` was therefore found only if `cell(x_q)` happened to
lie within ±1 cell of `cell(X0_p)`.  Since the list is built **once**, a restart
whose solid already carries a displacement of about one cell keeps the wrong
stresses for the entire run — silently.

Reproduction: shift the solid rigidly by `+dy` (a translation changes no
reference distance, so it must change nothing) and compare the list and the
stresses after one step, fp64:

| offset `dy` | list size | max abs error of `trace(Pia)` |
|---|---:|---:|
| 0 | 5 371 593 | — |
| 2 mm | 5 372 031 | 4.8e-8 Pa |
| 5 mm | 5 372 031 | 4.8e-8 Pa |
| 10 mm | 5 368 343 | 4.4e-8 Pa |
| 15 mm | 5 367 871 | 5.6e-8 Pa |
| **20 mm** | 5 163 935 | **5.34e+6 Pa** |
| 25 mm | 5 068 175 | 5.34e+6 Pa |
| 25 mm, `SPH_SOLDDYN=1` (control) | — | 5.9e-8 Pa |

It is a cliff, not a gradual degradation: nothing at 15 mm, then 5.3 MPa — four
orders of magnitude above the p99 stress (4.6 kPa) — and it persists, because the
list is never rebuilt.

### C2 — knife-edge pairs

The lattice puts a large part of the solid/solid pairs exactly at `r = 3*hsml`.
There `qr` can round to `2 - 1 ulp`, so `wq` returns a denormal (≈1e-40 in fp64,
≈1e-20 in fp32) instead of zero and the pair is stored.  In fp64 that was
527 049 of 5 371 593 entries (10 %), in fp32 655 647 (12 %); they only add a
quantity far below the ulp of every accumulator.

### C3 — the count and fill passes were not cross-checked

`solid_pair_fill` wrote from `start[k]` with no bound and trusted
`solid_pair_geom` to decide identically in a separately compiled kernel.  A
1-ulp disagreement would silently shift the range of every following particle.

---

## 2. What was changed

### 2.1 Correctness

| finding | change |
|---|---|
| C1 | `build_solid_lists()` now builds a cell list from a copy of the solver state whose `x` points at `X0`, builds the list, then rebuilds the live cell list from `x` (`build_cells_for()` / `build_cells()`). |
| C2 | `solid_pair_geom()` rejects `Dr >= 3*hsml*(1 - eps)` with `eps = 1e-9` in fp64 and `1e-6` in fp32 — the margin has to be representable in `Real`, otherwise the fp32 expression equals `3*hsml` and the filter is a no-op. |
| C3 | `k_solid_pair_fill` returns its per-particle count; the host compares it with `k_solid_pair_count` and aborts with the first offending particle (`exit(3)`). |
| C4 | `load_input()` validates the type layout (`fluid > 0` in `[0, NTOTAL)`, `solid < -10` in `[NTOTAL, NALL)`, `wall in [-10, 0]` after that), rejects positions outside the fixed grid, and warns about records past `SPH_NUM`.  A permuted deck would break the `eta = f(SurFD[max(p,q)])` rule, so it is refused rather than run. |
| C5 | `step()` ends with `CUDA_CHECK(cudaGetLastError())`. |
| C6 | the `--fmad=true` comment in `cuda/CMakeLists.txt` said the opposite of what the flag does; corrected. |
| C8 | `k_gra_invert` counts its `A = I` fallbacks; the host warns, and refuses fp32 (whose `F - I` displacement form needs `a*A = I`) when the count is not zero. |
| R1 | GCC/Clang builds of the CPU reference now use `-O3 -ffp-contract=off`; MSVC already used `/fp:precise`, which is what the validated numbers were produced with. |
| R2 | `DESIGN` §12.1 states the condition under which A3 matches the reference's current-configuration pair list (no pair stretching by more than ≈6 %) instead of claiming the reference's pair set is constant. |
| R3 | `README.md` notes that the printed `p` is one update older than `x`, `vx`, `rho` in the same row, so `p != c0^2 (rho - 1000)`. |

### 2.2 Storage: CSR → slot-major ELL (P2)

Entry `j` of solid particle `k` now lives at `e = j*SOLNUM + k`, and the nine
tensor components sit in nine separate arrays (component `c` at `c*sw + e`).
A warp walking step `j` therefore touches 32 consecutive slots, instead of 32
CSR segments that were ~89 entries apart.

The list is now exactly the set of reference pairs with
`|X0_p - X0_q| < 3*hsml`: **4 844 544 pairs**, ELL width 92 (padding 14.6 %),
402 MB in fp64 and 212 MB in fp32.  Both precisions store the same set; before
the C2 filter they stored 5 371 593 (fp64) and 5 500 191 (fp32).

### 2.3 fp64 `pass1_fluid` in two phases (P1)

The fused kernel runs its FP64 body whenever *any* lane of a warp has a hit,
while only ~15 % of the candidates are hits.  P1 first records each fluid
particle's neighbours — in exactly the order the fused traversal visits them —
and then runs the identical body over that list.

* `fnbr[j*NTOTAL + p]`: slot-major, 192 slots per particle (77.8 MB).
* Measured occupancy of this deck: mean 103.9, min 42, max 115 neighbours.
  `fcnt[p] < 0` marks a particle that did not fit; that particle alone falls
  back to the fused traversal, so the width is a memory knob and never a
  correctness limit.
* Both paths are one templated kernel (`k_pass1_fluid_body<Real, USE_LIST>`), so
  the arithmetic is the same source.  `SPH_FLUIDLIST=0` selects the fused path at
  run time for A/B comparison.
* fp32 keeps the fused kernel: it is traversal bound there (fused body 0.77 ms
  against 0.56 ms for the traversal alone), and the split was measured as a loss.

---

## 3. Verification

### 3.1 The regression test

`cuda/tools/check_rigid_translation.py` is the C1 test: one step at offset 0 and
one at +25 mm, comparing the list size and `trace(Pia)`.

```
> python cuda\tools\check_rigid_translation.py --exe cuda\build\sph_cuda.exe --input input.dat
[rigid] unshifted list: 4844544 pairs
[rigid] +0.025 m     list: 4844544 pairs
[rigid] |trace(Pia)| reference: median 593, p99 4.59e+03 Pa
[rigid] delta trace(Pia): max 5.81e-08, p99 3.12e-08 Pa
[rigid] PASS
```

Against the pre-fix binary it reports `5371593` vs `5068175` pairs and
`5.34e+06 Pa`, and exits with 1 — the test can fail.

### 3.2 Against the CPU reference (500 steps)

`_runs/cpu` = 32-thread reference run; files print `%.6g`, so "identical" means
identical to six significant digits.

fp64, `result9` (step 500):

| quantity | rows differing | max difference |
|---|---:|---:|
| `x0 x1 x2`, `rho`, fluid `p`, `mass`, `itype`, `X0`, `rho0` | 0 / 237 298 | — |
| solid `trace(Pia)` | 8 103 / 136 048 | **6.8e-8 Pa** |
| `vx0 vx1 vx2` | 45 / 60 / 6 | **1e-9 m/s** |

Split by the magnitude of the stress being compared:

| `|trace(Pia)|` | particles | max difference |
|---|---:|---:|
| ≥ 1000 Pa | 20 274 | 0 |
| 100 – 1000 Pa | 29 762 | 0 |
| 1 – 100 Pa | 1 468 | 0 |
| < 1 Pa | 84 544 | 6.8e-8 Pa |

So every particle carrying a meaningful stress is identical at the printed
precision, and the largest difference anywhere is 6.8e-8 Pa — 4e-12 of the peak
stress (1.82e4 Pa).  The pre-fix build measured the same 6.831e-8 Pa, i.e. this
change did not move the result away from the reference.

fp32, `result9`: `x` ≤ 1e-6 m, `vx` ≤ 1e-5 m/s, `rho` ≤ 0.01 kg/m³, fluid `p`
≤ 0.14 Pa, solid `trace(Pia)` ≤ 0.17 Pa (8.2e-4 relative on particles above
100 Pa).  Same order as the pre-fix fp32 build; the floor is the float
quantisation of `p = c0^2(rho - 1000)`, i.e. 0.14 Pa per step at `rho ≈ 1000`.

### 3.3 Post-contact

700-step runs (fluid/plate contact is at step 566) of the old and the new
binary, fp64, compared at step 650: `x`, `p`, `rho`, `mass`, `itype`, `X0`,
`rho0` identical at the printed precision; `vx` differs by ≤ 1e-10 m/s on a few
hundred particles.

### 3.4 Reproducibility and sanitizers

* Two identical 500-step runs are byte-identical in both precisions
  (`SHA256` over `result0..9`).
* `compute-sanitizer` memcheck (fp64 and fp32) and racecheck: 0 errors.
* `load_input()` validation was exercised with two bad decks: one with a
  permuted particle type (`particle 5: itype -20 not allowed in this index
  range`) and one with a position outside the grid (`particle 3: x[0] = 0.95
  outside the grid [-0.1, 0.7]`).  Both are refused before anything runs.

### 3.5 One deliberate incompatibility

Output is **not** byte-identical to the pre-fix build any more.  Fixing C1
changes which cell list the list is searched in, which changes the *order* in
which a solid particle's neighbours are summed (the set is now the correct one),
so `F`, `DFdt`, `Pia` and `PIJW` move by 1e-16 … 1e-8 absolute.  That is
rounding rearrangement, not new physics: the pair set is provably the correct
one (it equals brute force), run-to-run reproducibility is kept, and agreement
with the CPU reference is unchanged (3.2).  The trajectory over 650 steps does
not amplify the difference (3.3).

---

## 4. Results

500 steps, same binary, same deck, per precision:

| | before | after | speed-up | vs 32-thread CPU (163 ms/step) |
|---|---:|---:|---:|---:|
| fp64 | 12.412 ms/step | **5.006** | **2.48x** | **33x** |
| fp32 | 2.434 ms/step | **1.634** | **1.49x** | **100x** |

(The CPU baseline is the wall clock of a 500-step CPU run, 81.4 s.  Earlier
documents quoted 211 ms/step; that figure is not reproducible on this machine,
and [`docs/perf-work-hrb-jian`](../perf-work-hrb-jian/README.md) §10 has the
decomposition — the compute-only ratio is 26x / 80x.)

Per kernel (fp64, `SPH_BENCH=1`):

| kernel | before | after |
|---|---:|---:|
| `pass1_fluid` | 8.77 | 3.54 (list build 0.87 + body 2.67) |
| `pass1_solid` | 1.05 | **0.34** |
| `pass2_solid` | 1.14 | **0.62** |
| cell list | 0.13 | 0.13 |
| full step | 11.44 | **5.01** |

Per kernel (fp32):

| kernel | before | after |
|---|---:|---:|
| `pass1_solid` | 0.698 | **0.166** |
| `pass2_solid` | 0.751 | **0.296** |
| `pass1_fluid` (fused) | 0.768 | 0.768 |

Start-up (fp64): the A3 list takes 0.17 s to build, including the C3 check and
the second cell-list build.

---

## 5. Files touched

| file | change |
|---|---|
| `cuda/src/sph.h` | `sCnt` / `swidth` / `sw` / ELL arrays replace `sStart`; `fnbr` / `fcnt` / `fwidth`; `launch_fluid_nbr_build`, `gra_fallback_count` |
| `cuda/src/sph_kernels.cu` | ELL fill and both solid kernels; C2 filter; `k_fluid_nbr_build` and the templated `k_pass1_fluid_body<Real, USE_LIST>`; C8 fallback counter |
| `cuda/src/sph_host.cu` | `build_cells_for`; `build_solid_lists` (X0 cell list, ELL width, C3 assertion, size guard); C4 validation; C5 per-step launch check; C8 refusal; `SPH_FLUIDLIST`; bench rows |
| `cuda/tools/check_rigid_translation.py` | new: the C1 regression test |
| `cuda/CMakeLists.txt` | C6 comment |
| `cpu/CMakeLists.txt` | R1: `-O3 -ffp-contract=off` for GCC/Clang |
| `cuda/docs/DESIGN.md` | §6.4 numbers, §12 rewritten, §12.5 the C1 record, §13 post-review work |
| `README.md`, `cuda/README.md` | performance table, validation, `SPH_FLUIDLIST`, R3 note |
| `.gitignore` | `_runs/` |

---

## 6. Not done

| item | why |
|---|---|
| C7 (`k_cell_sort` is an insertion sort per cell) | O(n²) only when the clamp collects many particles into one boundary cell.  For this deck the worst cell holds a few hundred.  A stable radix sort on `(cell, index)` would give the same order in O(n) but changes the deterministic-order guarantee, so it wants a run that actually hits the cliff. |
| P3 (overlap the result writer) | after P1/P2 a result file costs ~55 ms, i.e. ~22 % of the wall clock in fp64 and ~40 % in fp32 at the 50-step cadence.  Next largest item. |
| P4 (launch overhead, CUDA graph) | one `cudaDeviceSynchronize()` per step remains (it is what `[time]` measures), and the kernels are not captured. |
| P6 (one evaluation per fluid pair) | would halve the FP64 body of P1, but the scatter breaks bit reproducibility; the deterministic alternative costs ~0.7 ms of traffic per step. |
| P5 (launch bounds) | moot for fp64 after P1 (the list kernel is insensitive), and fp32 never used the fused fp64 kernel. |
