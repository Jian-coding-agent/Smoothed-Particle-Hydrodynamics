# Review: CPU reference (`cpu/1.cpp`) and CUDA port (`cuda/src`)

| | |
|---|---|
| Date | 2026-11-18 |
| Reviewer | Claude (Opus 5.5) |
| Code | commit `355b24d` + working tree (the uncommitted `cpu/1.cpp` diff only translates comments; code is unchanged) |
| Test machine | RTX 4090 (`sm_89`, 128 SMs, driver 580.173.02), CUDA 12.2, host g++-11 (`-O3`), Ubuntu 22.04 |
| Scope | read `1.cpp` completely, review every CUDA kernel and the host driver against it, then **verify by running**: sanitizers, brute-force checks in Python, and prototype optimisations measured on the GPU |

Earlier reviews are not repeated here: the `eta` fix, `solidPack` bounds, fp32 resolution fix (E1–E7, V2) in `cuda/docs/DESIGN.md` §7, and the build-portability, CPU baseline and duplicated fluid–solid force points in `_local/CUDA_REPORT_4090.md`.

---

## 1. Summary

**Correctness.** For the shipped input deck the port is a faithful transcription of `1.cpp`. I re-derived every pair term (both sides of each branch, the `eta` rule, `PRii`, the A3 factorisation of `PIJW`, the A5 identities, the `Pia` AoS loads), and all of it checks out. All 16 `compute-sanitizer` runs are clean (memcheck / initcheck / racecheck / synccheck × fp64/fp32 × static/dynamic solid path). Runs are byte-reproducible.

There is **one real defect**, which is latent: the static solid pair list (A3) is built by looking around each particle's *reference* cell in a cell list that was binned from *current* positions. For this deck the solid has moved at most 0.17 mm, and nothing is lost. **After a rigid 2.5 cm offset, 235,296 in-support pairs vanish silently across 19,760 solids, and stresses are wrong by up to 5.3 MPa after one step** (§3, C1). Everything else is robustness or documentation.

**Performance.** Two changes measured on the 4090 make the step **2.2× faster in fp64 and 1.65× faster in fp32, with byte-identical output** (201 steps, all result files compared with `cmp`):

| | shipped | with the changes below | speed-up |
|---|---:|---:|---:|
| fp64 | 8.81 ms/step | **3.96 ms/step** (P1 + P2) | **2.2×** |
| fp32 | 2.12 ms/step | **1.28 ms/step** (P2) | **1.65×** |

* **P1.** Split fp64 `pass1_fluid` into "record the neighbours" + "run the identical body over the list". 6.19 ms becomes 0.69 + 2.00 ms. The fused kernel runs its expensive FP64 body under a sparse lane mask (SIMT divergence), and this split removes that.
* **P2.** Store the A3 solid pair lists slot-major (ELL) instead of CSR, so warp loads coalesce. Pass 1 solid goes 1.12 → 0.34 ms and pass 2 solid 1.13 → 0.53 ms (fp64); in fp32, 0.61 → 0.19 and 0.67 → 0.29 ms.

After these two changes, **writing result files becomes the largest remaining cost**: about 20–25 % of wall time in fp64 and 45–50 % in fp32 (P3).

### Findings at a glance

| id | severity | area | finding |
|---|---|---|---|
| C1 | **Medium** (latent, silent) | CUDA A3 | static solid list misses pairs once `‖x−X0‖` reaches the cm range |
| C2 | Low | CUDA A3 | 10–12 % of static-list entries are knife-edge pairs with weight ~1e-40 |
| C3 | Low | CUDA A3 | count and fill passes are not cross-checked |
| C4 | Low | CUDA host | input deck layout is never validated |
| C5 | Low | CUDA host | no launch-error check in the step loop |
| C6 | Low | build/docs | `--fmad=true` comment contradicts what the flag does |
| C7 | Low | CUDA | `k_cell_sort` is O(n²) on one thread per cell (performance cliff) |
| C8 | Info | CUDA fp32 | displacement form assumes no `A = I` fallback; verified true here, unchecked at run time |
| R1 | Low | CPU ref | reference pair set depends on FMA contraction (measured) |
| R2 | Info | CPU ref | solid `F` uses the current-configuration pair list; A3 is equivalent only below ~6 % strain |
| R3 | Info | CPU ref | output `p` lags `rho`, `x`, `v` by one update |
| R4 | Info | CPU ref | discontinuous switches explain the CPU/GPU `p` spread |
| P1–P7 | — | performance | §4 |

---

## 2. The CPU reference: observations that matter for the port

The algorithm summary in `cuda/docs/DESIGN.md` §2 is accurate and is not repeated. The points below are new, or they sharpen statements made there.

### R1 — The reference pair set depends on compiler FMA contraction (Low)

The membership test `d0*d0+d1*d1+d2*d2 < R2` ([1.cpp:530](../../cpu/1.cpp#L530)) is knife-edge for the lattice-placed wall and solid particles. I ran the existing variant binaries in `_local/cpu_variants` for one step:

| CPU build | pairs at T=0 |
|---|---:|
| `-O3`, `-O3 -march=x86-64-v3 -ffp-contract=off` | 11,102,707 |
| `-O3 -march=native`, `-O3 -march=x86-64-v3` (FMA allowed) | **11,103,787** (+1,080) |

The CUDA port matches the non-FMA count, which is what the MSVC build produces because it doesn't use `/arch:AVX2`. Anyone who regenerates reference data with an AVX2/FMA build (GCC `-march=native`, or MSVC `/arch:AVX2` with contraction) breaks the "pair set identical to `direct_find()`" claim. Physically nothing changes, because the extra pairs sit at `q = 2` where the kernel is 0.

*Recommendation:* add `-ffp-contract=off` (GCC/Clang) to the CPU build, keep MSVC off `/fp:contract`, and document this next to the validation claim.

### R2 — Solid `F` is built from the *current* neighbour list (Info; this defines A3's validity range)

`1.cpp` accumulates `F`/`DFdt` only for solid pairs that are in this step's current-distance pair list ([1.cpp:790](../../cpu/1.cpp#L790)). The kernel weight, however, is evaluated at the reference distance `Dr`. A pair with `Dr < 3h` that stretches to `r ≥ 3h` therefore drops out of the reference, but stays in the A3 static list. On this lattice the largest interior distance is `√8·0.01 = 0.0283 m`, so the two formulations agree while no such pair stretches by more than about **6 %**. The plate here moves 0.1 mm, so they agree. Arguably A3 is the more correct total-Lagrangian form.

*Recommendation:* state this condition in DESIGN §12. The current wording says the CPU "pair set and every per-pair weight are constants", which is not literally true of the reference.

### R3 — Printed `p` lags the other columns by one update (Info)

`p` is computed from `rho` at the start of the step ([1.cpp:643](../../cpu/1.cpp#L643)), while `x`, `v` and `rho` are written after the update. So in `result*.dat`, `p ≠ c0²(rho−1000)`. The port reproduces this correctly. It is worth a sentence in the README because it will confuse anyone post-processing the files.

### R4 — Discontinuous switches amplify rounding (Info; explains the validation spread)

The viscosity switch `|SurFD| < 2.5` changes `eta` by 50×. In the bulk, `SurF ≈ −3` (the continuum value of `Σ V W'(r) r`), while near the free surface it spreads across the threshold; DESIGN §7.1 counts 96,802 of 101,250 fluid particles at `|SurF| ≥ 2.5`. The solid clamp `y ≤ 0.01 || y ≥ 0.49 || z ≤ 0.01` lands exactly on lattice rows, and `rho < 960` is a clamp too. Any 1-ulp difference, whether from summation order, FMA or the CPU's `schedule(dynamic)`, can flip one of these switches. That is why `p` differs by 1e-3…1e-2 between CPU and GPU, and between two CPU runs, while `x` agrees to every printed digit. The validation tolerance is therefore sound. The fact that both pass-1 (mirror Riemann) and pass-2 (`SForce`) act on fluid–solid pairs was already raised in `_local/CUDA_REPORT_4090.md`. One addition: the pass-1 term acts on the fluid **only**, with no reaction on the solid, so momentum is not conserved for that part. The physics owner should decide whether this is intended.

### CPU acceleration (brief; the GPU is the production path)

* Replace the per-thread accumulation buffers (35 arrays × `num` × threads = **2.03 GB at 32 threads**) and the reduction passes with the GPU's gather formulation (every pair term is symmetric; DESIGN §3.1). That removes about 1 GB/step of zeroing and reduction traffic, which is roughly the 22 % reduction cost, and makes results independent of the thread count.
* Pass 1 calls `dwdx()` about 30 times per fluid–fluid pair, each with a division and the full polynomial. Hoisting `1/dr` and sharing the polynomial took the GPU from 56 to 37 ms/step (DESIGN §6.3), and the CPU would benefit in the same way.
* Use the static solid list (A3) on the CPU too, and build with `-O3`, since `-O2` → `-O3` was measured at 1.47× in `_local/CUDA_REPORT_4090.md`. Keep contraction off (R1).

---

## 3. CUDA port: correctness findings

### What was verified (no issue found)

* **Term-by-term transcription** against `1.cpp`:
  * both sides of fluid/fluid;
  * the fluid/(solid|wall) mirror branch (`PR`, `rhoR`, `UL = −v·n`);
  * `eta = f(SurFD[max(p,q)])`;
  * `F`/`DFdt` index order (`F_rc = Σ w Δx_r (A g0)_c`);
  * the constitution (`DFdtB = DFdt·A`, `SD`, `F⁻¹`, `log10`);
  * `PIJW` on both sides, and its A3 factorisation `(Pia_p A_p Σ vol g0 − Σ vol Pia_q A_q(−g0))/ρ_p`;
  * `SForce` coefficients on both sides;
  * update order and the solid clamp evaluated *before* the position update;
  * the output column semantics.
* **A5 identities**: `(2−q)⁴ − 2(1+2q)(2−q)³ = −5q(2−q)³`, `UL−U* = ½dU − Δp/(ρΣ c0)`, and the single-division trick. All correct.
* **A4 margin**: by my bound the float pre-filter's worst error is about 2.2e-8 against a 9e-8 margin, so no true neighbour can be rejected. The comment at [sph_kernels.cu:49](../../cuda/src/sph_kernels.cu#L49) says the estimate "can only be *larger*", which is not accurate (it can be smaller or larger), but the margin argument holds.
* **`Pia` AoS / `double2` loads**: the 96-byte (fp64) and 48-byte (fp32) strides keep 16- and 8-byte alignment.
* **fp32 `F − I` form** needs `a·Aᵀ = I`, which needs `a` symmetric and no `A = I` fallback. Computed from `input.dat`: `max|a − aᵀ| = 7e-17`, `min det(a) = 0.0169` (threshold 1e-6), **0 fallbacks** (see C8).
* **Static list (unshifted deck)**: I dumped `sStart`/`sQ` and compared them with a brute-force enumeration over `X0`. All **4,844,544** in-support pairs are present; the 527,049 extra entries are all knife-edge pairs.
* **Sanitizers**: 16/16 clean (listed in §1).

### C1 — Static solid list is built from a mixed reference/current cell lookup (Medium, latent)

[`k_solid_pair_count`](../../cuda/src/sph_kernels.cu#L196) and [`k_solid_pair_fill`](../../cuda/src/sph_kernels.cu#L222) walk the 27 cells around `sph_cell(X0_p)`. But `cellStart`/`sorted` were binned from the **current** positions `x` ([sph_host.cu:559](../../cuda/src/sph_host.cu#L559), before [line 565](../../cuda/src/sph_host.cu#L565)). A partner `q` is found only if `cell(x_q)` lies within ±1 of `cell(X0_p)`. With lattice offsets up to 0.02 m per axis and cell edges that coincide with lattice planes, that fails once `|x − X0|` reaches about 1–2 cm. The input deck is a restart: the solid already carries displacements up to 0.17 mm and 46,530 solids have non-zero velocity, so a restart from a more deformed state is a realistic way to hit this.

**Reproduction.** Use `input.dat` with the solid's current `y` shifted by +0.025 m and `X0` unchanged. This is a rigid translation, so stresses must not change.

| | A3 list size | in-support pairs missing | solids affected | `trace(Pia)` error after 1 step |
|---|---:|---:|---:|---:|
| unshifted deck | 5,371,593 | 0 | 0 | — |
| shifted, default (A3) | 5,068,175 | **235,296** | **19,760** | **max 5.3 MPa** (typical value 593 Pa) |
| shifted, `SPH_SOLDDYN=1` | — | — | — | 1e-5 Pa |

**Fix.** Bin the reference configuration for the one-off build, for example by building the cells from a copy of `g` whose `x` points to `X0`:

```cpp
SphPtrsT<Real> ref = g; ref.x = g.X0;          // reference configuration
build_cells_for(ref);                          // hist/scan/scatter/sort on X0
launch_solid_pair_count(ref, d_cnt, 0);  ...   // and fill
build_cells_for(g);                            // step() rebuilds from x anyway
```

Also assert at start-up that the list contains every pair with `Dr < 3h` (C3 gives the counts nearly for free). `GraCorrec` ([`k_gra_correc`](../../cuda/src/sph_kernels.cu#L1432)) is unaffected: like the CPU, it uses the current-position list consistently.

### C2 — 10–12 % of the static list is knife-edge dead weight (Low)

The list keeps any pair whose computed `wc ≠ 0`. For pairs exactly at `Dr = 3h`, `q = Dr·(1/h)` sometimes rounds to `2 − 1ulp`, giving `|g0| ~ 1e-40`. That is 527,049 of 5,371,593 entries in fp64, and 5,500,191 − 4,844,544 = 655,647 in fp32. These entries cost bandwidth in both solid kernels every step and inflate the ELL width from 92 to 115 (fp64) or 120 (fp32) slots (P2). Dropping them in fp64 left all 201-step output **byte-identical**.

Filter them **geometrically**, for example `Dr < 3h·(1−1e-9)`. A magnitude threshold does not work: `|G| > 1e-20` removed the fp64 entries but none of the fp32 ones.

### C3 — Count and fill passes are not cross-checked (Low)

`k_solid_pair_fill` writes from `start[k]` with no bound, and relies on `solid_pair_geom` deciding exactly as it did in `k_solid_pair_count`. The two kernels are compiled separately, so with `--fmad=true` a 1-ulp difference at the knife edge cannot be ruled out in principle. It would silently corrupt the neighbouring particle's CSR range. The fill kernel already returns `cur[k]`: copy it back and abort unless `cur[k] == start[k+1]` for all `k` (240 KB, once, at [sph_host.cu:248](../../cuda/src/sph_host.cu#L248)).

### C4 — The input deck is not validated (Low)

[`load_input`](../../cuda/src/sph_host.cu#L140) parses exactly `SPH_NUM` records and silently ignores any extra lines. It never checks the layout every kernel depends on:
* fluid `itype > 0` in `[0, NTOTAL)`;
* solid `< −10` in `[NTOTAL, NALL)`;
* walls after that;
* all positions inside the fixed grid.

The `eta = f(SurFD[max(p,q)])` rule is also only right if the original index order is kept. `sph_cuda.exe other.dat` makes it easy to feed a deck that breaks these assumptions, and the result would be wrong physics with no error. A 10-line check after parsing would close this.

### C5 — No launch-error check in the step loop (Low)

[`step()`](../../cuda/src/sph_host.cu#L368) launches about 11 kernels and 3 memsets, and only `cudaDeviceSynchronize()` is checked. Launch-configuration failures (for example too many resources) are reported by `cudaGetLastError()`, not by the synchronise. Such a kernel would silently not run. Add one `CUDA_CHECK(cudaGetLastError())` per step.

### C6 — `--fmad=true` is commented as "keep the fp64 maths in the same order as the reference build" (Low)

[cuda/CMakeLists.txt:76](../../cuda/CMakeLists.txt#L76): the flag does the opposite. FMA contraction is enabled everywhere except the neighbour test (`sph_add_rn`/`sph_mul_rn`), whereas the CPU reference never contracts (R1). The behaviour is fine and is what the validation was run with, so only the comment is wrong. It should read something like "FMA contraction allowed (nvcc default); the neighbour test opts out explicitly".

### C7 — `k_cell_sort` is O(n²) with one thread per cell (Low; performance cliff)

[sph_kernels.cu:302](../../cuda/src/sph_kernels.cu#L302). This is fine at about 7 particles per cell. But the clamp in `sph_cell` collects every escaped or NaN particle into a boundary cell, and DESIGN §7.9 records 60,320 particles landing in cell 0 (2152 ms/step). A stable key sort, such as `cub::DeviceRadixSort::SortPairs` on `(cell, index)` with the index already ascending, gives the same deterministic order in O(n). It also replaces the atomic scatter and removes 3 of the 4 cell-list kernels.

### C8 — fp32 displacement form has no run-time guard (Info)

The `F − I` accumulation ([sph_kernels.cu:692](../../cuda/src/sph_kernels.cu#L692)) is wrong for any particle whose `A` fell back to the identity. That is true for this deck (0 fallbacks, above), but nothing checks it for other decks. Count the fallbacks in `k_gra_invert` and warn, or refuse fp32, when the count is not zero.

### Nits

* Result files are written with LF, while the CPU on Windows writes CRLF, so CPU and GPU files never match byte-for-byte. Numeric comparison is unaffected.
* The `SPH_PROFILE` loop duplicates the body of `step()`, so the two will drift once kernels are fused (P4).
* `sph_neigh2` is used by `k_bench_loads` before its definition. This compiles through ADL, but it is fragile.

---

## 4. Acceleration

All numbers are from the RTX 4090. `bench` means `SPH_BENCH=1` (ms per kernel); "step" means the full loop with output time subtracted. The prototype code is in [`prototype/`](prototype/) (§5).

### Where the time goes (shipped code)

| kernel | fp64 | fp32 |
|---|---:|---:|
| `pass1_fluid` | **6.46** | 0.43 |
| `pass1_solid` (A3) | 1.12 | 0.61 |
| `pass2_solid` (A3) | 1.13 | 0.67 |
| cell list | 0.16 | 0.14 |
| `pass2_fluid`, constitution, updates | 0.15 | 0.14 |
| **step (measured)** | **8.81** | **2.12** |

In fp32 on this GPU, **A3 is slower than the old cell walk**: `pass1_solid` 0.61 vs 0.56 ms and `pass2_solid` 0.67 vs 0.57 ms (from `SPH_BENCH`'s pre-A3 lines). The CSR list streams 210 MB from DRAM with uncoalesced warp loads. P2 fixes this.

### P1 — Two-phase `pass1_fluid` for fp64: 6.19 → 2.69 ms, byte-identical  ★

In the fused loop each lane walks its own 27 cells, and the roughly 90-instruction FP64 body runs whenever *any* lane in the warp has a hit. About 15 % of the candidates are hits (≈104 hits out of the 687 candidates per particle that DESIGN §10 counts), so most lanes sit masked off while the FP64 pipe is busy.

That explains a contradiction in DESIGN §6.3: the kernel looks "FP64-pipe bound", yet more warps and more ILP did not help. A rough check agrees: 10.5 M neighbour visits × about 90 FP64 instructions is about 1.5 ms of FP64 issue on a 4090, against 6.2 ms measured. (The hardware counters that would confirm this directly are locked on this machine.)

**Prototype.** `k_fluid_nbr_build` records each fluid particle's neighbours, in the original visiting order, into a slot-major list `nbr[k·NTOTAL + p]`. `k_pass1_fluid_list` then runs the **verbatim** A5 body over that list. Because the order is the same, the bits are the same.

| fp64 | ms |
|---|---:|
| fused `pass1_fluid` (shipped) | 6.19 |
| list build + body over list | 0.69 + 2.00 = **2.69** |
| full step, P1 + P2 | 8.81 → **3.96** |

The body over the list runs at about the FP64 issue estimate, which is consistent with the divergence explanation. **For fp32, do not use it**: the step went 1.28 → 1.69 ms, because fp32 is traversal-bound and the body is cheap.

Implementation notes:
* Neighbour counts here are min 42, mean 103.9, max 115. 128 slots (52 MB) suffice today, but compression raises the count, so keep the overflow counter and fall back to the fused kernel, or regrow the list, when it fires.
* The same list can feed `pass2_fluid` (filter to solids) and the `pairs=` count, which saves those traversals as well.

### P2 — Slot-major (ELL) static solid lists: byte-identical, faster in both precisions  ★

Each thread walks its own CSR segment, so a warp load touches 32 segments that are about 89 × 24 B apart. Storing entry `j` of solid `k` at `j·SOLNUM + k`, with `G`, `B` and `D` as separate component arrays, makes every warp load contiguous. The per-particle order is unchanged, so the bits are identical.

| kernel | fp64 CSR → ELL | fp32 CSR → ELL |
|---|---:|---:|
| `pass1_solid` | 1.12 → **0.34** | 0.61 → **0.19** |
| `pass2_solid` | 1.13 → **0.53** | 0.67 → **0.29** |
| full step | (with P1) 3.96 | 2.12 → **1.28** |

With C2 applied, the width drops from 115 to 92 slots, padding from 29 % to 15 %, and fp64 memory from 503 to 402 MB. In production, have `k_solid_pair_fill` write slot-major directly (`width` = the maximum count from the existing count pass) and free the CSR arrays.

### P3 — Overlap the result writer with compute (next largest item)

Output costs about 55 ms per file here. After P1 and P2, at the 50-step cadence, that is **about 21 % of wall time in fp64 (55 vs 198 ms of compute) and 46 % in fp32 (55 vs 64 ms)**. In the 201-step test runs it measured 25 % and 53 %.

* `mass`, `itype`, `X0` and `rho0` never change, so fetch them once (6 of the 17 per-file copies).
* Pack the remaining columns on the device into one buffer, copy it with `cudaMemcpyAsync` into pinned memory on a side stream after an event, and format and write on a background thread with double buffering.

The GPU never needs to wait for the formatter.

### P4 — Launch overhead: drop the per-step sync, fuse, then capture a graph

The main loop calls `cudaDeviceSynchronize()` after every step, only to time it. Removing the per-step sync (`SPH_NOSYNC` in the prototype) took the step from 4.00 to 3.90 ms in fp64 and 1.30 to 1.26 ms in fp32, byte-identical. That is small on Linux. DESIGN measured about 1.1 ms of launch overhead per step under Windows/WDDM, which becomes about 25 % of the step once P1 and P2 are in, so on Windows this item matters. In order:

1. time with events instead of the per-step sync;
2. fold the 3 `cudaMemsetAsync` calls into `k_prepare`;
3. fuse `k_update_fluid` / `k_update_solid` / `k_update_pos`, and fuse `k_constitution` into the end of `pass1_solid` (it reads only its own particle's `F`, `DFdt`, `A`). Keep `pass1_fluid` launched before it, because it must see the previous step's solid `rho`;
4. capture `step()` in a CUDA graph. This needs a created stream instead of the legacy stream 0.

### P5 — Launch bounds of `k_pass1_fluid_fast` force spills (fp64)

With `__launch_bounds__(256, 4)` the fp64 kernel is capped at 64 registers and spills: 144 B stack and 220/344 B of spill stores/loads per thread (CUDA 12.2, `sm_89`). DESIGN's sweep only tried 4/5/6 blocks per SM, which are all tighter caps.

| minBlocksPerSM | registers | spill stores | fp64 bench |
|---|---:|---:|---:|
| 4 (shipped) | 64 | 220 B | 6.19 ms |
| 3 | 80 | 64 B | 6.05 ms |
| 2 | 102 | 0 | **5.72 ms** |
| none | 100 | 0 | 5.74 ms |

That is −8 % on the kernel and −1.5 % on the full step, byte-identical. It is moot if P1 is adopted, since the list kernel is insensitive to the bound. Re-sweep per architecture, including lower values.

### P6 — Larger, unmeasured options

* **Evaluate each fluid pair once.** `P*` and `dU = UL−UR` are symmetric, and `UL−U* = ½dU ∓ Δp/(ρΣ c0)` only flips sign in its second part. So one evaluation gives both halves, which roughly halves the FP64 body in P1. The catch is the scatter: atomics break byte reproducibility. A deterministic alternative is a per-pair buffer gathered through a reverse CSR, at about 0.7 ms of extra traffic per step. This is the remaining fp64 lever.
* **Verlet list with a skin** (for fp32 especially, which is traversal-bound: `count_pairs` 0.42 ms ≈ `pass1_fluid` 0.43 ms). At `v ≤ 3 m/s` and `dt = 1e-5`, a 1 mm skin lasts about 16 steps and cuts candidates from 687 to about 125 per particle. Keep the exact `r² < r2max` test every step, so the pair set is unchanged; only the summation order differs from today.
* **Particle reordering (A8), with a caveat.** Permuting the state arrays changes particle indices. That breaks `SurFD[max(p,q)]` and every type-range assumption (C4). Either keep an `orig_id` array for the `eta` rule, or reorder only the thread→particle mapping (process `sorted[]` order) and leave the state arrays alone.

### P7 — If P2 is not adopted: make A3 precision-dependent

On this GPU A3 loses in fp32 (see the profile above). Default to the cell walk for fp32 unless P2 lands.

### Projected step times (RTX 4090, measured components)

| configuration | fp64 ms/step | fp32 ms/step |
|---|---:|---:|
| shipped | 8.81 | 2.12 |
| + P2 (ELL) | — | 1.28 |
| + P1 (fp64 only) | 3.96 | — |
| + C2 knife-edge filter | 3.89 | 1.28 |
| + no per-step sync | ~3.8 | 1.26 |
| P3 output overlap | wall time −25 % | wall time −50 % |

---

## 5. Reproduction

Prototype files are in [`prototype/`](prototype/). They are a scratch patch for measurement, not production code.

```bash
# scratch copy of the CUDA sources + experiments (the repo is not modified)
mkdir -p /tmp/proto && cp -r cuda/src /tmp/proto/ && cp review/Claude/prototype/review_ext_kernels.inc /tmp/proto/
python3 review/Claude/prototype/patch.py /tmp/proto/src
nvcc -ccbin g++-11 -O3 -std=c++17 --fmad=true -arch=sm_89 -DP1_MINB=4 \
     -I/tmp/proto/src /tmp/proto/src/sph_kernels.cu /tmp/proto/src/sph_host.cu -o sph_proto

# run from a directory containing input.dat
SPH_BENCH=1 SPH_FLUIDLIST=1 SPH_ELL=1 ./sph_proto --fp64     # per-kernel timings
SPH_MAXSTEP=201 SPH_FLUIDLIST=1 SPH_ELL=1 ./sph_proto --fp64 # compare result0..4 with cmp
```

| switch | effect |
|---|---|
| `SPH_FLUIDLIST=1` | P1 (two-phase `pass1_fluid`) |
| `SPH_ELL=1` | P2 (ELL solid lists) |
| `SPH_ELL_DROPEDGE=1` | C2 (magnitude filter; fp64 only, see C2) |
| `SPH_NOSYNC=1` | P4 (no per-step sync) |
| `SPH_DUMPSOLIDLIST=1` | writes `solidlist_*.bin` for `check_list.py` |
| `-DP1_MINB=n` | P5 (launch bounds) |

The other two scripts:
* `check_solid.py input.dat`: brute-force reference pair counts and GraCorrec determinants (C8, C2).
* `check_list.py`: compares dumped static lists with brute force and measures the stress error of the shifted-deck test (C1). The shifted deck is `input.dat` with column 2 (+0.025) on lines 101251–161570.
