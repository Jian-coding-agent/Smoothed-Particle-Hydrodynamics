# Review: multi-GPU decomposition (D1), lean solid list (A6), runtime sizing and binary I/O

| | |
|---|---|
| Date | 2026-11-19 |
| Reviewer | Claude (Opus 5.5) |
| Code | `a106b7b` (runtime sizing, `.sphbin`) and `4807fcc` (A6, D1); HEAD = `4807fcc`, clean tree |
| Read | `cuda/src/*` in full, `docs/edit-2026-11-20-01-38.md`, `cuda/docs/BINARY_IO.md`, both READMEs, `cuda/tools/sphbin.py`, `cuda/tools/cmp_bin_text.py` |
| Test machine | one RTX 4090 (`sm_89`, 24 GB), driver 580.173.02, CUDA 12.2, g++-11, Ubuntu 22.04 |
| Deck | `input.dat` (237,298 particles: 101,250 fluid, 60,320 solid, 75,728 wall). The 1203k deck is not in the repository. |

**How D1 was tested on one GPU.** `SPH_RANKS_GPUS` accepts the same device more than once, so `SPH_RANKS=8 SPH_RANKS_GPUS=0,0,0,0,0,0,0,0` runs the complete decomposed path (partition, ghosts, every exchange point, repartition) with eight rank threads on one card. That tests correctness, not 8-GPU speed. Every timing for the 8 × 5090 D machine below is quoted from `docs/edit-2026-11-20-01-38.md` and marked as such. To see *why* runs diverge, I instrumented a scratch copy with a host-side halo check (§8): every 250 steps it verifies that each rank holds every particle within 3h of each particle it owns.

Finding IDs continue the numbering of [REVIEW.md](REVIEW.md) (C1–C8, P1–P7): correctness C9–C15, load balancing L1–L3, performance P8–P14, verification V1–V2, tools and documentation M1–M8.

---

## 1. Summary

**What holds up.** The structure of D1 is sound, and the parts I could check by running reproduce the author's results:

- `SPH_RANKS=1` is byte-identical to the single-GPU path.
- K = 2, 4 and 8 agree with one GPU after 501 steps to ≤ 3.3e-16 m, fluid `p` ≤ 5.2e-10 Pa, solid `trace(Pia)` ≤ 2.3e-7 Pa. The doc reports the same magnitudes on the 1203k deck.
- The lean and fat solid lists (A6) are byte-identical on a second architecture and compiler (sm_89, CUDA 12.2; fp64 and fp32; 1001 steps).
- Every ghost field a kernel reads is refreshed before it is read (§2.0). The `eta` rule via `gid` is right. The relay/barrier protocol has no race.
- The partitioner's boxes contain the owned particles, so the ghost set is complete at the moment it is built.

**What does not.** The decomposition is correct only while particles stay near where they were when the ghosts were chosen, and nothing checks that.

- **C9 (Critical).** The ghost set is chosen once and never revalidated. With the default `SPH_D1_REPART=0`, the default K=8 split starts dropping fluid interactions at step ~1750 on `input.dat`. By step 6000 its result is 4.4 mm and 19 kPa away from the single-GPU run; a run whose halo stayed complete is 5e-14 m away.
- **C10 (High).** Repartition, the obvious remedy, picks the new ghosts from the **t = 0** deck positions. On a split that was correct without repartition, `SPH_D1_REPART=1000` makes it wrong (5.0 mm / 643 Pa at step 6000). The doc's 3001-step `REPART=500` deviation (1.9e-10 m) is probably this bug already showing.

Both are fixed in the scratch build. With current positions at repartition, a 1.5h skin and a rebuild every 500 steps, K=8 has zero missing pairs over 6001 steps and stays at rounding level against one GPU (§2.1, §2.2).

**Performance.** At K=8 the step is bound by exchange points, and one of the three (E3) ships a field that is identically zero (fluid `PIJW`). Removing it is byte-identical over 6001 steps (P8). Dropping dead fields from E1 roughly halves the bytes (P9), double-buffering the send block halves the barriers (P10), and a two-layer solid halo removes E2 (P11). From the doc's own K=8 phase times, P8 + P10 + P11 should bring 3.4–4.7 ms/step down to roughly 2.7–3.3 ms; this needs measuring on KMG-XI.

**"Cards by computation load."** The code balances the work across the K cards it is given. It does not choose K or which cards. The fixed weights (fluid 1.0, solid 0.5, wall 0) fit the fat solid list (measured solid cost 0.41 of a fluid particle) but not the lean list that auto mode uses at 10^7 particles (0.97). With the lean costs, the K=8 split of this deck is about 31% imbalanced. §3 proposes `SPH_RANKS=auto` (rank the devices, then pick K from a cost model) and calibrated weights.

### Findings at a glance

| id | severity | area | finding |
|---|---|---|---|
| C9 | **Critical** | D1 halo | the ghost set is never revalidated; default runs silently drop interactions (measured from step ~1750) |
| C10 | **High** | D1 repartition | new ghosts and per-rank grids are chosen from t = 0 positions |
| C11 | Medium | D1 repartition | 20 device index arrays leak per repartition (measured ≈ 0.65 MiB/event here) |
| C12 | Low | A6 × D1 | the auto lean/fat decision is a data race between rank threads (3 of 6 identical runs lean, 3 fat) |
| C13 | Low | D1 | wall ghosts are never refreshed; decks with moving walls diverge |
| C14 | Low | D1 × A3 | solid ghosts are chosen in the current configuration, A3 partners in the reference one |
| C15 | Info | D1 | `sph_matvec` strides by `num`; memsets sized by `num`; unused `gs0`/`ngs`; profiling label |
| L1 | Medium | partition | K and the device list are manual; the default list always includes the slow display GPU 0 |
| L2 | Medium | partition | solid weight 0.5 fits the fat list; the lean list costs twice that |
| L3 | Low | partition | the objective counts ghosts at the owned rate |
| P8 | ★ | exchange | E3 carries an identically-zero field: drop it (verified byte-identical) |
| P9 | | exchange | E1 records: fluid 12 → 8 doubles, solid 24 → 7 (−46% bytes with P8) |
| P10 | | exchange | double-buffer the send block: one barrier per exchange point instead of two |
| P11 | | exchange | a two-layer solid halo removes E2 |
| P12–P14 | | misc | overlap; compact solid slots (−1.47 GiB on the 19.3 M single-GPU run); asynchronous output |
| V1–V2 | | testing | test long and check the halo directly; make K > 1 bit-identical to K = 1 |
| M1–M8 | Low / Info | tools, docs | `sphbin.py` rejects valid decks; header `hsml`/`dt` ignored; new-case `1.cpp` not committed; stale comments and READMEs |

### Suggested order of work

1. C10 (five lines), C11 (free the arrays) and C12: small and mechanical.
2. C9: skin plus displacement trigger, reusing the repartition with the C10 fix.
3. P8: delete E3. No risk; verified byte-identical.
4. V1: re-run the D1 validation for several thousand steps, with the halo check, repartition on and off.
5. L1/L2: `SPH_RANKS=auto` and per-mode weights.
6. P9–P11, then P12–P14.

---

## 2. Correctness of the decomposition

### 2.0 Ghost data flow (verified, no issue)

| ghost field | refreshed by | read by (on behalf of an owned particle) |
|---|---|---|
| `x`, `vx` | E1, every step | cell list, `pass1_fluid`, `pass1_solid`, `pass2_*` |
| `rho`, `irho` | E1 (`irho` rebuilt on arrival), E2 for solids | `pass1_fluid` (`m/rho`), `pass2_fluid` (solid `rho`), `pass2_solid` (`rho0/rho`) |
| `p` | E1 | `pass1_fluid` and `pass2_solid` (fluid neighbours) |
| `SurFD` | E1 | `pass1_fluid` (`eta` via `sph_pair_hi`) |
| `Pia` | E2 | `pass2_solid` |
| `A`, `vecn` | one-off static exchange | `solidPack`, the mirror terms of `pass1_fluid` / `pass2_fluid` |
| `PIJW` | E1 and E3 (fluids) | `pass2_solid`; always 0, see P8 |
| `mass`, `rho0`, `X0`, `itype` | deck rows | everywhere |

The order of operations matches `step()`, so `pass1_fluid` sees the previous step's solid `rho` exactly as on one GPU.

### 2.1 C9 — The halo is never revalidated (Critical)

[`build_ranks`](../../cuda/src/sph_host.cu#L1271-L1283) selects each rank's ghosts once: every particle inside the rank's box grown by 3h. The design note calls this "complete by construction", which is true **at that instant**. After that:

- the ghost lists stay fixed until a repartition, and by default there never is one ([`repart = 0`](../../cuda/src/sph_host.cu#L925));
- there is no skin and no check. A particle that drifts into a rank's 3h shell is never sent there, and an owned particle that drifts past its box loses its neighbours on the far side.

A lattice deck hides this for a while. The first excluded lattice plane sits at 4h, so the effective skin is about one particle spacing, which is why 501-step comparisons pass.

Measured on `input.dat`, where the fluid moves up to 12 mm by step 1000 and 54 mm by step 6000 (all ranks on one GPU, halo checked every 250 steps):

| run | first missing fluid/wall pair | missing pairs at step 6000 | closest missing pair | vs one GPU at step 6000 (max \|Δx\| / fluid max \|Δp\|) |
|---|---|---:|---|---|
| K=4, 2×2×1 (auto) | none¹ | 287 | 0.999·3h | 5.1e-14 m / 2.7e-7 Pa |
| K=8, 2×2×2 (auto) | step 1750 | 2,399 | 0.45·3h | **4.4e-3 m / 1.9e4 Pa** |
| K=4, 1×1×4 (forced) | step 1750 | 8,365 | 0.11·3h | **2.1e-2 m / 2.9e4 Pa** |

¹ Only current-configuration solid–solid pairs at r ≈ 3h, which the default A3 path does not use: it takes solid partners from the reference-configuration list, and that list was complete in every run.

The K=8 run follows one GPU at rounding level until step 1500 (5.6e-16 m). It departs exactly when fluid pairs start to go missing: 1.7e-9 m at step 2000, 1.9e-6 m at 3000, 3.2e-5 m at 4000, 2.8e-4 m at 5000 and 4.4e-3 m at 6000. A pair at 0.45·3h carries 26% of the peak kernel value and 71% of the peak kernel gradient. At 0.11·3h, two particles pass through each other without interacting.

**Fix (verified in the scratch build).**

1. Widen the ghost shell to 3h + s (a skin).
2. Keep each particle's position at the last ghost build. Every step, reduce max |x − x_build| over the owned particles (one small kernel per rank; the host max over ranks can ride on the E1 barrier), and rebuild when it exceeds s/2. This is the Verlet-list criterion, and it makes the halo provably complete. A particle that was not selected was more than 3h + s from the box on some axis, and every owned particle was inside the box. If both have moved at most s/2, they are still more than 3h apart.
3. "Rebuild" is the existing repartition, with the C10 fix.

With s = 1.5h and a fixed repartition every 500 steps (no trigger; displacements here stay below s/2 per interval), K=8 had **zero** missing pairs at all 25 checks and stayed at rounding level to step 6000 (3.9e-14 m, 1.6e-7 Pa). The price is more ghosts: 68,411 → 102,602 (+50%). The trigger replaces the guessed interval with a guaranteed one.

Even before the skin, a cheap guard is worth having: abort, or force a rebuild, when an owned particle leaves its box by more than the slack.

### 2.2 C10 — Repartition selects ghosts from the t = 0 positions (High)

[`repartition`](../../cuda/src/sph_host.cu#L1867) computes the new boxes from the current positions (`J.pos`) but calls `build_ranks(J, J.deckRows, …)`. `J.deckRows` is the deck as read at start-up ([line 2124](../../cuda/src/sph_host.cu#L2124)), and `build_ranks` tests `rows[g*14+0..2]` against the new boxes, so the test uses **t = 0 coordinates**. The same rows go to `finish_deck`, which derives each rank's neighbour grid from them ([lines 307–322](../../cuda/src/sph_host.cu#L307-L322)). A rank's particles can therefore lie outside its own grid and pile into clamped boundary cells, which is the C7 performance cliff.

Measured with K=4 (2×2×1, the split that stayed correct without any repartition, see C9) and `SPH_D1_REPART=1000`:

| repartition at step | 1000 | 2000 | 3000 | 4000 | 5000 |
|---|---:|---:|---:|---:|---:|
| missing pairs right after it | 16 | 58 | 3,709 | 5,191 | 6,327 |
| the same with current positions (the fix) | 0 | 0 | 0 | 0 | 0 |

Against one GPU the run is 2.6e-10 m / 1.2e-3 Pa off at step 3000, 8.5e-4 m / 274 Pa at 4000 and **5.0e-3 m / 643 Pa at 6000**. With the fix it is 4.9e-14 m / 1.5e-7 Pa at step 6000. Turning repartition on made a correct run wrong.

The doc's 3001-step result with `REPART=500` on the 1203k deck (1.9e-10 m, 2.2e-5 Pa) is about five orders of magnitude above the rounding-level drift of a complete halo at the same step count here (1.2e-15 m at step 3000). It is the same size as what this bug produced here at step 3000 (2.6e-10 m). The decks differ, so this is not proof, but that run is worth repeating with the halo check.

**Fix.** Build the rows from the current state: copy `J.deckRows`, overwrite `x` (and preferably `vx` and `rho`) from `J.pos` / `J.mig`, and pass the copy to `build_ranks`. The verified version is five lines (`SPH_FIX_REPART_POS` in [d1_check/patch_inst.py](d1_check/patch_inst.py)).

### 2.3 C11 — Exchange index arrays leak at every repartition (Medium)

[`build_send_lists`](../../cuda/src/sph_host.cu#L1909-L1987) `cudaMalloc`s 20 index/offset arrays per rank (`dSendIdx`, `dSendOff`, `dRecvIdx`, `dRecvOff` × 5 groups). [`xch_free`](../../cuda/src/sph_host.cu#L1746-L1751) frees only `hSend`, `hRecv` and `dStage`, and the next `build_send_lists` overwrites the pointers.

Measured with K=2 and a repartition every 10 steps (198 events): device memory went from 1,108 to 1,240 MiB, rising steadily, about 0.65 MiB per event. The leak is 16 B × (2·fluid + 3·solid + wall ghosts) per event plus allocator rounding. For the 1203k K=8 split (170,244 ghost slots) that is about 5 MB per event, and at the default `SPH_MAXSTEP` with `SPH_D1_REPART=500` about 13 GB per rank. The 10^7-particle decks leak proportionally more.

**Fix.** Free the 20 arrays in `xch_free` and zero `nSend`/`nRecv`, or allocate them once with a capacity.

### 2.4 C12 — The auto lean/fat decision races between ranks (Low)

`sph_a3_lean` is a plain global, set to −1 in `main`. [`build_solid_lists`](../../cuda/src/sph_host.cu#L446-L454) decides it in whichever rank thread gets there first, from *that rank's* list size. Measured with K=8 and `SPH_A3_LIMIT_MB=100` (the per-rank fat lists here are roughly 60 and 140 MB): six identical runs chose lean three times and fat three times.

The output is unaffected, because the layouts are bit-identical, but memory use and speed change from run to run. Worse, two ranks that both read −1 can store different values after one of them has already allocated for the other layout. The fat kernels then dereference null `sG`/`sB`/`sD`.

**Fix.** Make the flag a member of `SphPtrsT` / `Solver`, or decide once: barrier, take the largest per-rank fat size, decide, barrier.

### 2.5 C13 — Wall ghosts are never refreshed (Low)

`k_update_pos` moves every owned particle, walls included ([sph_kernels.cu:1508](../../cuda/src/sph_kernels.cu#L1508)), but E1 has only fluid and solid groups ([sph_host.cu:1337](../../cuda/src/sph_host.cu#L1337)). Walls get only the one-off static exchange. A deck with a moving wall (a gate or a piston) runs correctly on one GPU and wrongly on K > 1. Every wall velocity in `input.dat` is zero, so this is latent.

**Fix.** Refuse D1 when any wall has a non-zero velocity, or add a wall group (`x`, `vx`) to E1.

### 2.6 C14 — Solid ghosts and A3 partners use different configurations (Low)

Ghosts are selected by current position, but the A3 list is built from `X0` among the particles a rank holds. Under large strain (R2 in REVIEW.md: a pair stretched by more than about 6%), a reference partner can sit outside the current shell. A decomposed run then silently lacks a pair that one GPU has. The halo check never found such a case here (strain is small), and a skin covers it only up to s.

**Fix.** Select solid ghosts by current *or* reference position. The reference shell is static, so this adds few particles.

### 2.7 C15 — Smaller items (Info)

- [`sph_matvec`](../../cuda/src/sph.h#L320) strides by `s.d.num`, while with ghosts the arrays stride by `numP`. It is unreachable today (every solid neighbour is inside the pack range) but wrong if reused.
- [`init_reference_state`](../../cuda/src/sph_host.cu#L501-L505) memsets `3·num` and `9·num` elements of `numP`-strided arrays. This is harmless only because `dev_alloc` already zeroes everything.
- `SphDims::gs0` / `ngs` are computed but never read (P13 would use them).
- The "gather" column of `SPH_D1PROF` times only the kernel launch ([line 1451](../../cuda/src/sph_host.cu#L1451)); the gather itself is counted under "copyD2H+sync".

---

## 3. Load-based decomposition

### 3.1 What it does

Each particle gets a weight: fluid 1.0, solid 0.5, wall 0. Each axis is cut at equal cumulative weight, nested as x, then y within each x slab, then z within each column, and a cut never splits a group of equal coordinate. Every ordered factorisation of K is evaluated on the real coordinates, and the one with the smallest max_r(own + ghost weight) wins.

This is a good, cheap scheme. The boxes contain the owned particles, so the ghost set is right when built; empty ranks are rejected; and the search covers every factorisation. It picks 2×2×2 at K=8 on both decks.

### 3.2 L1 — Neither K nor the cards are chosen by load (Medium)

`SPH_RANKS` sets K and `SPH_RANKS_GPUS` sets the devices, defaulting to `0..K−1` ([line 2532](../../cuda/src/sph_host.cu#L2532)). On KMG-XI, the doc notes that GPU 0 sits on PCIe Gen2 and drives the desktop (GPU 4 also drives it). Every default run includes GPU 0, and because every exchange point ends in a barrier, the slowest card sets the pace. The doc attributes the K=8 spread (3.4–4.7 ms) to exactly this.

On the 1203k deck (doc §4), parallel efficiency is 72% at K=2, 44% at K=4 and 30–40% at K=8. For a single case, K=8 still gives the shortest time to solution. For a batch of cases, eight K=1 runs deliver 8 / 11.3 = 0.71 steps/ms in total, against 0.21–0.29 for one K=8 run: 2.4–3.3× more simulated time per hour.

**Recommendation: `SPH_RANKS=auto`.**

1. Rank the devices. Skip or demote display GPUs (`cudaDevAttrKernelExecTimeout`), and prefer the fastest PCIe link (NVML `nvmlDeviceGetCurrPcieLinkGeneration` / `Width`) and the most free memory.
2. For K = 1 … K_max, run `make_part`, which already returns per-rank fluid, solid, total and ghost counts, and predict

   `T(K) = max_r [c_f·F_r + c_s·S_r + c_p·(N_r + G_r)] + n_x·L + bytes / BW`

   with c_f, c_s, c_p from a calibration (the table in L2), n_x the number of exchange points (3 today, 1 after P8 + P11), and L the per-point floor (0.4–0.7 ms on KMG-XI, per the doc).
3. Choose argmin T for a single case, or the smallest K within about 15% of it when other cases are queued, subject to per-rank memory.

This costs seconds of host time at start-up and turns "how many cards" into a measured decision.

### 3.3 L2 — The weights are not calibrated, and they depend on the A3 mode (Medium)

Measured with single-GPU `SPH_BENCH`, fp64, RTX 4090, `input.dat`. Each entry is the per-particle cost of the kernels that particle type runs, relative to a fluid particle:

| | fat list | lean list | weight used |
|---|---:|---:|---:|
| fluid: `pass1_fluid` + `pass2_fluid` + `update_fluid` | 29.9 ns (1.00) | 27.6 ns (1.00) | 1.0 |
| solid: `pass1_solid` + constitution + `pass2_solid` + `update_solid` | 12.4 ns (**0.41**) | 26.9 ns (**0.97**) | 0.5 |
| every particle, walls and ghosts included: cell list + `prepare` + `update_pos` | 0.64 ns (0.02) | 0.59 ns (0.02) | 0 |

The lean list doubles the cost of a solid particle, and auto mode selects it once the fat list would exceed 4 GB, i.e. on the 10^7-particle decks that D1 exists for. By its own weights, the K=8 split of `input.dat` is balanced to within 0.02%. With the measured lean costs, the heaviest rank (6,128 fluid + 20,590 solid) carries about 1.3× the mean, so the step is about 30% longer than a balanced split would give. With the fat list the imbalance is 1.04.

**Recommendation.** Use per-mode weights, and re-measure them on the 5090 D, whose 1/64-rate fp64 may shift them. Better still, close the loop: time each rank's compute phases with events for the first few hundred steps, then re-cut with the measured per-rank cost. The repartition machinery is already there once C9 and C10 are fixed.

### 3.4 L3 — Ghosts are counted at the owned rate (Low)

The objective adds ghost weight at 1.0 / 0.5, but a ghost runs no physics kernel. Its own per-step work is the cell list and the exchange, about 0.02 of a fluid particle. The ghost term therefore acts as an exchange penalty with an arbitrary scale. Use c_p for ghosts in the compute term and model the exchange separately (bytes, peers, exchange points), so that a choice such as 2×2×2 versus 4×2×1 reflects the real costs.

---

## 4. Performance

All D1 timings for the 5090 D machine come from the doc. At K=8 on the 1203k deck the step takes 3.4–4.7 ms, made up of E1 0.58–1.11 ms, E2 0.22–0.58 ms, E3 0.35 ms and barriers 0.1–1.5 ms. Each exchange point has a floor of 0.4–0.7 ms that barely depends on the payload. The levers, in order, are therefore fewer exchange points, fewer barriers and fewer bytes.

### P8 — Drop E3 ★ (verified)

E3 ships fluid `PIJW` after `pass2_fluid`. No kernel ever writes `PIJW` for a fluid particle, and the project's own [DESIGN.md §2.5](../../cuda/docs/DESIGN.md#L115) (row 3) records that it is "never written for fluid particles -> always 0". E3 therefore copies zeros every step.

Measured with K=4 over 6001 steps, with and without E3: **all 13 result files are byte-identical**. On KMG-XI, per the doc, this saves 0.35 ms of 3.4–4.7 ms (7–10%) plus two barriers. Drop `PIJW` from the E1 record as well (P9).

### P9 — Trim the E1 records

| field | fluid record | solid record | why it can go |
|---|---|---|---|
| `x`, `vx` | keep | keep | |
| `rho` | keep | keep | a solid's `rho` changes only in the constitution, and E2 carries it; but after a rebuild the ghost rows are stale until the first E2 |
| `p` | drop | drop | fluid: `c0²(rho − 1000)` clamped at 0 is recomputable bit-exactly on arrival, as `irho` already is; a ghost solid's `p` is never read |
| `SurFD` | keep | drop | history for fluids; always 0 for solids |
| `PIJW` (3) | drop | drop | always 0 for fluids; a ghost solid's `PIJW` is never read |
| `Pia` (12) | — | drop | only `pass2_solid` reads `Pia`, and E2 has overwritten it by then |
| **doubles per record** | **12 → 8** | **24 → 7** | |

For the K=8 split of `input.dat` (44,717 fluid and 9,438 solid ghosts), P8 + P9 cut the per-step exchange from 8.2 MB to 4.4 MB (−46%). At 1203k the per-point floor dominates and the gain is small. At 10^7 particles with a real halo it matters: the doc's forced-2×2×2 9.6 M run reached 4.35×, against 7.08× for the split with no halo.

### P10 — One barrier per exchange point

Each exchange ends with a second barrier ([line 1530](../../cuda/src/sph_host.cu#L1530)) whose only job is to stop a rank from overwriting its `hSend` while peers still relay from it. Alternate two `hSend` blocks between consecutive exchange points. A rank can write block b again only after passing the *next* exchange's first barrier, and no rank reaches that barrier before finishing its relay. That takes 6 barriers per step down to 3, or 2 with P8. The doc measures barriers at 0.1–1.5 ms per step at K=8, and fewer sync points also let ranks absorb each other's jitter.

### P11 — Remove E2 with a two-layer solid halo

E2 exists because pass 2 reads the post-constitution `rho`, `irho` and `Pia` of ghost solids. Those ghost solids (the first layer) could compute their own constitution if the rank also held their A3 partners (a second layer, within 3h of the first in the reference configuration) and built A3 lists for them. `pass1_solid` and the constitution would then run over the owned solids plus the first-layer ghost solids, and E2 would disappear; E1 already delivers `x` and `vx` for every ghost.

The cost is more solid ghosts and some redundant solid work in the halo. With P8 this leaves **one** exchange point per step. Combining P8, P10 and P11 on the doc's phase times gives an estimate of roughly 2.7–3.3 ms per step at K=8 on the 1203k deck, against 3.4–4.7 ms today; this needs measuring.

### P12 — Overlap (later)

Everything runs on the legacy default stream, so each exchange is gather → D2H → wait → relay → H2D → scatter → wait, with the GPU idle meanwhile. The standard next step is to split the owned particles into interior and boundary sets and run the interior work on a second stream during the exchange. It pays off once P8–P11 have reduced the number of exchange points.

### P13 — Compact solid slots

Since D1, `Pia` (12 Reals) and `solidPack` (14 Reals) have a slot for every index ≥ `ntotal`, including every wall and every ghost fluid or wall. That is 92 MiB on the 1203k deck and 1.47 GiB on the 19.3 M single-GPU run, about 7% of its 22.1 GB. Map ghost solids to slots `solnum + (q − gs0)` (`gs0` and `ngs` are already computed) and size both arrays `solnum + ngs`.

### P14 — Asynchronous global output

Every 500 steps, all ranks wait while rank 0 writes the whole result file, 616 MB at 19.3 M particles. Hand the staging arrays to a writer thread (double-buffered) and let the ranks continue.

---

## 5. Verification

### V1 — Test long, and check the halo directly

C9 and C10 are invisible at 501 steps on these decks and appear between steps 1750 and 3000. Validate D1 over several thousand steps, with repartition on and off, and keep a halo-completeness check. The scratch version is about 120 lines of host code ([d1_check/patch_inst.py](d1_check/patch_inst.py), `SPH_HALOCHK=N`). A GPU version could run every few hundred steps in production as a guard.

### V2 — Make K > 1 bit-identical to K = 1 (not tested)

K > 1 differs from one GPU at the 1e-16 level only because the summation order differs. Each rank derives its own grid origin, and [`k_cell_sort`](../../cuda/src/sph_kernels.cu#L354) orders a cell by local index, which puts every ghost after every owned particle. If every rank used the global grid (the global origin plus integer offsets) and the cell sort ordered by `gid`, then the 27-cell visiting order and the within-cell order would equal the single-GPU order. So would the P1 fluid lists, the A3 lists, GraCorrec and NorVector, and the exchanged values are already bit-exact.

K > 1 would then be byte-identical to K = 1. A halo or exchange bug would show up as a non-zero difference instead of hiding under the 1e-16 noise, which is the standard the project already applies to every other change.

---

## 6. Runtime sizing, binary I/O and tools (`a106b7b`)

The runtime-sized grid, the chunked `k_cell_scan` (I re-derived the carry logic) and the deck validation in `finish_deck` are correct as far as I could check. The single-GPU path still runs this deck at 3.9 ms/step.

- **M1 — `sphbin.py make` rejects valid decks.** The order check ([sphbin.py:66–67](../../cuda/tools/sphbin.py#L66-L67)) requires exactly three runs of equal `itype`. A deck whose fluid block uses two `itype` values (tested: it fails with "deck is not ordered as fluid|solid|wall"), or a deck with no solid, is refused even though the solver accepts it. Classify rows as 0/1/2 and check that the class sequence never decreases. The docstring also says results have 5 fields (`x,y,z,p,itype`) when they have 4, and it omits flag bits 1–2. The variables `order` and `types` are unused.
- **M2 — Header `hsml`/`dt` are ignored.** The converter writes them, but the solver uses the compile-time `SPH_HSML` / `SPH_DT`, so a deck made with `--hsml 0.005` runs silently with 0.01. Check the header values and refuse a mismatch.
- **M3 — D1 trusts the header.** `d1::read_deck` takes `ntotal` / `solnum` from the header for the global result header and never checks the global block order. The single-GPU loader does check it, while D1 checks only each rank's owned rows. Reuse the single-GPU validation.
- **M4** — The comment on `k_cell_scan` says a 1024-thread block, but it launches 512. At 10^7 particles (3 M cells) a single-block scan may start to show; `cub::DeviceScan` is the drop-in replacement.

---

## 7. Documentation and repository

- **M5 — The reference for the new case is not in the repository.** The CUDA port now follows a "new-case `1.cpp`" (no solid clamp, `z > 1.0` reflection, output every 500 steps), but `cpu/1.cpp` is still the old case (clamp at [1.cpp:193](../../cpu/1.cpp#L193), text output every 50 steps). The top-level README still says both versions read `input.dat` and write `result<N>.dat` every 50 steps. The correctness baseline should be committed, or at least its diff.
- **M6** — Neither README mentions D1. `cuda/README.md` documents the A6 switches but not `SPH_RANKS`, `SPH_RANKS_GPUS`, `SPH_D1_REPART`, `SPH_D1_REGRID`, `SPH_D1_SPLIT`, `SPH_D1PROF` or `SPH_BARRIER`. Until C9 and C10 are fixed, it should also warn that long D1 runs are affected.
- **M7 — Stale comments.** [sph.h:388–391](../../cuda/src/sph.h#L388-L391) and [sph_kernels.cu:2049–2057](../../cuda/src/sph_kernels.cu#L2049-L2057) describe a mapped zero-copy arena, but the code stages through `dStage` and pinned blocks. The driver header at [sph_host.cu:758–783](../../cuda/src/sph_host.cu#L758-L783) still describes x-slabs with "at most two peers" and one copy per link. The design note's "complete by construction" needs "at the moment of partitioning".
- **M8 — A question for the physics owner.** [`k_update_fluid`](../../cuda/src/sph_kernels.cu#L1482) flips `v_z` whenever `z > 1.0`, including for a particle already moving down, which can then oscillate above the plane. The usual form reflects only outward motion (`v_z > 0`). In `input.dat` the fluid stays below 0.45 m, so I could not observe this; please confirm against the new-case `1.cpp`.

---

## 8. Reproduction

The scripts are in [`d1_check/`](d1_check/). They work on a scratch copy; the repository is not modified.

Linux, from the repository root (the binary and the run directories go to a scratch directory):

```bash
S=/path/to/scratch            # any directory outside the repository
DECK=$PWD/input.dat
cp cuda/src/* $S/ && python3 review/Claude/d1_check/patch_inst.py $S
nvcc -ccbin g++-11 -O3 -std=c++17 --fmad=true -arch=sm_89 -I$S \
     $S/sph_kernels.cu $S/sph_host.cu -o $S/sph_inst

# single-GPU reference, then 8 ranks on the same GPU with the halo check every 250 steps
mkdir -p $S/ref $S/k8
(cd $S/ref && SPH_MAXSTEP=6001 $S/sph_inst $DECK)
(cd $S/k8 && SPH_MAXSTEP=6001 SPH_RANKS=8 SPH_RANKS_GPUS=0,0,0,0,0,0,0,0 \
    SPH_HALOCHK=250 $S/sph_inst $DECK | grep halochk)
python3 review/Claude/d1_check/cmp.py $S/ref $S/k8
```

On Windows, copy the whole `cuda\` folder to a scratch folder, run `python review\Claude\d1_check\patch_inst.py <scratch>\src`, and build the copy with its `build.bat`. On KMG-XI the same checks run on real separate GPUs; drop `SPH_RANKS_GPUS` to use the default devices.

| switch (scratch build only) | effect |
|---|---|
| `SPH_HALOCHK=N` | every N steps and after each repartition, report missing neighbour pairs and missing A3 reference partners per rank |
| `SPH_NO_E3=1` | skip exchange point E3 (P8) |
| `SPH_FIX_REPART_POS=1` | repartition builds ghosts and grids from the current positions (the C10 fix) |
| `SPH_D1_SKIN=s` | ghost shell of 3h + s metres (C9) |

With every switch unset, the instrumented build is byte-identical to the original (checked with K=2 over 501 steps).

| experiment (6001 steps, `input.dat`) | settings | used in |
|---|---|---|
| e1 | K=4, auto split (2×2×1) | C9, P8 |
| e2 | K=8, auto split (2×2×2) | C9 |
| e3 | K=4, `SPH_D1_SPLIT=1,1,4` | C9 |
| e4 | K=4, `SPH_D1_REPART=1000` | C10 |
| e5 | e4 + `SPH_FIX_REPART_POS=1` | C10 |
| e6 | e1 + `SPH_NO_E3=1` | P8 |
| e7 | K=8, `SPH_D1_REPART=500 SPH_FIX_REPART_POS=1 SPH_D1_SKIN=0.015` | C9 |

Other runs: C11 used K=2 with `SPH_D1_REPART=10` over 2001 steps, sampling `nvidia-smi`. C12 used K=8 with `SPH_A3_LIMIT_MB=100` over 3 steps, six times. L2 used `SPH_BENCH=1` with `SPH_A3_FULL=1` and `=0`.
