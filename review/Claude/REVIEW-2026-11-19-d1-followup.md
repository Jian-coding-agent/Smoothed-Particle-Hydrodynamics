# Follow-up review: D1 after the fixes (C9–C15, P8–P14, L1–L3, V2, M8)

| | |
|---|---|
| Date | 2026-11-19 |
| Reviewer | Claude (Opus 5.5) |
| Code | `3f171f1` and `65216fa` (code), `c146aa7` (work log); HEAD = `c146aa7` |
| Read | every changed line of `cuda/src/*`, `docs/edit-2026-11-20-11-53.md`, `cuda/docs/DESIGN.md` §14, the README changes, `cuda/tools/sphbin.py` |
| Test machine | one RTX 4090 (`sm_89`), CUDA 12.2, g++-11, Ubuntu 22.04. Decomposed runs put every rank on that one GPU (`SPH_RANKS_GPUS=0,0,…`): this checks correctness and host-side costs, not 8-GPU step times. |
| Decks | `input.dat` (237,298 particles), plus two decks made from it (§8): one with a lone solid particle, and an 8× replication (1.9 M particles) |

This follows [REVIEW-2026-11-19-multi-gpu.md](REVIEW-2026-11-19-multi-gpu.md). Finding IDs continue its numbering: C16–C20, L4–L5, P15–P19, V3, M9.

---

## 1. Summary

**The fixes work.** I checked V2 independently on a second architecture and compiler. Every decomposed run is byte-identical to the single-GPU run:

| run | steps | halo rebuilds | result files byte-identical to one GPU |
|---|---:|---:|---|
| K=2 (2×1×1) | 1001 | 0 | 3 / 3 |
| K=4 (2×2×1) | 1001 | 1 | 3 / 3 |
| K=8, auto split (2×2×2) | 6001 | 19 | **13 / 13** |
| K=4, forced z-slabs (1×1×4) | 6001 | 19 | **13 / 13** |

The last two are the configurations that were 4.4 mm / 19 kPa and 2.1 cm / 29 kPa off in the previous review. Reading the code, C9–C15, P8–P11, P13 and P14 are implemented as described. The P10 double-buffer argument holds. The P11 two-layer solid halo is complete: layer 2 is the 3h-grown reference bounding box of layer 1, a superset of every layer-1 partner. A layer-1 ghost builds the same partner list, in the same order, as its owner.

**Errors found.**

- **C16 (Medium): a rank can skip the collective A6 decision and desynchronise every barrier after it.** Demonstrated: the run reads a displacement of `inf`, rebuilds from garbage, then hangs.
- **C17 (Low–Medium):** repartition re-cuts with solid weight 0.5, not the configured 0.45, so the load-balance setting is lost at the first rebuild.
- **C18 (Low):** the displacement trigger can overshoot s/2. The Verlet guarantee holds only approximately; nothing was lost in these runs.
- **C19 (Low):** a rank whose solids have no pairs silently switches to different solid kernels than the single GPU uses.
- **C20 (Info):** two benign data races and doc nits.

**Performance.** Two regressions, each with a fix that is cheap or already outlined:

- **P15 ★ (new, measured).** V2 gives every rank the *global* grid, and `k_cell_scan` is a single 512-thread block. Each rank now scans every cell of the domain every step. On the 19.3 M grid (~3.1 M cells) that is **6.5 ms per rank per step**, about +20% on the 28.6 ms K=8 step; on the 1203k grid it is about 0.46 ms (~+12%). A device-wide CUB scan does the same work in **0.012–0.030 ms** with identical output (checked).
- **P16 ★ (the log's open item, now measured).** A halo rebuild is a full repartition, and it happens more and more often as the flow speeds up: 19 rebuilds in 6000 steps at K=8, one every ~205 steps after step 4000. At the log's 260 ms per event, that is ~1.3 ms/step on a ~2.3 ms step late in the run, **+55% even at 237k**. The phase breakdown shows that about 60% of an event at 1.9 M is rank 0 working alone, half of that re-sorting the whole deck. A halo refresh needs none of it.

**Choosing the cards.** `SPH_RANKS=auto` predicts a 6.2× speedup at K=8 on this deck, while the log measured 1.4×. The exchange floor (4000 units ≈ 0.1 ms) is about 18× too small, so auto mode will nearly always take every card (L4). Its bandwidth probe also measures on-card memory, which cannot see PCIe links, and it leaves a context on every card (L5).

### Findings at a glance

| id | severity | area | finding |
|---|---|---|---|
| C16 | **Medium** | D1 × A6 | `width <= 0` returns before the collective A6 hook → barrier desync, `inf` displacement, hang (demonstrated) |
| C17 | Low–Medium | partition | repartition uses solid weight 0.5, start-up 0.45; `make_part` identifies solids by `w == 0.5` |
| C18 | Low | C9 trigger | the scheduled check lets the displacement pass s/2 by up to 10% of s (seen: 7%), and the detecting step runs on the old halo |
| C19 | Low | D1 × A3 | `npair > 0` picks static vs dynamic solid kernels per rank; a rank without pairs leaves V2's identity |
| C20 | Info | misc | `J.outStage` and `sph_a3_lean` written by every rank; doc nits |
| P15 | ★ | cell list | single-block scan over the global grid: 6.5 ms/rank/step at 19.3 M; CUB: 0.03 ms (verified identical) |
| P16 | ★ | rebuild | a halo refresh runs a full repartition (serial sorts, full re-upload); cost and frequency measured, plan given |
| P17 | | rebuild | carry the displacement in the E1 block: an exact check every step at no extra copy or sync (fixes C18) |
| P18 | | rebuild | per-particle slack: only motion that can reach a rank boundary should trigger a rebuild |
| P19 | | reporting | per-rank ms/step excludes rebuild and output time; quote the wall-clock line |
| L4 | Medium | auto-K | exchange floor ~18× too small; the floor is also charged to K = 1 |
| L5 | Low | auto-K | the probe measures on-card memory bandwidth, not PCIe; it leaves a context on every card |
| V3 | | testing | degenerate partitions, tagged barriers, runs ≥ 6000 steps |
| M9 | Info | docs | README "exchanged twice per step"; "the reduce runs every step"; M8 default vs the reference |

### Suggested order of work

1. **C16**: about 10 lines, plus tagged barriers in debug builds. **C17**: one weight function, used everywhere.
2. **P15**: drop-in CUB scan. Removes the V2 regression and speeds up the single-GPU path too.
3. **P16 + P17**: cheap halo refreshes, and the exact per-step check (which also fixes C18).
4. **L4/L5**: calibrate the auto-K model in milliseconds, and probe PCIe.
5. **P18**, then P12 (overlap).

---

## 2. Correctness

### C16 — A rank can skip the collective A6 decision (Medium)

`build_solid_lists()` returns early when its computing solids have no A3 pair at all ([sph_host.cu:529](../../cuda/src/sph_host.cu#L529)). That return sits *before* the C12 hook ([line 542](../../cuda/src/sph_host.cu#L542)), and the hook calls `J.bar.wait()` twice ([`a3_fat_agreed`](../../cuda/src/sph_host.cu#L1073)). The rank barrier only counts arrivals; it does not know which call site they come from. The rank that returned early arrives instead at its *next* barrier, the E1 exchange. Every barrier after that pairs up out of phase: ranks relay send blocks that have not been written yet, and the displacement slots they read are stale.

**Reproduction.** I took `input.dat` and moved one solid particle into the fluid, 0.41 m from every other solid (`make_decks.py lone`). At K=2, rank 0 owns that particle and has no solid pair. The same deck on one GPU runs to completion. At K=2:

```
[D1] repartition at step 8 (skin spent: inf mm moved, skin 15.00 mm): grid 2x1x1, ghosts 13085 (+5.5%) ...
```

The ghost slots dropped from 29,538 to 13,085 (a "rebuild" computed from inconsistent data), and the process then hung (killed after 60 s). With the hook also called on the `width <= 0` path, the run completes.

**Fix.** Join the hook before the early return, or better, take the A6 decision once in `rank_main`, with every rank participating unconditionally, before `build_solid_lists`. Structurally, any collective called from a function with early returns is fragile. A debug-mode **tagged barrier** would catch the whole class: `bar.wait(tag)` records the tag of each arrival, and the last arrival asserts they all match. That turns a silent desync into an immediate error naming the two call sites.

### C17 — The solid weight changes at the first rebuild (Low–Medium)

`build_job` and `run_auto` weight a solid with `g_w_solid` (0.45, or `SPH_D1_W_SOLID`) ([line 1400](../../cuda/src/sph_host.cu#L1400)). `repartition` hard-codes 0.5 ([line 2198](../../cuda/src/sph_host.cu#L2198)).

Measured: at start-up every rank of the K=8 split carries about 16,050 load units (balanced at 0.45). The first rebuild reports `max load 17091`, which is exactly rank 5's load at 0.5 (6,650 fluid + 0.5 × 20,880 solid). From then on the run is balanced at 0.5, whatever was configured.

Related: `make_part` recognises solids by comparing weights, `D.w[g] != 0.5` / `== 0.5` ([lines 1337 and 1363](../../cuda/src/sph_host.cu#L1337)). At the default 0.45, the C14 reference-shell term of the ghost estimate is skipped at start-up and in auto-K, but it is counted at rebuilds (where the weight is 0.5). Setting `SPH_D1_W_SOLID=1.0` would make solids indistinguishable from fluid. This only affects the estimate, not the real ghost selection, which `build_ranks` does by `itype`.

**Fix.** One weight function used in all three places, and an explicit type (or "is solid" flag) in `Deck` instead of comparing weights.

### C18 — The displacement trigger can overshoot s/2 (Low)

The check runs every `gap` ≤ 64 steps, with the gap chosen so that at most 10% of the skin grows unseen. It rebuilds when `dm > s/2` ([line 1876](../../cuda/src/sph_host.cu#L1876)). So `dm` can reach s/2 + 0.1 s before anyone sees it. Measured at the 19 triggers of the 6001-step K=8 run: **7.72–8.56 mm** against s/2 = 7.5 mm, i.e. up to 1.06 mm (7% of s) past the bound.

In addition, the rebuild happens after the step in which the crossing was detected, so that step still runs on the old halo. The Verlet argument needs `dm ≤ s/2` in every step that uses the halo. At 0.6 s per side, a pair can close by 1.2 s, which brings it 0.2 s (3 mm here) inside the 3h support unseen. Such a pair is at q ≥ 1.8, where the kernel is ≤ 5e-4 of its peak. The growth rate is extrapolated from the previous interval, so a sudden acceleration (impact, splash) can exceed the 10% allowance; with `MaxV = 30 m/s`, the worst case over 64 steps is 19 mm.

Nothing was lost in these runs (byte-identical through 6000 steps), so this is a gap in the guarantee, not an observed error.

**Fix.** P17 gives an exact check every step at no cost. As a one-line stopgap, reserve the allowance inside the budget: trigger at `dm > 0.4 * J.skin`.

### C19 — Static or dynamic solid kernels are chosen per rank (Low)

`launch_pass1_solid` and `launch_pass2_solid` use the static list only `if (sph_use_static_solid && s.npair > 0)` ([sph_kernels.cu:2027 and 2047](../../cuda/src/sph_kernels.cu#L2027)), and `npair` is per rank. A rank whose computing solids have no pair silently runs the cell-walk kernels, while one GPU runs the static ones for the same particles. The two paths are not bit-identical, and for a partner-less solid they disagree outright: static `pass2` gives `PIJW = NaN·0 = NaN`, dynamic gives 0.

With the lone-solid deck and the C16 fix, every fluid row differs at step 500 (up to 0.19 mm and 1,056 Pa). The lone solid stays finite and falls through the fluid on K=2, while on one GPU it goes NaN and drops out of the neighbour search. The deck is degenerate, but this breaks V2's identity for any deck with isolated solids.

**Fix.** Choose the path once, globally, and let the static kernels run with zero partners on such ranks. It would also help to warn at start-up about solids with no A3 partner (they already trigger the A = I warning).

### C20 — Smaller items (Info)

- Every rank executes `J.outStage = 1 - s` ([line 2477](../../cuda/src/sph_host.cu#L2477)), and every rank writes `sph_a3_lean`. They write the same value, so this is benign, but these are data races on plain ints. Let rank 0 write them; the following barrier publishes the value.
- The log attributes the ≤ 1-ulp change in some solid stresses to "instruction scheduling". Scheduling cannot change floating-point results; FMA contraction can. `--fmad=true` lets the compiler fuse differently when the surrounding code changes. This is harmless for V2, because K = 1 and K > 1 run the same binary. If bit-stability across refactors matters for the constitution, pin its arithmetic with `__fma_rn` / `__dmul_rn`.

---

## 3. Performance

### P15 — The global grid makes every rank scan every cell, every step ★

V2 hands every rank the grid derived from the whole deck. The per-step cell-list kernels that scale with *cells* rather than particles are then paid for the whole domain on every rank. Most of the cost is `k_cell_scan`, a single 512-thread block ([sph_kernels.cu:308](../../cuda/src/sph_kernels.cu#L308), launched at [line 1951](../../cuda/src/sph_kernels.cu#L1951)), which costs about 2.1 ns per cell.

Measured on the 4090 with the solver's own launch wrappers ([d1_check/bench_grid.cu](d1_check/bench_grid.cu); one rank occupying 1/8 of the cells). The kernel runs on one SM, so the 5090 D should be in the same range.

| grid | cells | memsets | `k_cell_scan` | `k_cell_sort` | **CUB scan + cursor copy** |
|---|---:|---:|---:|---:|---:|
| 237k deck | 21,850 | 0.006 | 0.076 | 0.013 | 0.013 ms |
| 1203k deck (105×50×38) | 199,500 | 0.007 | **0.464** | 0.017 | 0.012 ms |
| | 800,000 | 0.010 | 1.836 | 0.040 | 0.018 ms |
| 19.3 M deck (≈ 1656×50×38) | 3,100,000 | 0.021 | **6.508** | 0.097 | **0.030 ms** |

Before V2, each rank's grid covered roughly its own block. At K=8 on the 1203k deck, the scan per step went from about 0.06 ms to 0.46 ms (≈ +0.4 ms on a 3.4 ms step). On 19.3 M it went from about 0.9 ms to 6.5 ms (≈ +5.6 ms on 28.6 ms). The single-GPU path pays the same 0.46 ms (1203k) and 6.5 ms (19.3 M) and always has.

**Fix (verified).** Replace the kernel with `cub::DeviceScan::ExclusiveSum` plus `cudaMemcpyAsync(cellCur ← cellStart)` ([d1_check/bench_cub.cu](d1_check/bench_cub.cu)). The output is identical to `k_cell_scan` at all four sizes. It is an integer scan, so the cell lists, the summation order and V2's byte-identity are unchanged. For `start[ncell]` (the total), give `cellCur` one trailing zero slot and scan `ncell + 1` entries. An alternative that also keeps V2 is per-rank windows of the global grid: global origin, integer offsets, local dimensions.

### P16 — A halo refresh runs a full repartition ★

**Frequency.** The interval between rebuilds shrinks as the dam-break flow accelerates. Measured at K=8, 6001 steps: rebuilds at steps 712, 1296, 1880, 2400, 2791, then every 285, 252, 235, …, settling at **~205 steps from step 4000 on**, for 19 in total.

At the log's 260 ms per event on KMG-XI, that is about 0.45 ms/step early and **~1.3 ms/step late**, on a ~2.3 ms step: +20% early and +55% after step 4000, about +36% averaged over 6000 steps. The log's "~0.3–0.5 ms/step" covered only the first 3000 steps.

**Cost breakdown.** I instrumented `repartition()` ([d1_check/patch_rebuild_timing.py](d1_check/patch_rebuild_timing.py)), with K=8 and all ranks on one GPU. The host-side phases are representative:

| phase | 237k | 1.9 M (8× copies) | who |
|---|---:|---:|---|
| snapshot: 23 D2H copies + gid scatter into `J.pos` / `J.mig` | 7–16 ms | 97–123 ms | every rank |
| **3 axis sorts of the whole deck** | 36.5 ms | **315 ms** | rank 0 alone |
| `make_part` | 5.4 ms | 75 ms | rank 0 alone |
| `cur` = copy of the whole deck | 8.5 ms | 70 ms | rank 0 alone |
| `build_ranks` (O(num·K) ghost selection, rows) | 12.0 ms | 90 ms | rank 0 alone |
| state rebuild: free all, `finish_deck` (alloc + upload), `mig_restore`, `cudaHostAlloc` ×3 | ~43 ms | 176–192 ms | every rank |
| static exchange | 1–5 ms | 0–17 ms | every rank |
| solid pack + A3 list | ~11 ms | 19–43 ms | ranks with solids |
| **event total** | **118 ms** | **912 ms** | |

At 1.9 M, rank 0 works alone for 555 ms of the 912 ms while the other ranks wait, and the sorts alone are a third of the event. Extrapolated to 19.3 M, that is ~10 s on this machine, consistent with the log's 14–21 s on KMG-XI.

**Plan, in order of payoff.**

1. **A halo refresh is not a repartition.** For a skin rebuild, keep the cut planes (or the ownership itself) and only re-select ghosts. Re-cut, with its sorts, only for load balance: rarely, or when measured per-rank compute times drift apart. If re-cutting must stay cheap, weighted quantiles need a selection (`nth_element`-style) or a GPU radix sort, not three full `std::sort`s.
2. **Select ghosts in parallel.** Each rank thread can build its own ghost list from `J.pos`, or each owner can test its particles against the K−1 expanded boxes on the GPU and produce send lists directly. That removes rank 0's O(num·K) loops and the whole-deck `cur` copy (~2.2 GB at 19.3 M).
3. **Keep device state across rebuilds.** Size the arrays with capacity (`numP_cap`), so a rebuild only rewrites the ghost block and the strides do not change. With ownership kept, the owned block never leaves the GPU, so no snapshot or restore is needed. Reuse the pinned buffers; `cudaHostAlloc` per event is slow on WDDM.
4. **Keep the A3 lists in gid space.** The partners are fixed in the reference configuration, so store each list once, in the order the build produced (which V2 relies on), and remap gid → slot at a rebuild instead of re-binning `X0` and re-running count/fill.

With 1–3, a skin rebuild becomes O(ghosts) of parallel work. I would expect tens of milliseconds at 237k and well under a second at 19.3 M. That also makes a smaller skin affordable, which shrinks the halo.

### P17 — An exact displacement check every step, at no cost

The check is scheduled every ≤ 64 steps because a synchronous 8-byte readback costs ~0.6 ms on WDDM ([line 2097](../../cuda/src/sph_host.cu#L2097)). But E1 already copies the send block to the host and synchronises the stream every step. Let `k_disp_max` reduce into 8 bytes at the tail of `dStage`, just past the E1 records, and extend the D2H by those 8 bytes. After the E1 barrier, each rank reads every peer's value from the peer's send block, which is the same relay it already performs.

This gives the exact global maximum every step, with no extra copy, no extra sync and no rate extrapolation. It removes C18's overshoot. Trigger at `s/2 − MaxV·dt` to cover the step that is already running.

### P18 — Trigger only on motion that can reach a rank boundary

The trigger uses the largest displacement of *any* particle. A particle deep inside its rank's block cannot create a missing pair until it reaches the boundary. Take p, owned by r, at depth d_p inside r's box (the minimum over the box faces), and q, held nowhere near r, whose excess beyond r's 3h + s shell is e_q. They can only meet if δ_p + δ_q > s + d_p + e_q.

So the trigger may test max_i(δ_i − σ_i) > s/2, with σ_i = min(the particle's depth in its own box, its excess beyond every foreign shell that does not hold it). σ_i is computed once at build time. The per-step cost is the same as today (one extra array read in `k_disp_max`), and fast motion away from the boundaries stops forcing global rebuilds.

### P19 — Report wall-clock step time

`rank_main` accumulates `step_s` only around `step_d1` ([line 2453](../../cuda/src/sph_host.cu#L2453)). The per-rank "ms/step" lines therefore exclude rebuilds and output; only the final "slowest rank" line is wall clock. In the 6001-step K=8 run: 11.67 ms/step per rank against 12.04 ms/step wall, with 2.08 s spent in rebuilds. On KMG-XI, where a step is ~2.3 ms, the gap is much larger. Quote the wall-clock figure in performance tables, and add the rebuild count and total rebuild time to the summary line.

---

## 4. Choosing the cards (L1 follow-up)

### L4 — The auto-K model is not calibrated (Medium)

The prediction table for `input.dat` with eight devices (scratch build that stops after the table):

| K | split | predicted T (load units) | predicted speedup |
|---:|---|---:|---:|
| 1 | — | 132,394 | 1.0× |
| 2 | 1×2×1 | 69,679 | 1.9× |
| 4 | 2×2×1 | 37,696 | 3.5× |
| 8 | 2×2×2 | 21,413 | **6.2×** |

The log measured K=1 at 3.25 ms and K=8 at 2.2–2.4 ms on this deck, i.e. **1.4×**. One load unit is worth 3.25 ms / 128,394 ≈ 25 ns, so the default floor `xch_units = 4000` ([line 2595](../../cuda/src/sph_host.cu#L2595)) corresponds to ~0.1 ms. The measured K=8 overhead is ~1.8 ms, about 72,000 units. With the default the model will nearly always pick K = ndev, the opposite of the "keep cards free" intent. The floor is also charged to K = 1 ([line 2607](../../cuda/src/sph_host.cu#L2607)), which has no exchange.

**Fix.** Express the model in milliseconds and calibrate it:

- the per-unit cost from a K=1 timing (or `SPH_BENCH`);
- the floor from a short K=2 run's exchange split (`SPH_D1PROF`);
- a floor that grows with K (more peers, and the slowest card gates every barrier);
- no floor for K = 1.

### L5 — The device probe measures the wrong bandwidth and pins every card (Low)

- `rank_devices()` times a 64 MB device-to-device copy ([line 2546](../../cuda/src/sph_host.cu#L2546)). That is on-card DRAM bandwidth, identical on the eight 5090 Ds, and it cannot see that GPU 0 sits on a PCIe Gen2 slot. The exchange runs over pinned host↔device copies, so time a pinned D2H/H2D instead, or read the link with NVML (`nvmlDeviceGetCurrPcieLinkGeneration` / `Width`). On Windows every WDDM device reports the watchdog flag, as the log observed. NVML's `nvmlDeviceGetDisplayActive` identifies the cards actually driving a display.
- The probe creates a context and 128 MB of allocations on *every* device and never releases the context. The cards auto mode leaves free keep a context (~0.3–0.5 GB) for the whole run. `cudaDeviceReset()` the probed devices that are not chosen.

---

## 5. Verification (V3)

- **Degenerate partitions.** Add the lone-solid deck (C16, C19), a rank that owns no solid, and K larger than the number of solid-bearing blocks to the byte-identity set.
- **Tagged barriers** in debug builds (C16).
- **Run long.** The rebuild interval on this deck shrinks from ~580 to ~205 steps after step 3000. Validation and performance numbers taken over 501 or 3001 steps miss the expensive part.

## 6. Documentation (M9)

- [cuda/README.md:99](../../cuda/README.md#L99) says the fields are "exchanged twice per step"; there is one exchange point now.
- [cuda/README.md:124](../../cuda/README.md#L124) and the work log §5 say "the reduce runs every step". It runs only on check steps, inside `disp_publish`.
- The M8 default (reflect only outward motion) now differs from the reference `1.cpp`. A CPU-vs-GPU validation on a deck that reaches z = 1 needs `SPH_REFLECT=flip`. Say so where the validation is described, or keep the reference behaviour as the default until the physics owner decides.
- The step-time tables should say which figure they quote (P19).

---

## 7. What was checked and how

- **Byte-identity:** the table in §1, compared with [d1_check/cmp.py](d1_check/cmp.py) (SHA-256 per result file).
- **C16:** HEAD vs the same build with the hook joined on the `width <= 0` path, on the lone-solid deck at K=2 (HEAD: `inf` displacement, hang; fixed: completes).
- **C17:** start-up vs first-rebuild `max load` in the K=8 log, against per-rank weights at 0.45 and 0.5.
- **C18:** the `skin spent: … mm moved` values of all 19 triggers.
- **P15:** [bench_grid.cu](d1_check/bench_grid.cu) and [bench_cub.cu](d1_check/bench_cub.cu), linked against `cuda/src/sph_kernels.cu`.
- **P16:** [patch_rebuild_timing.py](d1_check/patch_rebuild_timing.py) on `input.dat` and on the 8× deck, with `SPH_D1_REPART=50` / `20` forcing events.
- **L4:** a scratch build that calls `choose_auto_k(D, 8, …)` and exits.

## 8. Reproduction

```bash
S=/path/to/scratch; DECK=$PWD/input.dat
cp cuda/src/* $S/
nvcc -ccbin g++-11 -O3 -std=c++17 --fmad=true -arch=sm_89 -I$S $S/sph_kernels.cu $S/sph_host.cu -o $S/sph
# byte-identity, 6001 steps
mkdir -p $S/one $S/k8
(cd $S/one && SPH_MAXSTEP=6001 $S/sph $DECK)
(cd $S/k8  && SPH_MAXSTEP=6001 SPH_RANKS=8 SPH_RANKS_GPUS=0,0,0,0,0,0,0,0 $S/sph $DECK)
python3 review/Claude/d1_check/cmp.py $S/one $S/k8
# C16: hangs at HEAD (use a timeout)
python3 review/Claude/d1_check/make_decks.py lone $DECK $S/lone.dat
mkdir -p $S/lone && (cd $S/lone && SPH_MAXSTEP=501 SPH_RANKS=2 SPH_RANKS_GPUS=0,0 timeout 60 $S/sph $S/lone.dat)
# P15
nvcc -ccbin g++-11 -O3 -std=c++17 -arch=sm_89 -Icuda/src review/Claude/d1_check/bench_cub.cu cuda/src/sph_kernels.cu -o $S/bench_cub && $S/bench_cub
# P16
python3 review/Claude/d1_check/make_decks.py rep $DECK $S/rep8.sphbin 8
mkdir -p $S/tim && cp cuda/src/* $S/tim/ && python3 review/Claude/d1_check/patch_rebuild_timing.py $S/tim
nvcc -ccbin g++-11 -O3 -std=c++17 --fmad=true -arch=sm_89 -I$S/tim $S/tim/sph_kernels.cu $S/tim/sph_host.cu -o $S/sph_tim
(cd $S/k8 && SPH_MAXSTEP=61 SPH_D1_REPART=20 SPH_RANKS=8 SPH_RANKS_GPUS=0,0,0,0,0,0,0,0 $S/sph_tim $S/rep8.sphbin | grep "\[tim\]")
```

`d1_check/patch_inst.py` (the halo checker of the previous review) targets the code before `3f171f1` and no longer applies. With V2 in place, the byte-identity comparison is the stronger check.
