# Review: why the decomposed run is barely faster than one GPU (D1, third pass)

| | |
|---|---|
| Date | 2026-11-23 |
| Reviewer | Claude (Opus 5.5) |
| Code | HEAD `5f05f99`; the D1 code is as of `f0b7700` (C16–C20, P15–P17) |
| Read | all of `cuda/src/*`; `docs/edit-2026-11-20-{01-38,11-53,13-05}.md`; `docs/perf-work-kmg-xi.md` §3–4; `cuda/README.md` |
| Test machine | one RTX 4090 (`sm_89`, 128 SMs), CUDA 12.2, g++-11, Ubuntu 22.04. **The card was shared with another user's training job for the whole session: 190 MiB of VRAM free after context creation, ~65 % utilisation by the other process.** `input.dat` did not fit, so most runs use two reduced decks (§9), with same-window ratios only. The other job ended late in the session, and the `input.dat` runs of §3.4 and §7 were then taken on the idle card. Decomposed runs put every rank on this one card (`SPH_RANKS_GPUS=0,0,…`): they test correctness and host logic, not 8-GPU step times. |
| Decks | `narrow.dat`, 79,424 particles (28,350 fluid, 15,090 solid, 35,984 wall): the tank of `input.dat` cut to 15 cm in y, closed by moving the far wall in, with walls and plate trimmed above z = 0.65 m. `dam.dat`, 63,044 particles (11,970 fluid): the same tank with only the fluid at x < 0.2 m kept, i.e. a dam-break column that runs into the plate. |

Finding IDs continue [REVIEW-2026-11-19-d1-followup.md](REVIEW-2026-11-19-d1-followup.md): C21–C23, P20–P29, V4, M10.

---

## 1. Summary

### Errors

- **C21 (High): the halo-refresh decision is not collective.** Since P17, a rank takes the maximum displacement over itself and its *linked* peers only. In any split where two ranks are not neighbours, they can disagree on the step at which to rebuild. The ranks that rebuild free their pinned send blocks while the others are still relaying from them. **Reproduced:** `dam.dat`, K = 4, `SPH_D1_SPLIT=4,1,1` → segmentation fault at the first refresh (step 1864). The **default** K = 8 split of the same deck (2×1×4) crashes the same way. Fix: reduce over all K tails (about 10 lines). With the fix, the K = 4 run is byte-identical to K = 1 through 2 refreshes, and a 30,001-step K = 8 run is byte-identical through 110 refreshes (61/61 files).
- **C22 (High with `SPH_D1_REPART`; certain to bite with `SPH_D1_REGRID`): a re-cut selects the halo around the previous boxes.** P16 restructured `repartition()` and lost `J.box = P.box`. **Measured:** K = 8 starting from 8×1×1 with `SPH_D1_REGRID=1` re-grids to 2×1×4 at step 1000. From then on it drifts away from K = 1: 0.36 mm / 1,594 Pa at step 1500, 4.8 mm / 3,700 Pa at step 3000. With the line restored, it is byte-identical.
- **C23 (Low): an empty block makes a rank own its ghosts.** Latent: `finish_deck()` reads `ownNum == 0` as "single GPU".

### Why K = 8 is barely faster

The KMG-XI logs give K = 8 at 2.2–2.4 ms/step against 3.25 ms on one card for 237k (1.4×). For 1.2 M they give 3.4–4.7 ms against 11.3 ms (2.4–3.3×); that was measured on the older three-exchange build, before the skin and the halo refreshes existed. The decomposition is sound and byte-identical, but most of a K = 8 step is not compute (§3.1):

1. **P20 ★★ — Ownership never changes, so the halo grows without bound.** P16 keeps every particle's owner for the whole run and re-derives a rank's box as the bounding box of its own particles. As the flow mixes, the boxes spread and overlap. Measured on `dam.dat`, K = 8, 30,001 steps: ghost slots went from **46,320 to 211,915** (73 % → 336 % of the deck), refresh cost from 27 to 69 ms, and the refresh interval from 1,083 to 210 steps. Exchange volume, cell-list size, layer-1 solid work, refresh cost and memory all grow with the halo. A long run ends up holding several copies of the deck.
2. **P21 ★ — Three host synchronisations and ~31 stream operations per step, where one sync is needed.** On WDDM each sync is a round trip, and around each one the GPU idles while the host relays and launches. This could not be measured here: on Linux, with every rank on one card, removing the two extra syncs changes nothing (10.89 vs 10.91 ms/step).
3. **P22 ★ — The per-rank kernels leave most SMs idle.** Every launch uses 256-thread blocks. At K = 8, a rank of the 237k deck has ~12.7k fluid particles: 50 blocks for 170 SMs. Measured on the 4090: with 256-thread blocks, `pass1_fluid` takes the same 0.52 ms for 12k fluid particles as for 28k. Choosing the block size per launch cuts it to 0.21 ms (2.45×). On `input.dat` with eight ranks on one card, the step drops from 10.92 to 7.80 ms (−29 %), and at K = 1 from 4.22 to 3.71 ms (−12 %); both are byte-identical. The same block granularity explains why `perf-work-kmg-xi.md` §4 found 1.53×, not 2×, for 170 SMs against 84.
4. **P23 — A halo refresh is still O(N) and serialised.** Each event frees and re-allocates ~60 device buffers per rank, pins three host blocks, uploads from pageable memory, and runs rank 0's O(N·K) ghost selection. Late in a 237k run that is 0.25–0.75 ms/step, and with P20 it grows.
5. **Environment (P27, P28).** KMG-XI is shared: the last two sessions ran next to a 24-worker GPU sweep, three workers per card. Whenever anything else runs on any card, a decomposed step waits at its barrier for the most contended card, while a single-GPU run only shares its own. In addition, GPU 0 sits on PCIe Gen2, GPUs 0 and 4 also drive the desktop, and the rank threads can land on the i9-14900K's E-cores.
6. **P29 — fp64 on GeForce.** The 5090 D runs fp64 at 1/64 of the fp32 rate. One card in fp32 (2.90 ms/step at 1.2 M) is already faster than eight cards in fp64 (3.4–4.7 ms).

### Findings at a glance

| id | severity | area | finding |
|---|---|---|---|
| C21 | **High** | P17 trigger | displacement max over linked peers only → ranks disagree on the refresh step → use-after-free or hang (reproduced; fix verified) |
| C22 | **High** with `SPH_D1_REPART` | P16 re-cut | `J.box` is not updated on a re-cut → the halo is chosen around the previous boxes |
| C23 | Low | rebuild | `ownNum == 0` means "single GPU"; an empty block after a re-cut makes a rank own its ghosts |
| P20 | ★★ | P16 refresh | ownership kept forever → halo grew 4.6× in 30k steps (measured) |
| P21 | ★ | step loop | 3 host syncs per step → 1; fold `k_xch_fixup` and `k_copy_xf` into the E1 scatter (byte-identical; the gain needs WDDM to show) |
| P22 | ★ | launches | block size chosen per launch from n and the SM count (byte-identical); `input.dat`: −29 % with 8 ranks on one card, −12 % at K = 1 (measured) |
| P23 | ★ | refresh | O(N) with realloc, pinning and pageable copies → make it O(ghosts) on the device |
| P24 | | exchange | zero-copy: the gather writes, the scatter reads mapped pinned memory → no D2H/H2D call, no CPU relay |
| P25 | | launches | CUDA Graphs for the step between refreshes |
| P26 | | overlap | interior/boundary split (P12), after P20 |
| P27 | | balance | per-rank times measured → weights; keep the Gen2 and display cards out (L4/L5 still open) |
| P28 | | platform | idle box, rank threads on P-cores, HAGS and maximum-performance power, Linux for production runs |
| P29 | | precision | fp32 or mixed precision on the decomposed path |
| V4 | | testing | slab splits, re-cuts, 30k-step runs that watch the ghost total, a consensus assertion |
| M10 | Info | docs | README on P16/P17 and on where byte identity was shown |

### Suggested order of work

1. **C21 and C22.** A few lines each; both are in [d1_scaling/quickwins.patch](d1_scaling/quickwins.patch).
2. **P21 and P22.** Also in the patch, and byte-identical to K = 1 at HEAD (§7).
3. **P20 with P23.** Own particles by position against fixed cut planes, migrate at a refresh, and make the refresh O(ghosts). Re-cut only for load balance.
4. **P24 and P25**, then **P26**.
5. **P27 and P28** before the next timing campaign: an idle machine, the right cards, pinned threads.
6. **P29** if the physics owner accepts single or mixed precision.

---

## 2. Correctness

### C21 — The refresh decision is not collective (High)

`disp_collect()` ([sph_host.cu:2106](../../cuda/src/sph_host.cu#L2106)) reads the displacement tail of its own E1 block and of each *linked* peer's block:

```cpp
double mx = R.hSendB[R.sendSlot][R.nSendE1];
for (const DLink& L : R.links)                          // linked peers only
    mx = max(mx, J.ranks[L.peer].hSendB[R.sendSlot][L.peerE1]);
R.disp = mx;
```

`step_d1()` turns that into `R.rebuild_due` ([line 1908](../../cuda/src/sph_host.cu#L1908)), and `rank_main()` enters `repartition()` on it ([line 2516](../../cuda/src/sph_host.cu#L2516)). `repartition()` is collective: it has two barriers of its own and two more in `xch_static()`. But links exist only between ranks that hold each other's particles, so in a slab split with K ≥ 3, rank 0 never sees rank 2's value. The code calls the value the "exact global max displacement" and the README says it is taken "over the owned particles of every rank"; it is the maximum over a neighbourhood.

When ranks disagree, the ranks that rebuild pair their first `repartition()` barrier with the others' next E1 barrier. They then run `xch_free()`, which calls `cudaFreeHost` on both send blocks, while the ranks that did not rebuild relay from those blocks.

**Reproduction** (§8). `dam.dat`, K = 4, `SPH_D1_SPLIT=4,1,1`, `SPH_D1_DBG=1`, HEAD:

```
[dbg] step: global max displacement 5.42823 mm (skin 15 mm)      <- rank 0: max over ranks 0 and 1
[dbg] rank 3: repartition(skin spent: 6.902 mm moved, skin 15.00 mm, recut=0) start
[dbg] rank 2: repartition(...) start
[dbg] rank 1: repartition(...) start
[dbg] rank 1/2/3: ... repartition stage 3 (...)
Segmentation fault
```

Rank 0 links only rank 1, and the fastest particles are at the plate, in ranks 2 and 3. Ranks 1–3 crossed the 6.9 mm trigger at step 1864; rank 0 did not. With other timings the same mismatch gives a hang, or a rebuild from inconsistent data as in C16.

**Exposure.** Every 1-D split with K ≥ 3, and every 2-D or 3-D split with a rank that does not touch all the others. That includes the **default** K = 8 split of `dam.dat` (2×1×4, where ranks have 2–5 peers out of 7). HEAD segfaults there too: at step 1864, ranks 0, 1, 4 and 5 entered the refresh and ranks 2, 3, 6 and 7 did not. On `input.dat` the auto split is 2×2×2 and all eight ranks are mutually linked, which is why the 6001-step validation of P17 passed. A deck whose central corner is empty would break 2×2×2 as well. A rank that sends no E1 record (`nSendE1 == 0`) returned 0 whatever its peers saw, which is the same failure.

**Fix (verified).** Every rank's D2H has completed before it arrives at the E1 barrier, and no rank can rewrite that half before the next barrier. Reading a non-peer's tail is therefore exactly as safe as the relay:

```cpp
double mx = 0.0;
for (int r = 0; r < J.K; ++r) {
    const Rank& P = J.ranks[(size_t)r];
    if (!P.dXBuild) continue;                      // rank with no owned particle
    mx = std::max(mx, P.hSendB[R.sendSlot][P.nSendE1]);
}
R.disp = mx;
```

Also drop the per-step `R.nSendE1 = bytes / sizeof(double)` in `xch_pack()` ([line 1769](../../cuda/src/sph_host.cu#L1769)). It rewrites the value `build_ranks()` set while other ranks read it: a benign race today, a real one once every rank reads every tail.

With the fix, the K = 4 x-slab run completes with two refreshes and is byte-identical to K = 1 (7/7 result files). A 30,001-step K = 8 run with the auto split (2×1×4, 110 refreshes) is byte-identical too (61/61).

A debug-build assertion would catch this whole class of bug. After the decision, each rank writes `rebuild_due` into a per-rank slot; after the next barrier, every rank checks that all the slots agree. The tagged barriers proposed in V3 are still open as well.

### C22 — A re-cut selects ghosts around the previous boxes (High with `SPH_D1_REPART`)

A re-cut happens only with `SPH_D1_REPART`; `SPH_D1_REGRID` changes what a re-cut does.

In `repartition()`, the re-cut branch ([lines 2200–2251](../../cuda/src/sph_host.cu#L2200)) computes a new `Part P` from the live positions and publishes `J.owner = P.owner`. It never publishes `P.box`. `build_ranks()` then selects every rank's ghosts against `J.box` ([line 2277](../../cuda/src/sph_host.cu#L2277)). Only the other branch refreshes `J.box`, so after a re-cut it still holds the bounding boxes of the *previous* ownership. `git show f0b7700` shows the deleted line: `-        J.box = P.box;`.

If a cut plane moved, a rank owns particles outside the box its halo was selected for, and their neighbours beyond `box + 3h + s` are never ghosts. The displacement trigger cannot catch this: it measures motion since the rebuild, and nothing has moved yet. The halo stays wrong until the next skin refresh recomputes `J.box` from the new ownership, typically a few hundred steps later.

**Measured.** `dam.dat`, K = 8, started as 8×1×1 (`SPH_D1_SPLIT=8,1,1`) with `SPH_D1_REPART=1000 SPH_D1_REGRID=1`, so the first re-cut changes the block grid and the stale boxes belong to a different factorisation. Binary: HEAD + the C21 fix.

| result | `J.box` stale (HEAD) | `J.box = P.box` restored |
|---|---|---|
| ghosts after the re-cut at step 1000 | 116,176 (selected around the old slabs) | 46,085 |
| step 1000 | byte-identical | byte-identical |
| step 1500 | 0.36 mm, fluid 1,594 Pa | byte-identical |
| step 2000 | 1.2 mm, 3,560 Pa | byte-identical |
| step 3000 | **4.8 mm, 3,700 Pa** | byte-identical |

The stale halo even held 2.5× more ghosts and still missed the ones the new blocks needed. A plain `SPH_D1_REPART=2000` run (same grid, 30,001 steps, 14 re-cuts) stayed byte-identical. There the reused boxes were the drifted bounding boxes of P20, which were *larger* than the new cut boxes: each re-cut selected ~1,200 more ghosts than the next refresh did. So the bug shows when a cut moves outward, past the old box, by more than the skin. A re-grid is the sure way; strong flow is another.

**Fix:** `J.box = P.box;` next to `J.owner = P.owner;`. It is in the patch.

### C23 — An empty block makes a rank own its ghosts (Low)

`finish_deck()` uses `ownNum == 0` to mean "single GPU, every row is owned" ([line 333](../../cuda/src/sph_host.cu#L333)). A decomposed rank whose block is empty therefore treats its ghost rows as owned particles: it computes and integrates them, and those particles then have two owners. `build_job()` rejects empty splits ([line 1463](../../cuda/src/sph_host.cu#L1463)), but the re-cut in `repartition()` does not check `P.empty`. With weight-balanced cuts this needs a degenerate deck. Give "single GPU" its own sentinel, and keep the old layout when a re-cut produces an empty block.

---

## 3. Where a K = 8 step goes

### 3.1 The budget

There is no idle-machine `SPH_D1PROF` table for the current build, so this is an estimate. Run `SPH_D1PROF=1` on an idle KMG-XI to replace it.

| 237k deck, K = 8, KMG-XI | ms/step | source |
|---|---:|---|
| measured step | 2.2–2.4 | `edit-2026-11-20-11-53.md` §6.2 |
| one-card step ÷ 8 (perfect scaling, no halo) | 0.41 | 3.25 / 8 |
| per-rank kernels with 256-thread blocks | ≈ 1.1 | the 4090 runs a deck of one rank's size (`dam.dat`, 12k fluid) in 1.10 ms of kernels (§3.4). In this one-block-per-SM regime a 5090 D's extra SMs sit idle, so the figure carries over. |
| the same with per-launch block sizes (P22) | ≈ 0.7 | ditto, 0.71 ms |
| **everything that is not compute**: E1 copies, three syncs, barrier wait, relay, ~31 launches | **≈ 1.1–1.3** | residual |
| halo refresh, amortised (P23) | + 0.25–0.75 | 150 ms per event (contended window) every ~200–600 steps |

About half of the step is synchronisation and transfer latency. Of the compute, a rank spends ~1.1 ms on what would take 0.41 ms if it scaled with the work. Neither term shrinks when you add GPUs. At 1.2 M the compute share is larger, but the same terms apply, and the refresh term grows with N.

### 3.2 P20 — Ownership never changes, so the halo grows without bound ★★

Since P16, a skin refresh keeps `J.owner` and recomputes each rank's box as the bounding box of the particles it owns ([lines 2252–2272](../../cuda/src/sph_host.cu#L2252)). Ownership changes only on an explicit `SPH_D1_REPART`, which is off by default. In a dam break the particles a rank owns spread across the tank. Its box grows to cover them, the boxes of different ranks overlap, and every refresh selects more ghosts.

Measured on `dam.dat` at K = 8 (auto split 2×1×4, C21 fixed, 30,001 steps, byte-identical to K = 1 throughout):

| step | ghost slots | % of the deck | refresh (ms, this box) | steps since previous refresh |
|---:|---:|---:|---:|---:|
| 0 | 46,320 | 73.5 | — | — |
| 1,864 (first refresh) | 44,902 | 71.2 | 45 | 1,864 |
| 5,554 | 51,008 | 80.9 | 17 | 537 |
| 10,109 | 66,182 | 105.0 | 31 | 362 |
| 14,932 | 93,647 | 148.5 | 32 | 297 |
| 20,056 | 127,192 | 201.8 | 36 | 135 |
| 25,565 | 178,269 | 282.8 | 60 | 187 |
| 29,805 (110th) | **211,915** | **336.1** | 69 | 217 |

From the first refresh on, the growth is monotonic, and it does not level off within the run. Everything that scales with the halo grows with it: E1 bytes, the cell-list passes over `numP`, the layer-1 ghost solids that run `pass1_solid` and the constitution, the refresh itself, and device memory. A production run (`SPH_MAXSTEP` defaults to 10⁷) ends with every rank holding most of the deck.

**Fix: keep the cut planes, not the ownership.** At a refresh, give each particle to the block that contains its current position under the existing cuts. That is an O(N) classification against nx + ny + nz planes, with no sort. The boxes stay the cut boxes, so the halo stays a shell of thickness 3h + s around a fixed box, and the number of peers stays bounded. Each refresh migrates the few hundred particles that crossed a plane; P23 shows how to do that on the device. Re-cut (with the sorts) only when the measured per-rank times drift apart (P27). Ownership does not enter any per-particle arithmetic, because cells are ordered by gid. The existing re-cut path already shows that, so K > 1 stays byte-identical.

### 3.3 P21 — One host synchronisation per step instead of three ★

Per step, each rank thread blocks three times:

| where | needed? |
|---|---|
| `cudaStreamSynchronize` after the E1 D2H ([line 1785](../../cuda/src/sph_host.cu#L1785)) | yes: the barrier publishes the block |
| `cudaStreamSynchronize` after the H2D, the scatters and `k_xch_fixup` ([line 1825](../../cuda/src/sph_host.cu#L1825)) | no: everything after it is on the same stream, and `hRecv`/`dStage` are rewritten only after the next step's first sync |
| `cudaDeviceSynchronize` after `step_d1()` ([line 2481](../../cuda/src/sph_host.cu#L2481)) | no: it only feeds `step_s`, which P19 replaced with the wall clock |

The second and third block the host until the GPU drains, and then the GPU idles while the host relays and launches the next batch. Without them, the host enqueues the whole step and blocks once, at the next E1 copy, while the GPU works. The logs put a synchronous round trip on KMG-XI at 0.15–0.6 ms (a 150 µs `cudaMemcpyAsync`, a 0.6 ms 8-byte readback), so this is worth measuring there. Here it shows nothing. With the patch and `SPH_BLK=256` (HEAD's block size, so only P21 differs), K = 8 on `input.dat` with all ranks on one card runs at 10.91 ms/step against HEAD's 10.89. On Linux a sync is cheap, and with eight ranks on one card the GPU always has another rank's work queued.

Two of the ~31 stream operations, `k_xch_fixup` and `k_copy_xf` (the latter over all `numP`), only post-process the slots the E1 scatter has just written. The scatter can do it with the same expressions. The patch does both changes; the results are byte-identical (§7).

### 3.4 P22 — The per-rank kernels leave most SMs idle ★

Every launch is `<<<ceil(n/256), 256>>>` ([sph_kernels.cu:1840](../../cuda/src/sph_kernels.cu#L1840)). `pass1_fluid` is fp64-pipe bound (`perf-work-kmg-xi.md` §4), and on these cards each SM has its own small fp64 pipe. So the kernel takes as long as the SM that received the most blocks, and SMs with no block contribute nothing.

| case | fluid / rank | blocks of 256 | SMs | blocks on the busiest SM | efficiency |
|---|---:|---:|---:|---:|---:|
| 237k, K = 1, 5080 | 101,250 | 396 | 84 | 5 (4 + 1 tail) | 94 % |
| 237k, K = 1, 5090 D | 101,250 | 396 | 170 | 3 | 78 % |
| 237k, K = 8, 5090 D | ~12,700 | 50 | 170 | 1 (120 SMs idle) | **29 %** |
| 1.2 M, K = 8, 5090 D | ~70,000 | 274 | 170 | 2 | 80 % |

The first two rows predict the 5080 → 5090 ratio of pass 1: 2.02 (SMs) × 0.92 (boost clock) × 0.78 / 0.94 ≈ 1.55. `perf-work-kmg-xi.md` §4 measured 1.53× and attributed it to fp64 not scaling with SM count.

**Measured on the 4090** (128 SMs, `SPH_BENCH`, minimum of three interleaved rounds on the shared card). `dam.dat` has 11,970 fluid particles, about one K = 8 rank of the 237k deck; `narrow.dat` has 28,350:

| kernel, ms | 12k fluid: 256 | 128 | 64 | 32 | per-launch | 28k fluid: 256 | per-launch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pass1_fluid` over the list | 0.520 | 0.272 | 0.268 | 0.211 | **0.212** | 0.529 | **0.458** |
| fluid neighbour-list build | 0.284 | 0.251 | 0.280 | 0.284 | 0.285 | 0.297 | 0.316 |
| `pass1_solid` (15k solids) | 0.113 | 0.068 | 0.069 | 0.068 | 0.067 | 0.113 | 0.068 |
| `pass2_solid` | 0.102 | 0.078 | 0.081 | 0.080 | 0.078 | 0.207 | 0.173 |
| constitution | 0.020 | 0.013 | 0.012 | 0.012 | 0.013 | 0.020 | 0.013 |
| all kernels of a step | 1.10 | | | | **0.71** | | |

- With 256-thread blocks, `pass1_fluid` takes the same time for 12k and 28k fluid particles: both are one block per busy SM. The model predicts 2.67× and 1.14× for the per-launch sizes; the measurements are 2.45× and 1.15×.
- The neighbour-list build is latency bound, not fp64 bound. It gains nothing from smaller blocks, and at 28k the per-launch choice costs it 6 %. Keep 256 for that kernel, or tune per kernel.
- Whole step at K = 1, interleaved A/B, minimum of three: `dam.dat` **1.145 → 0.792 ms/step (−31 %)**, `narrow.dat` 1.459 → 1.343 (−8 %).

**`input.dat`, idle card** (two interleaved rounds, 3001 steps; the rounds agree to 0.5 %):

| | HEAD | patch | |
|---|---:|---:|---|
| K = 1 step | 4.22 ms | 3.71 ms | −12 % |
| K = 8, all ranks on one card | 10.92 ms | **7.80 ms** | −29 % |
| K = 8, all ranks on one card, patch with `SPH_BLK=256` (P21 only) | 10.89 ms | 10.91 ms | ±0 |
| K = 1 `pass1_fluid` body (`SPH_BENCH`) | 2.010 ms | 1.748 ms | 1.15× |

With all ranks on one card, the ranks share the legacy default stream, so a step costs the sum of eight ranks' kernels: the −3.1 ms is the per-rank saving of P22, ~0.4 ms per rank. That matches the 1.1 → 0.7 ms per rank of the budget in §3.1. On eight cards each rank saves its share in parallel, so it comes off the critical path. At K = 1 the model over-predicts: it says 1.28× for the `pass1_fluid` body on 128 SMs, and the measurement is 1.15×.

On the 5090 D the model gives the same 2.7× for a K = 8 rank of the 237k deck (50 → 397 blocks of 32, three per SM). For one card on the full deck it gives 1.26× on the `pass1_fluid` body; going by the 4090, expect about 1.15×.

**Fix (in the patch).** Choose the block size per launch from n and the SM count. The patch takes the size in {256, 128, 64, 32} that minimises the per-SM critical path `ceil(ceil(n/bs)/SMs)·bs`. `SPH_BLK` forces one size for A/B tests. Each thread still does exactly the same arithmetic, so results are bit-identical. A further step is to give each fluid particle a warp-wide share of its neighbour list instead of a thread. That balances the 60–115 neighbour spread too, but it changes the summation order and therefore V2's reference bits.

### 3.5 P23 — A refresh is still O(N), with realloc, pinning and pageable copies ★

What each rank does at a skin refresh today ([`repartition()`](../../cuda/src/sph_host.cu#L2167)):

- 3 + 20 synchronous D2H copies into pageable vectors, and a scatter of them into the global `J.pos` / `J.mig`;
- `xch_free()` + `free_state()`: about 60 `cudaFree`/`cudaFreeHost` calls;
- `finish_deck()`: the block-order check over all rows, 37 `cudaMalloc` (most followed by a `cudaMemset`), and 8 uploads from pageable memory with AoS → SoA transposes;
- 20 H2D copies to restore the owned state, 2 `cudaMalloc` for the trigger, 3 `cudaHostAlloc` + about 12 `cudaMalloc` for the exchange;
- the static exchange, the solid pack, and the A3 list rebuilt from scratch (X0 re-binned, count, D2H, fill, D2H check, x re-binned).

Rank 0 alone also runs `build_ranks()`: two O(N) passes per rank (O(N·K)) plus a copy of every rank's rows. On WDDM, allocation and pinning are kernel-mode calls that serialise across the eight rank threads.

With ownership kept, almost none of this is needed: the owned block, its order and its A3 partner sets do not change; only the ghost block does. With cut-plane ownership (P20), a few hundred owned particles move per event. Plan:

1. **Stride by capacity.** Allocate `numP_cap` (owned + ghost headroom, grown ×1.5 when exceeded) and index the SoA arrays with the capacity instead of `numP` (the `N` macro and `sph_matvec`). A refresh then never re-lays out the owned block.
2. **Keep device state.** No snapshot/restore. Upload only the static fields (X0, mass, rho0, itype) of the new ghost rows. The dynamic fields arrive with the next E1, A and vecn with the static exchange. Migration (P20) appends arrivals and compacts departures on the device.
3. **Never free the exchange buffers; grow them.** No `cudaHostAlloc` per event.
4. **Select ghosts on the owner's GPU.** Each rank tests its owned particles against the K − 1 expanded boxes and compacts its send lists. Receivers get counts and offsets through the existing relay. Rank 0's O(N·K) loop and the whole-deck rows go away.
5. **Keep the A3 lists in gid space.** The partner sets are fixed in the reference configuration. Remap gid → slot with one device scatter (`map[gid[i]] = i`) instead of re-binning X0 and running count/fill again. V2 needs the partner order, and the stored list keeps it.

A refresh then costs a few milliseconds at 237k, mostly the static exchange, and should stay well below 100 ms at 19.3 M. That also makes a smaller skin affordable, which shrinks the halo.

---

## 4. Acceleration beyond the fixes

### P24 — Zero-copy exchange

There is no P2P on KMG-XI, so every byte goes through host memory, but the CPU does not have to touch it. Allocate the E1 send blocks of all ranks in one pinned arena with `cudaHostAllocMapped | cudaHostAllocPortable`, so every GPU can address it.

- The gather kernel writes its records straight into the arena (coalesced 64-byte records over PCIe). No D2H call.
- After the stream sync and the barrier, the scatter kernel reads each ghost record straight from the owner's half of the arena. It uses precomputed per-record source offsets, one set per buffer parity, and `__ldcv` loads so no stale line is reused. No CPU relay, no `hRecv`, no H2D call.

Per step that leaves one sync, one barrier and the launches. The bytes over PCIe are unchanged (5.43 MB/step summed over eight ranks at 237k, well under 1 GB/s per card). The step loses two copy submissions and the relay, about 0.1–0.2 ms of CPU memcpy per rank, more with 26 peers or a busy CPU.

A further step takes the host out of the per-step path entirely. The gather's last block issues `__threadfence_system()` and then writes a per-rank step counter into the arena. The scatter spins on its peers' counters (volatile loads) before reading. The displacement decision then reads the tails one step late, which costs one more `MaxV·dt` of trigger margin. On WDDM, call `cudaStreamQuery` after enqueuing the gather so the driver's batching does not hold it back. Prototype this on Linux first.

### P25 — CUDA Graphs for the step

Between two refreshes the step's parameters do not change: `SphPtrsT` is passed by value with fixed pointers. Capture from the scatter to the D2H of the next E1 as one graph after each refresh, and launch it once per step. The CUB scan is capturable once its scratch exists. On WDDM this replaces ~30 submissions of 5–10 µs each with one, and it shortens the gaps between dependent kernels. The single-GPU loop benefits the same way.

### P26 — Overlap the exchange with interior work (P12)

This needs P20 first: with cut-plane ownership, ghosts lie only outside the block, so a cell whose 27-neighbourhood lies at least one cell inside the block has no ghost in reach. Proposed sequence:

1. Build the cell list from the owned particles. For an interior cell it is identical to the full list, because cells are ordered by gid.
2. Run `pass1` for the interior particles on one stream while E1 is in flight on another.
3. Then insert the ghosts, run `pass1` for the boundary particles, and run the rest of the step.

Pass 1 is ~73 % of the step, and the interior of a block is most of it. For a rank with ≥ 1 ms of pass 1 (1.2 M and up at K = 8), that hides the exchange. pass2 and the constitution depend on boundary results, so they stay after the exchange.

### P27 — Balance by measurement, and choose the cards

- Every step ends at a barrier, so the slowest rank sets the pace. The weights (`g_w_solid`, `g_w_ghost`) are estimates from a 4090, and L2 still waits for the 5090 D numbers. Time each rank's compute with events (no sync needed). When the max/mean exceeds ~1.1, correct each rank's weight by its measured ratio and re-cut once.
- Keep GPU 0 (PCIe Gen2) and any card that drives a display out of the set, or give them a smaller share through a per-device speed factor in the partition objective. L5 (a probe that cannot see PCIe) and L4 (an uncalibrated auto-K model that nearly always takes every card) are still open. With the current fixed costs, K = 4 on four clean Gen5 cards may well match K = 8 on 237k–1.2 M decks; measure it.

### P28 — Platform

- **Measure on an idle machine,** and check that no other process holds a context on any of the K cards (`nvidia-smi`). With a 24-worker sweep on the same eight cards, every barrier waits for the most contended card, and WDDM time-slices the contexts of different processes. A rank's 5 µs kernels can wait behind another process's work. A single-GPU run shares only its own card, so the comparison is biased against K = 8.
- **Pin the rank threads to P-cores.** The i9-14900K has 8 P-cores and 16 E-cores, and `perf-work-kmg-xi.md` §5 already showed what an E-core does to a barrier-synchronised loop. Use one P-core per rank (`SetThreadAffinityMask` in `rank_main`) and keep the writer thread off them.
- **Windows settings.** Turn on hardware-accelerated GPU scheduling, and set NVIDIA power management to "prefer maximum performance" (or lock clocks), so that a card that idles at every barrier does not drop clocks.
- **Linux for production runs.** The same code without WDDM has much cheaper launches and syncs. This repository already builds there.

### P29 — Precision

On the 5090 D, fp64 runs at 1/64 of the fp32 rate. The measured single-card numbers for 1.2 M are 11.31 ms/step in fp64 and 2.90 in fp32 (fat list): fp32 on one card beats fp64 on eight. Options, in order of effort:

- **fp32 on the decomposed path.** Records stay doubles on the wire, or become floats. The fp32 path needs ghost solids' `u` in E1, because `k_pass1_solid_static` reads `u[q]` of its partners. `xc` and `rhoc` are owner-only. V2's identity carries over (the same arithmetic on both sides).
- **Mixed precision in `pass1_fluid`.** Form the pair geometry in fp64 (`d = x_p − x_q`), evaluate the kernel and the Riemann terms in fp32, and accumulate in fp64. This kernel is most of the step.
- **Cheaper fp64 `sqrt` and division.** `rsqrtf` plus one fp64 Newton step instead of the correctly rounded fp64 routines. This changes the last bits against the stored references; K > 1 versus K = 1 identity still holds.

All three need the physics owner's acceptance criterion; the fp32 accuracy numbers in the README are a starting point.

---

## 5. Verification (V4)

- Add to the byte-identity matrix: a slab split (`SPH_D1_SPLIT=4,1,1` and `8,1,1`), the auto split of a deck that is not 2×2×2, and `SPH_D1_REPART` runs. The current matrix (2×2×2 on `input.dat`) cannot see C21 or C22.
- Run ≥ 30,000 steps and print the ghost total at each refresh. P20 only shows up late.
- A debug assertion that all ranks agree on `rebuild_due` (C21), and the tagged barriers of V3.

## 6. Documentation (M10)

- [cuda/README.md](../../cuda/README.md): "A skin-triggered refresh keeps the ownership" should say that ownership then never changes unless `SPH_D1_REPART` is set, and that the halo grows (P20).
- README and `docs/edit-2026-11-20-13-05.md`: the displacement is reduced over the linked peers, not globally (C21). Once fixed, say "over all ranks".
- "K > 1 is byte-identical to K = 1" was shown for 2×2×2, a forced 1×1×4 before P17, and `SPH_D1_REPART=1000` before P16. Since P16/P17 the slab and re-cut cases fail (C21, C22). Say which configurations the claim covers.

## 7. What was checked and how

All comparisons use [d1_check/cmp.py](d1_check/cmp.py) (SHA-256 per result file) against HEAD's K = 1 run of the same deck. Decomposed runs use `SPH_A3_FULL=0`, because eight fat lists do not fit in 190 MiB; lean and fat are byte-identical on `dam.dat` (7/7 files at K = 1).

| check | binary | runs | result |
|---|---|---|---|
| C21 | HEAD | `dam.dat`, K = 4, `SPH_D1_SPLIT=4,1,1` | segfault at the first refresh (step 1864) |
| C21 | HEAD | `dam.dat`, K = 8, default split (2×1×4) | segfault at step 1864; ranks 0, 1, 4, 5 entered the refresh, ranks 2, 3, 6, 7 did not |
| C21 fix | HEAD + C21 | both of the above, 3001 steps | completes; 7/7 byte-identical |
| C21 fix, long | HEAD + C21 | K = 8 default split, 30,001 steps, 110 refreshes | 61/61 byte-identical |
| C22 | HEAD + C21 | K = 8 from 8×1×1, `SPH_D1_REPART=1000 SPH_D1_REGRID=1`, 3001 steps | diverges after the re-grid: 4.8 mm / 3,700 Pa at step 3000 |
| C22 fix | HEAD + C21 + C22 | the same | 7/7 byte-identical |
| C22, plain re-cut | HEAD + C21 | K = 8, `SPH_D1_REPART=2000`, 30,001 steps, 14 re-cuts | 61/61 byte-identical (not triggered, see C22) |
| P20 | HEAD + C21 | K = 8 default split, 30,001 steps | ghosts 46,320 → 211,915; refresh 27 → 69 ms |
| the patch | `git archive HEAD` + [quickwins.patch](d1_scaling/quickwins.patch) | `dam.dat` K = 1; K = 4 x-slabs; K = 8 default; K = 8 re-grid; 3001 steps each | 7/7 byte-identical in every case |
| the patch, K = 1 | prototype of the same code | `narrow.dat`, 3001 steps | 7/7 byte-identical |
| the patch on the real deck | the patch | `input.dat`, 6001 steps: K = 1, and K = 8 (2×2×2, 22 refreshes) | 13/13 byte-identical to HEAD's K = 1 in both |
| HEAD on the real deck | HEAD | `input.dat`, K = 8, 6001 steps | 13/13 byte-identical (2×2×2: all ranks linked, so C21 cannot occur) |
| P22 | the patch, `SPH_BLK` forced | `SPH_BENCH`, 5 block sizes × 2 decks × 3 interleaved rounds; K = 1 step A/B × 3 | §3.4 |
| P22, real deck | HEAD vs the patch | `input.dat`, K = 1 and K = 8 on one card, 2 interleaved rounds; `SPH_BENCH` × 2 | §3.4 |
| P21 | the patch with `SPH_BLK=256` | `input.dat`, K = 8 on one card, 2 rounds | no change on Linux; the benefit needs WDDM and eight cards |

Not checked: anything on the 5090 D or under WDDM, and the 1.2 M / 9.6 M / 19.3 M decks (not in the repository).

## 8. Reproduction

```bash
S=/path/to/scratch; R=$PWD/review/Claude
NV="nvcc -ccbin g++-11 -O3 -std=c++17 --fmad=true -arch=sm_89"
python3 $R/d1_scaling/narrow_deck.py input.dat $S/dam.dat    0.15 0.65 0.2
python3 $R/d1_scaling/narrow_deck.py input.dat $S/narrow.dat 0.15 0.65
# HEAD and patched builds
mkdir -p $S/head $S/fixed
git archive HEAD cuda/src | tar -x -C $S/head
git archive HEAD cuda/src | tar -x -C $S/fixed && (cd $S/fixed && patch -p1 < $R/d1_scaling/quickwins.patch)
for v in head fixed; do (cd $S/$v/cuda/src && $NV -I. sph_kernels.cu sph_host.cu -o $S/sph_$v); done
K8="SPH_RANKS=8 SPH_RANKS_GPUS=0,0,0,0,0,0,0,0 SPH_A3_FULL=0"
# reference
mkdir -p $S/k1 && (cd $S/k1 && SPH_MAXSTEP=3001 $S/sph_head $S/dam.dat)
# C21: HEAD segfaults at step 1864 with the default split; SPH_D1_DBG shows which ranks rebuilt
mkdir -p $S/c21 && (cd $S/c21 && env $K8 SPH_MAXSTEP=2001 SPH_D1_DBG=1 $S/sph_head $S/dam.dat 2>&1 | grep "repartition(")
# the fixed build: byte-identical, default split and x-slabs
mkdir -p $S/k8 && (cd $S/k8 && env $K8 SPH_MAXSTEP=3001 $S/sph_fixed $S/dam.dat)
python3 $R/d1_check/cmp.py $S/k1 $S/k8
# C22: byte-identical with the fixed build; delete its "J.box = P.box;" line and rebuild
# to see the drift that starts after the re-grid at step 1000
mkdir -p $S/c22 && (cd $S/c22 && env $K8 SPH_MAXSTEP=3001 SPH_D1_SPLIT=8,1,1 SPH_D1_REPART=1000 SPH_D1_REGRID=1 $S/sph_fixed $S/dam.dat)
python3 $R/d1_check/cmp.py $S/k1 $S/c22
# P20: the ghost total at each refresh (about 6 min with 8 ranks on one 4090)
mkdir -p $S/p20 && (cd $S/p20 && env $K8 SPH_MAXSTEP=30001 $S/sph_fixed $S/dam.dat | grep "halo refresh")
# P22: per-kernel times, forced 256-thread blocks against the per-launch choice
SPH_BLK=256 SPH_BENCH=1 $S/sph_fixed $S/dam.dat | grep bench
SPH_BENCH=1 $S/sph_fixed $S/dam.dat | grep bench
```

On KMG-XI, the same patch builds with `build.bat`. The comparisons worth making there, on an idle machine and with `SPH_D1PROF=1`, are K = 1 against K = 8 at 237k and 1.2 M, with and without `SPH_BLK=256`, plus the phase table and the exchange split, so that §3.1 can be replaced with measured numbers.

## 9. The reduced decks

[d1_scaling/narrow_deck.py](d1_scaling/narrow_deck.py) builds both decks from `input.dat`:

```bash
python3 review/Claude/d1_scaling/narrow_deck.py input.dat narrow.dat 0.15 0.65        # 79,424 particles
python3 review/Claude/d1_scaling/narrow_deck.py input.dat dam.dat    0.15 0.65 0.2    # 63,044, dam-break column
```

It keeps every particle with y < 0.145 m and translates the far y-wall and the plate's far edge (y ≥ 0.505 m) by −0.36 m, onto the first lattice row past the kept walls, so the tank stays closed. It drops walls and solids above z = 0.65 m (the fluid stays below 0.45 m). The optional last argument keeps only the fluid with x < 0.2 m. The block order fluid | solid | wall is preserved, and X0 moves with x. K = 1 with the fat and the lean A3 list is byte-identical on `dam.dat` (7/7 files, 3001 steps), as on the full deck.
