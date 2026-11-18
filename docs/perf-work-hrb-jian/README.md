# Performance work: fp64 fluid neighbour lists (P1) and slot-major solid lists (P2)

| | |
|---|---|
| Date | 2026-11-19 |
| Machine | RTX 5080 (`sm_120`, 84 SMs, 16 GB), CUDA 13.3, Windows 11, MSVC 2022 |
| Case | `input.dat`: 237 298 particles (101 250 fluid, 60 320 solid, 75 728 wall), `dt = 1e-5`, 11.10 M neighbour pairs at t=0 |
| Baseline | `d284042` (the build Claude reviewed), measured in the same session on the same machine |
| Companion | [`docs/work-2026-11-19-12-19`](../work-2026-11-19-12-19/README.md) — correctness side of the same change |

All numbers are milliseconds per time step unless stated otherwise, measured
with `SPH_BENCH=1` (20 iterations per kernel) and with 500-step runs for the
end-to-end figure.  The full step time comes from the `[time]` line, which
excludes result-file writing; the wall-clock figures below include it.

---

## 1. Headline

| | before | after | speed-up | wall clock, 500 steps |
|---|---:|---:|---:|---:|
| fp64, compute | 12.412 | **5.006** | **2.48x** | 2.50 s |
| fp64, total wall | — | — | — | **3.23 s** (incl. 0.17 s init + 0.56 s output) |
| fp32, compute | 2.434 | **1.634** | **1.49x** | 0.82 s |
| fp32, total wall | — | — | — | **1.47 s** (incl. 0.15 s init + 0.51 s output) |
| fp64 vs 32-thread CPU (211 ms/step) | 17x | **42x** | | |
| fp32 vs 32-thread CPU (211 ms/step) | 87x | **129x** | | |

Each row is measured back to back on the same machine, same deck; the "before"
column is the pre-change executable, preserved from the same session.  The
vs-CPU column uses the 32-thread reference run at 211 ms/step.

---

## 2. Where the time went (the reviewed build, fp64)

`SPH_BENCH=1`, 20 iterations each:

| kernel | ms | share |
|---|---:|---:|
| `pass1_fluid` (fused, A5) | 8.732 | 76 % |
| `pass2_solid` (CSR list) | 1.112 | 10 % |
| `pass1_solid` (CSR list) | 1.040 | 9 % |
| cell list | 0.126 | 1 % |
| `pass2_fluid` | 0.128 | 1 % |
| constitution | 0.051 | |
| prepare + 3 update kernels | 0.034 | |
| **full step (measured)** | **11.44** | |

Two observations drove the work:

1. **`pass1_fluid` is 76 % of the step and is FP64-pipe bound, not
   memory bound.**  In the fused loop each lane walks its own 27 cells, so the
   ~60-instruction FP64 body executes whenever *any* lane of the warp has a hit.
   Only about 15 % of the candidates are hits (104 accepted out of 687
   candidates per particle), so the FP64 units are busy while most lanes are
   masked off.  Adding warps, more ILP or fewer loads does not help a
   divergence problem.
2. **The solid lists are CSR and the access pattern is uncoalesced.**  5.37 M
   entries streamed per step, each warp reading 32 segments ~89 entries apart.

The pre-change dynamic path (no static list) measured `pass1_solid` 4.995 ms and
`pass2_solid` 5.881 ms, so the static list was already worth 4.7x — the problem
was how it was stored.

---

## 3. P1 — fp64 `pass1_fluid` in two phases

### Design

Phase 1 (`k_fluid_nbr_build`) records, for every fluid particle, the ids of its
accepted neighbours **in exactly the order the fused traversal visits them**;
phase 2 (`k_pass1_fluid_body<Real, true>`) runs the *same* body over that list.
Because the order is identical, the FP64 sum order is identical and the result
is bit-for-bit what the fused kernel produced (verified: identical `drhodt`,
`dvxdt`, `viscos`, `SurF` in a one-step dump, and a byte-identical
`result0.dat`).

* Storage: slot-major `fnbr[j*101250 + p]`, 192 slots per particle = **77.8 MB**.
* Both phases are one templated kernel with a `USE_LIST` flag, so the arithmetic
  is literally the same source; `SPH_FLUIDLIST=0` selects the fused version at
  run time for A/B measurement.

### Measured

| fp64 | ms |
|---|---:|
| fused traversal | 8.768 |
| neighbour-list build | 0.868 |
| body over the list | 2.673 |
| **total** | **3.541** |

The fused kernel and the list body visit exactly the same neighbours (10.52 M
visits per step, i.e. 103.9 per fluid particle counting fluid, solid and wall
partners alike), so the ratio below is the divergence effect alone:

| | ns per 1000 visits | visits/s |
|---|---:|---:|
| fused | 833 | 1.20 G/s |
| over the list | 254 | 3.93 G/s |

The hit rate (15 %) suggests up to ~6x if the body were the only cost;
3.3x is what the FP64 issue rate supports, and 2.5x is what survives after
paying the 0.868 ms for the list itself.

Result: `pass1_fluid` 8.77 -> 3.54 ms, and `SPH_FLUIDLIST=0` reproduces the
fused path bit for bit (same 1-step `drhodt`, `dvxdt`, `viscos`, `SurF`, and
byte-identical `result0.dat`), which is what makes the A/B above trustworthy.

### Occupancy of the list

Measured on this deck (`[bench] fluid neighbours:` line):

| | neighbours |
|---|---:|
| mean | 103.9 |
| min | 42 |
| max | 115 |
| overflow with 192 slots | 0 |

If a particle ever exceeded the width, `fcnt[p]` is set to -1 and **that
particle alone** falls back to the fused traversal, so the width is a
memory/speed knob and never a correctness limit.  Width 192 leaves 67 % headroom
over today's maximum; 128 would also fit but leaves less room for compression
(see §7) and costs the same in practice.

### Why fp32 does not use it

fp32 is traversal bound, not body bound: the fused body is 0.768 ms and the
traversal *alone* (`count_pairs`, no arithmetic) is 0.560 ms.  Splitting was
measured as a loss on the review's 4090 and again here, so the launch function
keeps the fused kernel for `Real = float` via `if constexpr`.

---

## 4. P2 — slot-major (ELL) static solid list

### Design

`entry j of solid particle k` now lives at `e = j*SOLNUM + k`, and the nine
tensor components live in nine separate arrays (component `c` at `c*sw + e`).
A warp that walks step `j` therefore reads 32 consecutive slots from each array
instead of 32 CSR segments ~89 entries apart.  The fill pass writes the entries
in the same visiting order as before, so the ELL rewrite itself does not reorder
any particle's sum.  (The byte-level difference against the pre-change build
comes from C1's pair set, not from this layout change -- see the work log §3.5.)

ELL pads every particle to the widest one: `width` is the maximum count, taken
from the count pass, and the fill pass writes directly into the padded layout
(no CSR arrays are kept).  Padding on this deck is 14.6 % (width 92, mean 80.3).

### Measured

| kernel | fp64 CSR | fp64 ELL | fp32 CSR | fp32 ELL |
|---|---:|---:|---:|---:|
| `pass1_solid` | 1.040 | **0.340** | 0.698 | **0.166** |
| `pass2_solid` | 1.112 | **0.624** | 0.751 | **0.296** |

Pair throughput (`pass1_solid`): 5.2 G pairs/s in the CSR build (5.37 M pairs in
1.040 ms) -> **14.2 G pairs/s** in the ELL build (4.84 M pairs in 0.340 ms); in
fp32, 7.7 -> 29 G/s.  `pass2_solid` moves less per pair (only `G` and `D`, plus a
3x3 matvec with the neighbour's `Pia`), so its gain is smaller.

Memory for the list (width 92):

| | fp64 | fp32 |
|---|---:|---:|
| slots | 5 549 440 | 5 549 440 |
| bytes per slot (9 Reals + 1 int) | 76 | 40 |
| **list size** | **402 MB** | **212 MB** |
| before (width 115, 5 371 593 entries) | 503 MB | 276 MB |

The width drop is the C2 filter: knife-edge pairs at `r = 3*hsml` whose kernel
weight is ~1e-40 (fp64) or ~1e-20 (fp32) are dropped geometrically.  The list
holds exactly the 4 844 544 pairs with `|X0_p - X0_q| < 3*hsml`, which is also
what a brute-force enumeration finds.

---

## 5. Current profile

### fp64, `SPH_BENCH=1`

| kernel | ms |
|---|---:|
| `pass1_fluid` (list build + body) | 3.529 |
| `pass2_solid` | 0.624 |
| `pass1_solid` | 0.340 |
| cell list | 0.129 |
| `pass2_fluid` | 0.131 |
| constitution | 0.051 |
| prepare + 3 update kernels | 0.025 |
| **sum of kernels** | **4.84** |
| **full step (measured)** | **5.01** |

The 0.17 ms difference between the kernel sum and the measured step is the
per-step launch and `cudaDeviceSynchronize()` overhead (see §6).

A/B rows still present in the binary: fused `pass1_fluid` 8.768, literal A5
11.123, pre-A3 solid path 5.002 / 5.903.

### fp32, `SPH_BENCH=1`

| kernel | ms |
|---|---:|
| `pass1_fluid` (fused) | 0.768 |
| `pass2_solid` | 0.296 |
| `pass1_solid` | 0.166 |
| cell list | 0.128 |
| `pass2_fluid` | 0.105 |
| constitution | 0.062 |
| prepare + 3 update kernels | 0.022 |
| **sum of kernels** | **1.55** |
| **full step (measured)** | **1.63** |

Pre-A3 comparison: 0.698 / 0.751.

### Phase timing (`SPH_PROFILE=1`, 100 steps)

| phase | fp64 | fp32 |
|---|---:|---:|
| prepare | 0.0 | 0.0 |
| cell list | 0.1 | 0.1 |
| pass 1 | 4.0 | 1.0 |
| constitution | 0.1 | 0.1 |
| pass 2 | 0.8 | 0.4 |
| update | 0.0 | 0.0 |
| **step** | **5.01** | **1.62** |

---

## 6. Where the time goes now

fp64 at the shipped 50-step output cadence (250 ms of compute per 50 steps):

| block | ms | share of wall |
|---|---:|---:|
| compute, 50 steps | 250 | 79 % |
| result file (1 x 20.5 MB) | 56 | **18 %** |
| launch / sync gap | ~9 | 3 % |

fp32 (82 ms of compute per 50 steps):

| block | ms | share of wall |
|---|---:|---:|
| compute, 50 steps | 82 | 60 % |
| result file | 51 | **37 %** |
| launch / sync gap | ~4 | 3 % |

Output costs 56 ms per file in fp64 and 51 ms in fp32, measured as
`wall - compute - init` over 500 steps (10 files: 0.56 s and 0.51 s).  A result
file is 20.5 MB, so the writer runs at ~370-400 MB/s — but it runs *between*
steps and the GPU waits for it.  That, not the compute, is what the next item
should attack.

---

## 7. What is left, in order of expected gain

| item | estimate | note |
|---|---|---|
| **P3 result writer** | −18 % wall (fp64), −38 % (fp32) | four columns (`mass`, `itype`, `X0`, `rho0`) never change: fetch once.  Pack the rest on the device, `cudaMemcpyAsync` into pinned memory on a side stream after an event, format on a background thread with double buffering. |
| **P4 launch overhead / CUDA graph** | up to −0.2 ms/step here, more on WDDM in general | one `cudaDeviceSynchronize()` per step remains (it is what `[time]` reports); 10 kernels + 3 memsets per step, and the kernel sum falls 0.08-0.17 ms short of the measured step.  Fuse the updates, fold the memsets into `k_prepare`, then capture a graph. |
| **P6 one evaluation per fluid pair** | up to −1.3 ms fp64 | `P*` and `dU` are symmetric, so one evaluation of a pair could serve both sides; the scatter breaks bit reproducibility, so it needs the deterministic per-pair-buffer + reverse-CSR variant (~0.7 ms of traffic). |
| **Verlet list with a skin** | not measured | fp32 especially: a 1 mm skin lasts ~16 steps at 3 m/s and `dt = 1e-5`, cutting candidates from 687 to ~125 per particle.  The exact distance test would stay, so only the summation order changes. |
| **Compression / list width** | small | shrinking the fluid list from 192 to 128 slots saves 26 MB and nothing measurable; below 128 it needs per-block widths. |
| C7 `k_cell_sort` | none for this deck | insertion sort per cell, O(n²) only when the cell clamp collects many particles into one boundary cell. |

---

## 8. Reproducing

```bat
:: per-kernel profile (fp64, then fp32)
cd cuda && build.bat && cd ..
set SPH_BENCH=1
cuda\build\sph_cuda.exe
cuda\build\sph_cuda.exe --fp32

:: end-to-end, 500 steps
set SPH_MAXSTEP=500
cuda\build\sph_cuda.exe
cuda\build\sph_cuda.exe --fp32

:: phase timing
set SPH_PROFILE=1
set SPH_MAXSTEP=100
cuda\build\sph_cuda.exe
```

The A/B switches used above: `SPH_FLUIDLIST=0` (fused fp64 `pass1_fluid`),
`SPH_SOLDDYN=1` (walk the grid in the solids instead of the static list),
`SPH_LITERALFLUID=1` (literal A5 transcription).  The bench mode prints all of
them in one run.

## 9. Caveats

* One GPU, one deck, one driver.  The absolute numbers moved by a few percent
  between runs (12.4 vs 11.4 ms/step for the pre-change build, depending on
  whether the run was 1, 500 or 700 steps — clock and pair-count drift), so only
  the back-to-back comparisons in §1 and §4 should be read as ratios.
* The 500-step "before" run was made with the pre-change executable preserved
  from the same session, not with a rebuilt tree.
* `[time]` excludes result writing; wall-clock figures include it.
