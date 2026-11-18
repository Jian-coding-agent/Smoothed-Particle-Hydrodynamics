# Performance re-check on the RTX 4090: `98fcf58` vs `d284042`

| | |
|---|---|
| Date | 2026-11-19 |
| Machine | RTX 4090 (`sm_89`, 128 SMs, 24 GB), driver 580.173.02, Ubuntu 22.04.5, i9-14900KF (32 threads), 62.5 GiB RAM, host `alienware` |
| Toolchain | CUDA 12.2 (`nvcc` V12.2.91), GCC 11.4 as host compiler, CMake bypassed |
| Case | `input.dat`, 237 298 particles (101 250 fluid, 60 320 solid, 75 728 wall), `dt = 1e-5` |
| Revisions | `d284042` (base, rebuilt here) vs `98fcf58` (HEAD) |
| Companion | [`docs/perf-work-hrb-jian`](perf-work-hrb-jian/README.md) — the same change measured on the RTX 5080 / Windows; [`docs/work-2026-11-19-12-19`](work-2026-11-19-12-19/README.md) — correctness side |

This is the independent re-measurement of the P1/P2 change on different hardware
and a different toolchain: different GPU architecture (sm_89 vs sm_120), Linux
instead of Windows, CUDA 12.2 instead of 13.3, GCC 11.4 instead of MSVC, and the
baseline rebuilt from `d284042` in this tree rather than a preserved executable.

The GPU is shared with another user's resident process (7.9 GiB allocated, idle
during these runs). Both revisions were built here and run back-to-back and
interleaved, six rounds each, so the ratios are what should be read; absolute
milliseconds include whatever that tenant did.

## 1. Headline

Step time is the `[time]` line (500 steps, result writing excluded); wall is the
whole process including 10 × 20.5 MB result files.

| | base `d284042` | HEAD `98fcf58` | speed-up |
|---|---:|---:|---:|
| fp64 step, ms | 9.09 (9.04–9.18) | **4.14 (4.11–4.20)** | **2.20x** |
| fp32 step, ms | 2.19 (2.18–2.23) | **1.28 (1.26–1.30)** | **1.73x** |
| fp64 wall, 500 steps, s | 5.38 | 2.89 | 1.86x |
| fp32 wall, 500 steps, s | 1.85 | 1.42 | 1.30x |

3000-step fp64 run: **4.343 ms/step** (13.028 s of compute), no NaN/Inf.

Against the CPU reference built from the same tree on this machine (double
precision, 16 threads pinned to the P-cores, 84.8 ms/step): **20.5x** in fp64,
and the fp32 GPU path is 66x the double-precision CPU. Details in §7.

## 2. Which change bought what

Separated by running the HEAD binary with `SPH_FLUIDLIST=0` (fused fp64
`pass1_fluid`, ELL solid lists kept), 500 steps, three runs (spread 0.5 %):

| configuration | fp64 step, ms | gain |
|---|---:|---:|
| base: fused fluid + CSR solids | 9.09 | — |
| HEAD, `SPH_FLUIDLIST=0`: fused fluid + ELL solids | 7.59 | **1.20x** (P2 + C1/C2) |
| HEAD: two-phase fluid + ELL solids | 4.14 | **1.83x** (P1) |

1.20 × 1.83 = 2.20 — the two changes are independent and multiply.

## 3. Per-kernel, `SPH_BENCH=1` (20 iterations each)

Best of three runs per row; the small kernels move up to ±70 % between runs on
this shared GPU, so treat the rows as indicative and §1/§2 as the results.

fp64:

| kernel | base | HEAD | ratio |
|---|---:|---:|---:|
| `pass1_fluid`, fused path | 5.985 | **5.984** | 1.00 (same kernel) |
| `pass1_fluid`, list path | — | **3.019** (build 0.781 + body 2.010) | 1.98x vs fused |
| `pass1_solid` | 1.122 | **0.305** | 3.68x |
| `pass2_solid` | 1.126 | **0.510** | 2.21x |
| `pass2_fluid` | 0.091 | 0.092 | — |
| cell list | 0.145 | 0.156 | — |
| constitution | 0.041 | 0.041 | — |
| update ×3 + prepare | 0.024 | 0.024 | — |

The two builds' fused `pass1_fluid` agree to 0.01 % (5.985 / 5.984), which is
what makes the split meaningful: the fp64 gain is entirely the list path plus the
solid layout, not a changed arithmetic path.

fp32 (very stable, ±0.5 %):

| kernel | base | HEAD | ratio |
|---|---:|---:|---:|
| `pass1_fluid` (fused, by design) | 0.435 | 0.435 | 1.00 |
| `pass1_solid` | 0.604 | **0.161** | 3.75x |
| `pass2_solid` | 0.666 | **0.267** | 2.49x |
| pass2_fluid / cell | 0.088 / 0.144 | 0.088 / 0.144 | — |

fp32 keeps the fused `pass1_fluid` as intended and is unchanged; the whole fp32
gain comes from the ELL rewrite, and it is larger here than on the 5080
(1.73x vs 1.49x).

Throughput (4 844 544 solid pairs; 10.52 M fluid neighbour visits):

| kernel | base | HEAD |
|---|---:|---:|
| `pass1_solid`, fp64 | 4.32 G pairs/s | **15.9 G pairs/s** |
| `pass1_solid`, fp32 | 8.02 | **30.1** |
| `pass2_solid`, fp64 | 4.30 | **9.50** |
| `pass2_solid`, fp32 | 7.27 | **18.1** |
| `pass1_fluid`, fp64 | 1.76 G visits/s (fused) | **3.48** (list, incl. build) |

`pass1_solid` reads `sQ + sB(3) + sG(3)` = 52 B per pair in fp64, so 0.305 ms is
754 GB/s of the 1008 GB/s peak: that kernel is now bandwidth bound, which is the
right place for it. `pass2_solid` reaches ~490 GB/s because it also pulls the
partner's `Pia`, kept in L2 by the shared partner sets inside a warp.

## 4. Where the wall clock goes now

fp64, 500 steps: compute 2.065 s + init 0.123 s + **output 0.702 s** = 2.89 s.
Output is **24 % of wall, 70 ms per 20.5 MB file (~293 MB/s)**, serialised
between steps.
fp32, 500 steps: compute 0.640 s + init 0.110 s + **output 0.665 s** = 1.42 s —
output is **47 % of wall, 66 ms/file**, up from 37 % on the 5080 because the
compute got cheaper and the writer did not change.

That is why the end-to-end speed-up (1.86x / 1.30x) sits well below the step
speed-up (2.20x / 1.73x). **P3 is the largest remaining item on this machine,
and in fp32 it is nearly half of the wall clock.**

## 5. Cross-check against the 5080 numbers

| | 5080 / CUDA 13.3 / MSVC | 4090 / CUDA 12.2 / GCC 11 | 4090 advantage |
|---|---:|---:|---:|
| fp64 step, base | 12.412 | 9.09 | 1.37x |
| fp64 step, HEAD | 5.006 | 4.14 | 1.21x |
| fp32 step, base | 2.434 | 2.19 | 1.11x |
| fp32 step, HEAD | 1.634 | 1.28 | 1.28x |
| fp64 gain | 2.48x | 2.20x | |
| fp32 gain | 1.49x | 1.73x | |

The fp64 gain is smaller here: the fused baseline was already faster on the 4090,
and the list build costs 0.78–1.26 ms of the 3.02 ms `pass1_fluid` (26–40 %,
against 25 % on the 5080), so the split has less to recover. The fp32 gain is
larger because the ELL rewrite pays off more on this memory system.

P4 (launch overhead) is worth ~nothing on Linux: HEAD's kernel sum (4.15 ms) is
already equal to its measured step (4.13–4.20 ms). The 0.17 ms/step gap recorded
on Windows 11 does not reproduce here — it is a WDDM artefact. The fused-path
configurations do show a ~0.5 ms gap between bench sum and step, so the 20-iteration
bench reads the fused kernel ~6 % cheap; the step-level numbers are the reliable ones.

## 6. Correctness re-check on this machine

| check | result |
|---|---|
| builds on CUDA 12.2 + GCC 11 | clean, 3.6 s, `sm_89` only, no source edits needed |
| `check_rigid_translation.py`, HEAD | **PASS** — 4 844 544 pairs at 0 and at +25 mm, Δtrace(Pia) max 5.81e-8 Pa |
| same script, base build | **FAIL** as documented — 5 371 593 → 5 068 175 pairs, 5.34e6 Pa |
| fp64 vs CPU reference at t=500 | x, rho, mass, itype, X0, rho0 identical; p 6.9e-8 Pa; v ≤ 1e-9 m/s |
| HEAD vs base, t=500 | x, p, rho identical; v ≤ 1e-9 m/s |
| HEAD vs base, t=2950 | x, rho identical; p ≤ 1e-3 Pa; v ≤ 1e-9 m/s — no trajectory divergence |
| fp32 vs fp64 at t=500 | x 1e-6 m, p 0.164 Pa, v 1e-5 m/s, rho 0.01 kg/m³ |
| determinism | two 500-step runs byte-identical, both precisions (10 files each) |
| `compute-sanitizer --tool memcheck` | 0 errors, fp64 and fp32, 3 steps |
| 3000 steps | no NaN/Inf; p max 34.5 kPa, median 1987 Pa, rho 960–1015 kg/m³ |
| peak device memory | 1050 MiB fp64 (base 966 MiB): ELL 402.2 MB fp64 / 211.7 MB fp32 + 77.8 MB fluid list |

Not done here: racecheck (only memcheck), the fp32 `A = I` refusal path, and the
two bad-deck input-validation cases; those are covered by the work log on the
5080 and no code changed since.

## 7. CPU reference baseline (same machine, same deck)

Built from the same tree with the project's own CMake (`-O3 -ffp-contract=off
-fopenmp`, GCC 12.3, 0.9 s build), 500 steps, `SPH_THREADS` set and threads
pinned with `taskset`. Phase split from the instrumented copy
(`_local/1_timed.cpp`, identical code plus `std::chrono` around the three
sections of the loop):

| threads | neighbour search | physics | update | compute total | ms/step | output | wall, 500 steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 (P-cores 0-15) | 9.41 s | 32.60 s | 0.37 s | 42.38 s | **84.8** | 5.94 s (540 ms/file) | 48.3 s |
| 32 (all) | 6.07 s | 32.00 s | 0.56 s | 38.63 s | **77.3** | 6.00 s (546 ms/file) | 44.6 s |

- `continue_density` (physics) is **77 %** of the CPU time and does not scale at
  all from 16 to 32 threads (32.60 → 32.00 s) because the second 16 logical CPUs
  are E-cores; the neighbour search does scale (9.41 → 6.07 s), so 16 → 32 buys
  only 1.10x overall.
- GPU vs CPU, same deck, same 500 steps: **20.5x** fp64 against 16 threads,
  18.7x against 32 threads. The fp32 GPU path is 66x the double-precision CPU.
- The CPU writer costs **540 ms per result file** against 70 ms on the GPU: every
  field goes through `ostream <<` (237 298 rows × 14 fields), while the GPU formats
  in bulk with `to_chars` (A2). Output is 12 % of the CPU wall clock and 47 % of
  the GPU fp32 wall clock.
- The 5080 log now quotes **163 ms/step of wall clock** on its 9950X
  (131 ms/step of compute, `docs/perf-work-hrb-jian` §10). The same source on
  this machine is 77–85 ms/step of compute, 89–97 ms/step of wall, so the
  4090/CPU ratio here is 20.5x rather than their 33x. The CPU side is the
  difference, not the GPU: a GCC `-O3` build against MSVC `/fp:precise` is worth
  about 1.47x on this solver (measured here as `-O3` vs `-O2`), which is the size
  of the 131 → 85 ms/step gap.
- R1 is visible in the pair count: the CPU now reports 11 102 707 pairs at t=0,
  identical to the GPU (an FMA-contracting build gave 11 103 787).
- **The CPU reader has no error check.** A missing or unreadable `input.dat`
  leaves every array at zero, which puts all 237 298 particles in one cell, makes
  the neighbour search O(n²) (18 s/step), overflows the pair counter to a negative
  value, and still exits 0. The CUDA port's C4 validation is worth mirroring in
  `cpu/1.cpp`; today a typo in the working directory looks like a 200x slowdown.

## 8. Reproducing

The CMake project still requires CMake ≥ 3.24 and hard-codes MSVC flags
(`/W3 /wd4819`), so on this box the build is a direct `nvcc` call, out of tree:

```bash
# all commands run from the repository root
nvcc -ccbin g++-11 -O3 -std=c++17 --fmad=true -arch=sm_89 -Xcompiler -O3 \
     -DSPH_HAVE_FP64=1 -DSPH_HAVE_FP32=1 -Icuda/src \
     cuda/src/sph_kernels.cu cuda/src/sph_host.cu -o _local/bin/sph_new

git archive d284042 | tar -x -C _local/base_d284042   # baseline tree, same nvcc line

python3 cuda/tools/check_rigid_translation.py --exe _local/bin/sph_new --input input.dat

# run directories need the deck a known number of levels up: check the link resolves
cd _local/bench_new && ln -sfn ../../input.dat input.dat
SPH_BENCH=1 ../bin/sph_new --fp64                                    # per-kernel
SPH_MAXSTEP=500 /usr/bin/time -f "WALL %e" ../bin/sph_new --fp64     # step + wall
SPH_FLUIDLIST=0 SPH_MAXSTEP=500 ../bin/sph_new --fp64                # fused A/B
cd ../..

# CPU reference through the project's own CMake (this one accepts CMake 3.22)
cmake -S cpu -B _local/cpu_build_new -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _local/cpu_build_new
g++ -O3 -ffp-contract=off -fopenmp -std=c++17 _local/1_timed.cpp -o _local/cpu_build_new/SPH_timed
cd _local/cpuperf && ln -sfn ../../input.dat input.dat
taskset -c 0-15 env SPH_THREADS=16 SPH_MAXSTEP=500 ../cpu_build_new/SPH_timed
```

`_local/1_timed.cpp` is the CPU source plus `std::chrono` around the three
sections of the time loop; it is a measurement aid, not part of the tree.

Artifacts: `_local/bin/sph_new`, `_local/bin/sph_base`,
`_local/perf/r1_*` and `_local/perf/rr_*` (500-step runs), `_local/perf/foil`
(`SPH_FLUIDLIST=0`), `_local/perf/cmp501*` (numeric), `_local/perf/p_*` (phase
profile), `_local/perf/long3000`, `_local/perf/san`, `_local/cpu_build_new`,
`_local/cpuperf` (CPU runs).
