# CUDA Port on RTX 4090 — Test Report

Date: 2026-11-18
Machine: Ubuntu 22.04.5, i9-14900KF, 62.5 GiB RAM, RTX 4090 24 GB (driver 580.173.02)
Target: `cuda/` (sph_cuda, fp64/fp32 paths)

## 0. Summary

- **Builds, runs, and is reproducible.** fp64 agrees with the CPU reference to printed precision over 3000 steps.
- **501 steps: fp64 8.78 ms/step, fp32 2.11 ms/step** (kernel time, excluding output).
- Against the local CPU build with 16 threads (73.2 ms/step): **8.3x / 34.6x**. The README's 17x / 87x uses a 211 ms/step CPU baseline that is not reproducible on this machine.
- Against the README's RTX 5080 numbers (12.42 / 2.43 ms): the 4090 is 1.41x faster in fp64 and 1.15x in fp32 (different GPU, CUDA 13 vs 12.2, MSVC vs GCC — not a controlled comparison).
- No repository source was modified. The repo's build assumptions (CUDA 13.x + MSVC + CMake ≥ 3.24) do not hold here and had to be bypassed.
- dsh's earlier claim that CUDA could not run was a limitation of its sandbox (no `/dev/nvidia*` inside the sandbox), not of the machine. The GPU was idle during these runs (43 MiB used, 0% util).

## 1. Build

`cuda/CMakeLists.txt` cannot be used as-is:

- `cmake_minimum_required(VERSION 3.24)`; this machine has CMake 3.22.1.
- `target_compile_options` hard-codes MSVC flags `/W3 /wd4819`; GCC rejects them.
- `CMAKE_CUDA_ARCHITECTURES native` also requires CMake 3.24+.

CUDA 12.2's nvcc also rejects GCC 12.3 (`unsupported GNU version`), so GCC 11.4 was used as the host compiler and nvcc was invoked directly:

```bash
cd /data/jian/Smoothed-Particle-Hydrodynamics
mkdir -p _local/cuda_build
nvcc -ccbin g++-11 -O3 -std=c++17 --fmad=true -arch=sm_89 \
     -Xcompiler -O3 \
     -DSPH_HAVE_FP64=1 -DSPH_HAVE_FP32=1 \
     -Icuda/src cuda/src/sph_kernels.cu cuda/src/sph_host.cu \
     -o _local/cuda_build/sph_cuda
```

- Compile time 3.6 s; binary 1.66 MB; SASS contains only `sm_89`.
- The run directory must contain `input.dat`:

```bash
mkdir -p _local/run && cd _local/run
ln -s ../../input.dat input.dat
SPH_MAXSTEP=501 SPH_PROFILE=1 ../cuda_build/sph_cuda --fp64
```

## 2. Performance

### 2.1 Overall

| Run | Steps | Kernel time | ms/step | Wall |
|---|---:|---:|---:|---:|
| fp64 | 501 | 4.401 s | **8.784** | ~5.3 s |
| fp32 | 501 | 1.059 s | **2.113** | ~1.8 s |
| fp64 | 3000 | 27.755 s | 9.252 | ~32 s |
| fp32 | 3000 | 6.488 s | 2.163 | ~11 s |

CPU reference (same source, 16 threads, `_local/build/SPH`):

| Run | Steps | Wall | ms/step (with output) | ms/step (without output) |
|---|---:|---:|---:|---:|
| CPU 16T | 501 | 43.25 s | 86.3 | 73.2 |
| CPU 16T | 3000 | 260 s | 86.7 | — |

Speedups (kernel time, excluding output): fp64 73.2 / 8.78 = **8.3x**; fp32 73.2 / 2.11 = **34.6x**.

### 2.2 Per-kernel (`SPH_BENCH=1`, ms/iter)

| Kernel | fp64 | fp32 |
|---|---:|---:|
| pass1_fluid | **6.473** | **0.471** |
| pass1_solid | 1.116 | 0.606 |
| pass2_solid | 1.137 | 0.674 |
| pass2_fluid | 0.089 | 0.096 |
| cell (hist+scan+scatter) | 0.155 | 0.156 |
| count_pairs (traversal) | 1.442 | 0.467 |
| constitution | 0.040 | 0.046 |
| update (three kernels) | 0.018 | 0.016 |
| prepare | 0.006 | 0.003 |

In fp64, 74% of the time is in `pass1_fluid`. In fp32 the three solid/fluid kernels are roughly balanced (0.47–0.67 ms), so there is no single hotspot.

### 2.3 GPU state

Sampled during the 501-step fp64 run:

- SM utilisation **100%**, core clock 2745 MHz, temperature 39 °C;
- peak board power **95.6 W** (4090 limit is 450 W).

This is not a power or clock limit; the kernels are memory/latency bound. That also explains why fp64 is only 4.2x slower than fp32 rather than the 64x of the FP64 throughput ratio.

### 2.4 Memory and startup

- Peak GPU memory **966 MiB** (`nvidia-smi` reports 1009 MiB including the 43 MiB idle baseline).
- Host RSS about 180 MB.
- Initialisation (read `input.dat`, build the static solid pair list, NorVector/GraCorrec): 0.137 s fp64, 0.108 s fp32.
- Static solid pair list: 5.37M pairs, 389 MB.

## 3. Compiler sensitivity (CPU reference)

GCC variants were built from the same `cpu/1.cpp` and run for 201 steps with 16 threads pinned to the P-cores (CPUs 0–15). Wall time includes initialisation and 5 output files, so compare the variants to each other rather than to the 73.2 ms/step kernel-only figure.

| Variant | Flags | Wall (s) | Relative |
|---|---|---:|---:|
| O3 | `-O3` | **20.06** | 1.00 |
| O3native | `-O3 -march=native` | 20.06 | 1.00 |
| O3v3 | `-O3 -march=x86-64-v3` | 19.97 | 0.99 |
| O3novec | `-O3 -fno-tree-vectorize` | 20.48 | 1.02 |
| O3v3nocontract | `-O3 -march=x86-64-v3 -ffp-contract=off` | 20.52 | 1.02 |
| O2 | `-O2` | 29.50 | 1.47 |
| O2novec | `-O2 -fno-tree-vectorize` | 29.82 | 1.49 |

Findings:

- **`-O3` vs `-O2` is the dominant compiler effect: about 1.47x.**
- Auto-vectorisation and FMA contraction contribute at most 2–3% on this code. `-march=native` does not help.
- MSVC `/O2` is broadly comparable in aggressiveness to GCC `-O2`, so a Windows/MSVC build at "100+ ms/step" versus GCC `-O3` at ~73–80 ms/step is consistent with the measurements here. This is a plausible and sufficient explanation.
- Compiler flags alone do **not** explain the README's 211 ms/step: even `-O2` here is ~125 ms/step in loop time (29.5 s for 201 steps minus ~4.5 s of init/output), and a 1.47x penalty on the GCC `-O3` figure gives ~110 ms/step. The remaining gap to 211 ms is therefore likely a different source revision (the code contains later `[OPT-R1]`, `[OPT-OMP]`, `[OPT-R6]` optimisations), a 32-thread SMT configuration, or both. It is not a pure compiler effect.

The 14900KF also differs from the 9950X in topology: 8 P-cores + 16 E-cores versus 16 full cores. dsh measured 16 threads faster than 32 on this machine because the E-cores slow `continue_density` down, so thread count and affinity matter as much as the compiler.

## 4. Correctness

### 4.1 CPU reference comparison (all columns, 6 significant digits)

| Time | Files | Result |
|---|---|---|
| t=500 | `runA/result10` vs `gpu501_fp64/result10` | x/y/z, rho, mass, itype, X0/rho0 **identical**; p 6.86e-8; vx 1.0e-9; vy/vz 1.0e-10 |
| t=600 (in contact) | `cpu650/result12` vs `gpu650/result12` | x/y/z, rho etc. identical; p 6.86e-8; v 1.0e-10 |
| t=2950 | `cpu3000/result59` vs `gpu3000_fp64/result59` | x/y/z, rho etc. identical; p **1.0e-3**; vx 1.0e-9; vy 1.0e-10; vz 1.0e-9 |

The 1e-3 pressure difference at 3000 steps is within the CPU's own rerun-to-rerun noise (1e-2), and x/rho are identical at printed precision.

### 4.2 fp32 accuracy (t=500, relative to fp64)

| Quantity | Max abs. difference | README claim |
|---|---:|---:|
| x/y/z | 1.0e-6 m | 1e-6 |
| p | 0.164 Pa | 0.17 |
| vx | 1.0e-5 m/s | 1e-5 |
| vy | 2.2e-6 | — |
| vz | 1.4e-6 | — |
| rho | 1.0e-2 kg/m³ | — |

This matches the README's fp32 accuracy claims. At 3000 steps the fp32 pair count is 11,273,246 versus 11,169,276 for fp64 (0.9% difference), because float positions fall on different sides of the `r=3h` membership test. This is expected behaviour, not an error.

### 4.3 Reproducibility and checks

- **Determinism:** two 100-step fp64 runs produce byte-identical `result0.dat` and `result1.dat`. This is stronger than the CPU build, whose OpenMP `dynamic` scheduling changes the p last digits by up to 1e-2.
- **compute-sanitizer:** `memcheck` over 2 steps reports `ERROR SUMMARY: 0 errors`.
- **A/B switches:** `SPH_SOLDDYN=1` (disable the static solid pair list) differs from the default by p 2.16e-8 and vx 1e-9; `SPH_LITERALFLUID=1` differs only by vy 1e-15. Both are at the last printed digit, consistent with the README's "identical to printed precision".

## 5. Long run and contact

3000-step fp64 run, no NaN/Inf; at t=2950:

| Quantity | Value |
|---|---|
| Fluid p | 0 – 34,472 Pa, median 1,987 Pa |
| Fluid velocity | max 3.01 m/s |
| Density | 960 – 1015 kg/m³ |
| Fluid x front | 0.4947 m (plate at 0.51 m, inside the support radius) |
| Max solid displacement | 0.101 mm |

Pressure rises over time: p max 1.74 kPa at t=1000, 15.4 kPa at t=1500, 34.5 kPa at t=2950; the median rises from 154 Pa to 1987 Pa. The CPU build gives the same values at the same times (p difference 1e-3), so this is not introduced by the GPU port; it is the original algorithm's behaviour in this closed or near-closed domain (water-hammer-like compression).

Caveat: the "double force on fluid–solid pairs" that dsh flagged is fully replicated in CUDA — `k_pass2_fluid` handles fluid particles (p < NTOTAL) and accumulates SForce from solid neighbours, while `k_pass2_solid` handles the solid side. CPU–GPU agreement therefore proves the port is faithful, not that this coupling path is physically correct. Whether part of the 3000-step pressure rise comes from that duplicate term requires a separate "mutually exclusive force" control run.

## 6. Differences from the README / recommendations

1. **Build:** the repo hard-codes MSVC + CUDA 13 + CMake 3.24. Make `/W3 /wd4819` compiler-conditional, fall back from `native` for older CMake, and document a Linux command; otherwise "out of the box" does not hold on non-Windows machines.
2. **Speedup baseline:** the README's 17x / 87x is against a 211 ms/step CPU baseline. The current source on this machine is 73.2 ms/step with 16 threads, giving 8.3x / 34.6x. Re-measure the current CPU source before quoting a speedup.
3. **fp64 hotspot:** `pass1_fluid` is 74% of the time, the GPU is at 100% SM utilisation but only 95 W, so the limit is memory/latency. Further work should target the fluid pass1 data layout and per-thread work rather than adding parallelism.
4. **Fluid–solid coupling:** confirm whether the pass1/pass2 double application is intentional. If it is a bug, CPU and CUDA must be fixed together, with a 600+ step regression case that actually contains fluid–solid pairs.
5. **Long-run pressure:** pressure reaches 34 kPa and density is compressed by 1.5% at 3000 steps. Add a pressure/density monitor, or confirm that this is the intended closed-domain water hammer.

## 7. Artifacts

```
_local/cuda_build/sph_cuda              build output
_local/gpu501_fp64/  gpu501_fp32/       501-step benchmarks (result0..10)
_local/gpu3000_fp64/ gpu3000_fp32/      3000-step long runs (result0..59)
_local/gpu650/                          650-step contact case (compared with cpu650)
_local/cpu3000/                         CPU 3000-step reference
_local/gpu_det1/ gpu_det2/              determinism test
_local/gpu_ab_dyn/ gpu_ab_literal/      A/B switch tests
_local/cpu_variants/                    CPU compiler-flag variants
_local/var_run/                         201-step variant runs
```

Run logs are the `*.log` files in each directory; rerun with `SPH_PROFILE=1` to reproduce the per-phase timings.
