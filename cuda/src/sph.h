// ============================================================================
//  sph.h -- shared declarations for the CUDA port of the SPH solver (1.cpp)
//
//  The whole solver is templated on the scalar type `Real`:
//      Real = double  -> fp64 path (default, validates against the CPU build)
//      Real = float   -> fp32 path (fast, lower accuracy - see docs section 10)
//
//  Both instantiations are compiled into the executable by default; which one
//  runs is selected at start-up:
//      sph_cuda.exe --fp32        or      set SPH_PRECISION=fp32
//      sph_cuda.exe --fp64        or      set SPH_PRECISION=fp64   (default)
//  CMake can trim the build with -DSPH_ENABLE_FP64=OFF / -DSPH_ENABLE_FP32=OFF.
//
//  Design rules:
//   * arrays are component-major SoA, exactly like `double x[3][num]`;
//   * EVERY floating point literal in the physics is written Real(literal),
//     otherwise C++ promotion rules would silently turn parts of the fp32
//     path into fp64 (which on this GPU is 67x slower and defeats the point);
//   * the file is pure ASCII on purpose (cp936 locale -> C4819 noise).
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---------------------------------------------------------------------------
// Problem size (from 1.cpp: num, ntotal, SolNum)
//   [0, NTOTAL)        fluid       itype =  2
//   [NTOTAL, NALL)     solid       itype = -20
//   [NALL, NUM)        wall        itype = -2
// ---------------------------------------------------------------------------
#define SPH_NUM    237298
#define SPH_NTOTAL 101250
#define SPH_SOLNUM 60320
#define SPH_NALL   (SPH_NTOTAL + SPH_SOLNUM)
#define SPH_NWALL  (SPH_NUM - SPH_NALL)

// ---------------------------------------------------------------------------
// Physical / scheme constants (1.cpp main())
// ---------------------------------------------------------------------------
#define SPH_HSML   0.01
#define SPH_DT     0.00001
#define SPH_C0     48.0
#define SPH_LAMD   13333333.0
#define SPH_MUYOU  2222222.0
#define SPH_MAXV   30.0
#define SPH_PI_    3.141592653589793

// ---------------------------------------------------------------------------
// Neighbour grid: fixed domain, cell edge = 3*hsml
// nx = floor(0.8/0.03)+1 = 27, ny = 27, nz = floor(1.3/0.03)+1 = 44
// ---------------------------------------------------------------------------
#define SPH_GX0   (-0.10)
#define SPH_GY0   (-0.10)
#define SPH_GZ0   (-0.10)
#define SPH_GX1   ( 0.70)
#define SPH_GY1   ( 0.70)
#define SPH_GZ1   ( 1.20)
#define SPH_CS    (3.0 * SPH_HSML)
#define SPH_NX    27
#define SPH_NY    27
#define SPH_NZ    44
#define SPH_NCELL (SPH_NX * SPH_NY * SPH_NZ)
#define SPH_PIA_STRIDE 12

// ---------------------------------------------------------------------------
// which instantiations exist (CMake usually defines both)
// ---------------------------------------------------------------------------
#ifndef SPH_HAVE_FP64
#define SPH_HAVE_FP64 1
#endif
#ifndef SPH_HAVE_FP32
#define SPH_HAVE_FP32 1
#endif

// ---------------------------------------------------------------------------
// Device arrays.  Tensors are flat, component-major: M[(r*3+c)*SPH_NUM + i]
// matching 1.cpp's nine arrays A11..A33 / F11..F33 / ...
// ---------------------------------------------------------------------------
template <class Real>
struct SphPtrsT
{
    Real* x;        // [3*NUM]
    Real* vx;       // [3*NUM]
    Real* X0;       // [3*NUM]
    Real* vecn;     // [3*NUM]
    Real* PIJW;     // [3*NUM]
    Real* SForce;   // [3*NUM]
    Real* dvxdt;    // [3*NUM]
    Real* viscos;   // [3*NUM]

    Real* rho;
    Real* irho;     // 1/rho, refreshed in k_prepare / k_constitution
    Real* rho0;
    Real* mass;
    Real* p;
    Real* itype;
    Real* SurF;
    Real* SurFD;
    Real* drhodt;

    Real* A;        // [9*NUM]
    Real* F;        // [9*NUM]
    Real* DFdt;     // [9*NUM]
    // Pia is stored array-of-structs: 12 Reals per solid particle (9 used + 3
    // padding) at index (p - NTOTAL).  12*sizeof(Real) is a multiple of 16, so a
    // neighbour's 3x3 stress can be fetched with 16-byte loads instead of nine
    // scattered 8-byte ones (pass2_solid is load bound).
    Real* Pia;      // [12*SOLNUM]

    // static per-solid pack: {X0[3], A[9], rho0, mass} at index (q - NTOTAL)
    // built once; turns 14 scattered loads per solid neighbour into 1-2 lines
    Real* solidPack; // [14*SOLNUM]

    // ------------------------------------------------------------------
    // A3: static solid pair list (see docs section 12).  The solid kernels
    // evaluate the kernel gradient at the *reference* positions, which never
    // change, so their neighbour set and per-pair weights are constants.  The
    // list is built once after GraCorrec in CSR form:
    //     solid particle k = p - NTOTAL owns entries sStart[k] .. sStart[k+1]
    //     entry a: partner sQ[a], and 3 Reals each at 3*a in sG / sB / sD
    // ------------------------------------------------------------------
    int   npair;
    int*  sStart;   // [SOLNUM+1]
    int*  sQ;       // [npair]     partner particle index
    Real* sG;       // [3*npair]   w0 * gradW(X0_q - X0_p)   -> DFdt, PIJW
    Real* sB;       // [3*npair]   w0 * (A_p gradW)          -> F
    Real* sD;       // [3*npair]   w0 * (A_q (-gradW))       -> PIJW

    int* cellStart; // [NCELL+1]
    int* cellCur;   // [NCELL]
    int* cellSolid; // [NCELL] number of solid particles per cell
    int* cellFluid; // [NCELL] number of fluid particles per cell
    int* sorted;    // [NUM]

    // A4: float copy of the live positions, refreshed by k_update_pos (and once
    // after load_input).  The neighbour test uses it as a cheap pre-filter in
    // front of the exact Real test; see sph_neigh() for the margin argument.
    float* xf;      // [3*NUM]

    // fp32 accuracy support -- allocated for both precisions, used only when
    // Real = float (the fp64 path is untouched and stays bit-identical):
    Real* u;        // [3*NUM] displacement x - X0 of the solid-type particles
    Real* xc;       // [3*NUM] running compensation of the position/displacement sum
    Real* rhoc;     // [NUM]   running compensation of the density sum
};

// 2-wide vector type used to pull the Pia block with 16-byte loads
template <class Real> struct SphVec2T;
template <> struct SphVec2T<double> { typedef double2 type; };
template <> struct SphVec2T<float>  { typedef float2  type; };

typedef SphPtrsT<double> SphPtrsD;
typedef SphPtrsT<float>  SphPtrsF;

// ---------------------------------------------------------------------------
// Rounded add/multiply that the compiler must NOT fuse into an FMA.
// The neighbour-list membership test lands exactly on r = 3*hsml for a large
// part of the lattice, so the test is knife-edge; keeping the rounding
// explicit makes the fp64 pair set identical to direct_find().
// ---------------------------------------------------------------------------
__device__ __forceinline__ double sph_add_rn(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ float  sph_add_rn(float a, float b)   { return __fadd_rn(a, b); }
__device__ __forceinline__ double sph_mul_rn(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ float  sph_mul_rn(float a, float b)   { return __fmul_rn(a, b); }

// ---------------------------------------------------------------------------
// Device helpers -- verbatim transcriptions of the host functions in 1.cpp.
// ---------------------------------------------------------------------------
// 1/h and the kernel coefficients are compile-time constants: keep them out of
// the inner loop.  (FP64 division on this GPU is a software routine costing
// ~20-40 instructions -- 17 of them per neighbour visit was the hot spot.)
template <class Real>
__device__ __forceinline__ constexpr Real sph_inv_h()
{
    return Real(1.0) / (Real(1.5) * Real(SPH_HSML));
}

template <class Real>
__device__ __forceinline__ Real sph_kernel_q(Real q)
{
    const Real h = Real(1.5) * Real(SPH_HSML);
    const Real coe = Real(21.0) / (Real(256.0) * h * h * h * Real(SPH_PI_));
    if (q >= Real(0.0) && q <= Real(2.0))
        return coe * (Real(1.0) + Real(2.0) * q) * (Real(2.0) - q) * (Real(2.0) - q)
               * (Real(2.0) - q) * (Real(2.0) - q);
    return Real(0.0);
}

template <class Real>
__device__ __forceinline__ Real sph_dwdr_q(Real q)
{
    const Real h = Real(1.5) * Real(SPH_HSML);
    const Real coe = Real(42.0) / (Real(256.0) * h * h * h * h * Real(SPH_PI_));
    if (q >= Real(0.0) && q <= Real(2.0))
        return coe * ((Real(2.0) - q) * (Real(2.0) - q) * (Real(2.0) - q) * (Real(2.0) - q)
                      - Real(2.0) * (Real(1.0) + Real(2.0) * q) * (Real(2.0) - q)
                        * (Real(2.0) - q) * (Real(2.0) - q));
    return Real(0.0);
}

// The dwdr and dwdx kernels share the same polynomial; evaluating it once and
// reusing it for dwdr + the three dwdx components saves ~24 FP64 ops per
// neighbour visit (the kernel is arithmetic bound, see docs section 11).
template <class Real>
__device__ __forceinline__ Real sph_wq_coe()
{
    const Real h = Real(1.5) * Real(SPH_HSML);
    return Real(42.0) / (Real(256.0) * h * h * h * h * Real(SPH_PI_));
}

template <class Real>
__device__ __forceinline__ Real sph_wq(Real q)
{
    const Real coe = sph_wq_coe<Real>();
    if (q >= Real(0.0) && q <= Real(2.0))
        return coe * ((Real(2.0) - q) * (Real(2.0) - q) * (Real(2.0) - q) * (Real(2.0) - q)
                      - Real(2.0) * (Real(1.0) + Real(2.0) * q) * (Real(2.0) - q)
                        * (Real(2.0) - q) * (Real(2.0) - q));
    return Real(0.0);
}

// dx/dr is passed in as dx * (1/dr) so the caller can hoist the reciprocal
template <class Real>
__device__ __forceinline__ Real sph_dwdx_q(Real dx, Real q, Real inv_dr)
{
    const Real h = Real(1.5) * Real(SPH_HSML);
    const Real coe = Real(42.0) / (Real(256.0) * h * h * h * h * Real(SPH_PI_));
    if (q >= Real(0.0) && q <= Real(2.0))
        return coe * ((Real(2.0) - q) * (Real(2.0) - q) * (Real(2.0) - q) * (Real(2.0) - q)
                      - Real(2.0) * (Real(1.0) + Real(2.0) * q) * (Real(2.0) - q)
                        * (Real(2.0) - q) * (Real(2.0) - q))
               * (dx * inv_dr);
    return Real(0.0);
}

// 1.cpp defines max()/min() with if/else; keep the exact NaN/inf semantics.
template <class Real>
__device__ __forceinline__ Real sph_max(Real mi, Real mj) { return mi >= mj ? mi : mj; }
template <class Real>
__device__ __forceinline__ Real sph_min(Real mi, Real mj) { return mi >= mj ? mj : mi; }

// Cell index, with the same clamping as direct_find()
template <class Real>
__device__ __forceinline__ int sph_cell(Real x, Real y, Real z)
{
    const Real inv = Real(1.0) / Real(SPH_CS);
    int ix = (int)floor((x - Real(SPH_GX0)) * inv); if (ix < 0) ix = 0; if (ix >= SPH_NX) ix = SPH_NX - 1;
    int iy = (int)floor((y - Real(SPH_GY0)) * inv); if (iy < 0) iy = 0; if (iy >= SPH_NY) iy = SPH_NY - 1;
    int iz = (int)floor((z - Real(SPH_GZ0)) * inv); if (iz < 0) iz = 0; if (iz >= SPH_NZ) iz = SPH_NZ - 1;
    return (iz * SPH_NY + iy) * SPH_NX + ix;
}

// y = M * v  for the component-major 3x3 stored in M at particle i
template <class Real>
__device__ __forceinline__ void sph_matvec(const Real* M, int i,
                                           Real v0, Real v1, Real v2,
                                           Real& y0, Real& y1, Real& y2)
{
    y0 = M[0 * SPH_NUM + i] * v0 + M[1 * SPH_NUM + i] * v1 + M[2 * SPH_NUM + i] * v2;
    y1 = M[3 * SPH_NUM + i] * v0 + M[4 * SPH_NUM + i] * v1 + M[5 * SPH_NUM + i] * v2;
    y2 = M[6 * SPH_NUM + i] * v0 + M[7 * SPH_NUM + i] * v1 + M[8 * SPH_NUM + i] * v2;
}

// ---------------------------------------------------------------------------
// Kernels / launch wrappers (defined in sph_kernels.cu, instantiated for
// double and float there)
// ---------------------------------------------------------------------------
// A3 switch: 1 = use the static solid pair list (default), 0 = walk the cell
// list every step like before A3 (kept for A/B verification; SPH_SOLDDYN=1).
extern int sph_use_static_solid;
// A5 switch: 1 = factored inner loop of pass1_fluid (default), 0 = the literal
// transcription of 1.cpp (kept for verification; SPH_LITERALFLUID=1).
extern int sph_fast_fluid_loop;

template <class Real> void launch_cell_hist(SphPtrsT<Real> s, int* d_count, int* d_solid, int* d_fluid, cudaStream_t st);
void launch_cell_scan(int* d_cellStart, int* d_count, cudaStream_t st);
void launch_cell_sort(const int* d_cellStart, int* d_sorted, cudaStream_t st);
template <class Real> void launch_cell_scatter(SphPtrsT<Real> s, int* d_cursor, cudaStream_t st);
template <class Real> void launch_count_pairs(SphPtrsT<Real> s, int* d_niac, cudaStream_t st);
template <class Real> void launch_nbr_count(SphPtrsT<Real> s, int* d_out, cudaStream_t st);
template <class Real> void launch_nbr_list(SphPtrsT<Real> s, int target, int* d_out, int* d_n, cudaStream_t st);

template <class Real> void launch_bench_loads(SphPtrsT<Real> s, Real* out, bool scatter, cudaStream_t st);
template <class Real> void launch_build_solid_pack(SphPtrsT<Real> s, cudaStream_t st);
// A4: refresh the float position copy the neighbour test pre-filters with
template <class Real> void launch_copy_xf(SphPtrsT<Real> s, cudaStream_t st);
// A3: two passes over the reference configuration (count, then fill)
template <class Real> void launch_solid_pair_count(SphPtrsT<Real> s, int* d_cnt, cudaStream_t st);
template <class Real> void launch_solid_pair_fill(SphPtrsT<Real> s, const int* d_start, int* d_cur, cudaStream_t st);
template <class Real> void launch_prepare(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_pass1(SphPtrsT<Real> s, cudaStream_t st);
// individual kernels (used by the benchmark mode)
template <class Real> void launch_pass1_fluid(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_pass1_solid(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_pass2_fluid(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_pass2_solid(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_update_fluid(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_update_solid(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_update_pos(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_constitution(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_pass2(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_update(SphPtrsT<Real> s, cudaStream_t st);

template <class Real> void launch_nor_sumvecn(SphPtrsT<Real> s, Real* d_sumvecn, cudaStream_t st);
template <class Real> void launch_nor_normalize(SphPtrsT<Real> s, Real* d_sumvecn, cudaStream_t st);
template <class Real> void launch_gra_correc(SphPtrsT<Real> s, Real* d_a, cudaStream_t st);
template <class Real> void launch_gra_invert(SphPtrsT<Real> s, Real* d_a, cudaStream_t st);
