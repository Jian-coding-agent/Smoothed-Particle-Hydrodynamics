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
// Runtime problem size.  Everything that used to be a compile-time constant
// (num / ntotal / solnum and the neighbour grid) now comes from the deck, so
// one binary runs any case.  The host fills s.d before the first launch; the
// kernels read it through the SphPtrsT they already receive.
//   [0, ntotal)        fluid       itype > 0
//   [ntotal, nall)     solid       itype < -10
//   [nall, num)        wall        -10 <= itype <= 0
// ---------------------------------------------------------------------------
struct SphDims
{
    int num;        // total particles (owned by this rank)
    int ntotal;     // fluid
    int solnum;     // solid
    int nall;       // fluid + solid
    int nwall;      // rigid
    int nx, ny, nz; // neighbour grid
    int ncell;
    double gx0, gy0, gz0;   // grid origin (cell edge = SPH_CS = 3*hsml)

    // ------------------------------------------------------------------
    // Domain decomposition (D1).  A rank owns [0, num) and may also hold a
    // copy of neighbouring ranks' boundary particles ("ghosts") in
    // [num, numP): first the ghost fluid, then the ghost solid, then the ghost
    // wall.  Ghosts are read by the physics kernels but never written: every
    // process loop is bounded by the owned counts above, so a ghost can never
    // race with the owner.  numP == num on a single-GPU run.
    // ------------------------------------------------------------------
    int numP;       // particles present in the arrays (num + ghosts)
    int gs0;        // first ghost-solid index (numP when there are none)
    int ngs;        // number of ghost solids
};

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

// Neighbour grid: cell edge = 3*hsml, origin/dims computed from the deck.
#define SPH_CS    (3.0 * SPH_HSML)
#define SPH_PIA_STRIDE 12

// ---------------------------------------------------------------------------
// D1: the domain decomposition keeps a rank's ghost set complete by selecting
// every particle within 3h of its box, plus a skin (C9).  The skin is what the
// displacement trigger spends before a rebuild, so it is in metres; the default
// is one and a half smoothing lengths.
// ---------------------------------------------------------------------------
#define SPH_D1_SKIN_DEFAULT (1.5 * SPH_HSML)

// ---------------------------------------------------------------------------
// D1 exchange records (doubles per particle).  One record per exchange point,
// holding only the fields the receiving kernels actually read (P9):
//
//   E1  fluid  x, vx, rho, SurFD                                    (8)
//   E1  solid  x, vx, rho                                           (7)
//   E2  solid  rho, Pia[12]                                         (13)
//   init solid/wall  A[9], vecn[3]                                  (12)
//
// p and irho are pure functions of rho, so the receiving side rebuilds them
// with the identical expressions instead of paying two doubles per particle
// (launch_xch_fixup).  PIJW is never written for a fluid (see DESIGN 2.5) and
// never read for a ghost solid, so it is not exchanged at all: E3 is gone (P8).
// ---------------------------------------------------------------------------
#define SPH_XCH_REC_F   8
#define SPH_XCH_REC_S   7
#define SPH_XCH_REC_E2  13
#define SPH_XCH_REC_ST  12

// P1: slots per fluid particle in the neighbour list.  This deck's worst fluid
// particle has 115 neighbours (mean 104); the build kernel counts what does not
// fit and the body kernel falls back to the fused traversal for that particle,
// so the value is a memory/speed trade-off and never a correctness limit.
#define SPH_FLIST_SLOTS 192

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
    SphDims d;      // runtime sizes / grid, filled by the host before any launch
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
    // list is built once, from a cell list binned from X0 (review finding C1),
    // and stored slot-major (ELL, finding P2) instead of CSR: entry j of solid
    // particle k lives at
    //     e = j*SOLNUM + k        with  0 <= j < sCnt[k] <= swidth
    // so a warp walking step j touches 32 consecutive slots.  The nine tensor
    // components are stored in separate arrays (component c at c*L + e), which
    // makes each of those loads a contiguous 32-lane access.
    // ------------------------------------------------------------------
    int   npair;    // valid pairs stored (sum of sCnt), padding excluded
    int   swidth;   // ELL width: max sCnt[k]
    int   sw;       // swidth * SOLNUM (slot count = length of one component array)
    int*  sCnt;     // [SOLNUM]  valid entries of each solid particle
    int*  sQ;       // [sw]      partner particle index
    Real* sG;       // [3*sw]    w0 * gradW(X0_q - X0_p)   -> DFdt, PIJW
    Real* sB;       // [3*sw]    w0 * (A_p gradW)          -> F
    Real* sD;       // [3*sw]    w0 * (A_q (-gradW))       -> PIJW

    // ------------------------------------------------------------------
    // P1: per-fluid-particle neighbour list (fp64 only; the fp32 path is
    // traversal-bound and keeps the fused kernel).  Slot-major like A3:
    // slot j of fluid particle p is fnbr[j*NTOTAL + p].  Rebuilt every step
    // by k_fluid_nbr_build, in exactly the order the fused kernel would visit
    // the neighbours, so the summation order (and the bits) are unchanged.
    // fcnt[p] < 0 means "did not fit in fwidth" and the step falls back to the
    // fused traversal for that particle.
    // ------------------------------------------------------------------
    int*  fnbr;     // [fwidth*NTOTAL]
    int*  fcnt;     // [NTOTAL]
    int   fwidth;   // slots per particle (0 = list disabled)

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

    // ------------------------------------------------------------------
    // D1: global particle id of every local slot: the deck index for an owned
    // particle, the id it was copied from for a ghost.  nullptr on a
    // single-GPU run, where the local index IS the global id -- the two
    // kernels that need the reference solver's "higher indexed particle" rule
    // (the eta selection in pass1_fluid) then keep using the local index and
    // their behaviour is unchanged bit for bit.
    // ------------------------------------------------------------------
    int* gid;       // [NUM] (or [NUM_P] with ghosts)
};

// The reference solver's eta rule takes the SurFD of the *higher indexed* of
// the two particles (the second of the two assignments in 1.cpp wins, and the
// original always has i < j).  With DomainDecomposition a ghost slot is
// appended after every owned one, so the local index no longer reproduces the
// deck order; s.gid restores it.
template <class Real>
__device__ __forceinline__ int sph_pair_hi(const SphPtrsT<Real>& s, int p, int q)
{
    if (s.gid != nullptr)
        return (s.gid[p] > s.gid[q]) ? p : q;
    return (p > q) ? p : q;
}

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
__device__ __forceinline__ int sph_cell(const SphPtrsT<Real>& s, Real x, Real y, Real z)
{
    const Real inv = Real(1.0) / Real(SPH_CS);
    const SphDims& d = s.d;
    int ix = (int)floor((x - Real(d.gx0)) * inv); if (ix < 0) ix = 0; if (ix >= d.nx) ix = d.nx - 1;
    int iy = (int)floor((y - Real(d.gy0)) * inv); if (iy < 0) iy = 0; if (iy >= d.ny) iy = d.ny - 1;
    int iz = (int)floor((z - Real(d.gz0)) * inv); if (iz < 0) iz = 0; if (iz >= d.nz) iz = d.nz - 1;
    return (iz * d.ny + iy) * d.nx + ix;
}

// y = M * v  for the component-major 3x3 stored in M at particle i
template <class Real>
__device__ __forceinline__ void sph_matvec(const SphPtrsT<Real>& s, const Real* M, int i,
                                           Real v0, Real v1, Real v2,
                                           Real& y0, Real& y1, Real& y2)
{
    // C15: the tensors are numP-strided (owned + ghosts), not num-strided
    const int N = s.d.numP;
    y0 = M[0 * N + i] * v0 + M[1 * N + i] * v1 + M[2 * N + i] * v2;
    y1 = M[3 * N + i] * v0 + M[4 * N + i] * v1 + M[5 * N + i] * v2;
    y2 = M[6 * N + i] * v0 + M[7 * N + i] * v1 + M[8 * N + i] * v2;
}

// ---------------------------------------------------------------------------
// Kernels / launch wrappers (defined in sph_kernels.cu, instantiated for
// double and float there)
// ---------------------------------------------------------------------------
// A3 switch: 1 = use the static solid pair list (default), 0 = walk the cell
// list every step like before A3 (kept for A/B verification; SPH_SOLDDYN=1).
extern int sph_use_static_solid;
// A6 switch: 1 = the static list stores only the partner index (sQ) and the
// per-pair weights are recomputed from solidPack every step (default), so the
// list costs 4 B/pair instead of 88 B/pair; 0 = store sG/sB/sD like A3 did
// (kept for A/B verification and as the faster-but-fatter mode; SPH_A3_FULL=1).
// Both modes are bit-identical: the lean path evaluates the same expressions,
// in the same order, as k_solid_pair_fill did when it filled the arrays.
extern int sph_a3_lean;
// C12: the auto lean/fat decision must be the same on every rank of a D1 run,
// so the decomposed driver installs this hook.  build_solid_lists() offers its
// own fat size and gets back the agreed maximum (all ranks meet at the rank
// barrier); a null hook means "single rank, decide locally".
extern double (*sph_a3_fat_hook)(double fat_mb);
// A5 switch: 1 = factored inner loop of pass1_fluid (default), 0 = the literal
// transcription of 1.cpp (kept for verification; SPH_LITERALFLUID=1).
extern int sph_fast_fluid_loop;
// P1 switch: 1 = two-phase fp64 pass1_fluid (default), 0 = fused traversal.
// Ignored in fp32, where the fused kernel is faster (SPH_FLUIDLIST=0).
extern int sph_use_fluid_list;
// C8: particles whose gradient-correction matrix fell back to the identity.
// Non-zero makes the fp32 displacement form (F - I) invalid; checked at start-up.
int gra_fallback_count();

template <class Real> void launch_cell_hist(SphPtrsT<Real> s, int* d_count, int* d_solid, int* d_fluid, cudaStream_t st);
void launch_cell_scan(int* d_cellStart, int* d_count, int ncell, cudaStream_t st);
void launch_cell_sort(const int* d_cellStart, int* d_sorted, int ncell, cudaStream_t st);
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
template <class Real> void launch_solid_pair_fill(SphPtrsT<Real> s, int* d_cur, cudaStream_t st);
// P1: record the fluid neighbour lists, then run pass1_fluid over them
template <class Real> void launch_fluid_nbr_build(SphPtrsT<Real> s, cudaStream_t st);
template <class Real> void launch_pass1_fluid_over_list(SphPtrsT<Real> s, cudaStream_t st);
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

// ---------------------------------------------------------------------------
// D1: exchange kernels (fp64 only).  One gather kernel per record type packs
// the owned particles of a link into a device staging block, the host copies
// the whole block once, relays the peers' blocks with plain memcpy, and one
// scatter kernel per record type writes the ghost slots.  See sph_kernels.cu.
// ---------------------------------------------------------------------------
void launch_xch_gather(const SphPtrsD s, int mode, const int* d_idx, const int* d_off,
                       int n, double* d_stage, cudaStream_t st);
void launch_xch_scatter(SphPtrsD s, int mode, const int* d_idx, const int* d_off,
                        int n, const double* d_base, cudaStream_t st);
// TASK 2 / P9: rebuild 1/rho (and, for the ghost fluid block, p) after E1
// instead of sending either of them
void launch_xch_fixup(SphPtrsD s, cudaStream_t st);
// C9: max |x - x_build| over the owned particles, as the IEEE bit pattern of a
// non-negative double so it can be reduced with atomicMax
void launch_disp_max(const SphPtrsD s, const double* d_xbuild,
                     unsigned long long* d_max, cudaStream_t st);
