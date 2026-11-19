// ============================================================================
//  sph_kernels.cu -- device kernels for the CUDA port of 1.cpp
//
//  Everything here is templated on Real (double / float); the launch wrappers
//  are explicitly instantiated for both types at the bottom of this file.
//
//  Architecture (see docs/CUDA_PORT_ANALYSIS.md):
//    * one thread per particle, each particle "gathers" its own half of every
//      pair interaction  ->  no atomics on the physics, no per-thread buffers,
//      no reduction passes at all;
//    * neighbour search = fixed uniform grid (same domain / cell size / clamp
//      rule as direct_find), 27 neighbour cells scanned per particle;
//    * every branch below is a literal transcription of the corresponding
//      block in 1.cpp (the pair symmetry that makes gather possible, and the
//      one quantity that is NOT symmetric -- eta -- are documented inline).
// ============================================================================
#include "sph.h"

#define N (s.d.num)

// C8: number of particles whose gradient-correction matrix is singular and fell
// back to A = I (k_gra_invert).  Declared here so the kernel below can bump it;
// the host reads it through gra_fallback_count().
static __device__ int sph_gra_fallbacks = 0;

// A4 pre-filter: is candidate q certainly outside 3*hsml?  Compiled only for
// Real = double -- for Real = float the exact test below is already fp32, so a
// float pre-filter would just add three loads and six operations per candidate
// (measured: pass1_fluid 0.74 -> 0.83 ms in the fp32 build).
template <class Real>
__device__ __forceinline__ bool sph_reject_far(const SphPtrsT<Real>& s, int q,
                                               float f0p, float f1p, float f2p, float lim2)
{
    if constexpr (sizeof(Real) > sizeof(float))
    {
        const float f0 = f0p - s.xf[q];
        const float f1 = f1p - s.xf[N + q];
        const float f2 = f2p - s.xf[2 * N + q];
        return (f0 * f0 + f1 * f1 + f2 * f2) >= lim2;
    }
    else
    {
        return false;
    }
}

// ---------------------------------------------------------------------------
// neighbour loop: visits every q inside 3*hsml, with d = x_p - x_q (the same
// difference the original code forms as x[i]-x[j] for the i side)
//
// A4: each candidate is first rejected by a float distance estimate
//     r2f >= r2max * (1 + 1e-4)  ->  skip
// and only the survivors take the exact sph_add_rn/sph_mul_rn test below, which
// is what decides membership.  The float estimate can only be *larger* than the
// exact r2 up to a bounded rounding error: with |x| <= 1.2 m the bound is
// 3.5*ulp_float(1.2)/(3*hsml) = 1.4e-5 of r2max for this domain, and the margin
// is 1e-4, i.e. 7x inside it.  So no true neighbour is ever skipped and the
// accepted set (and every result bit) stays identical -- verified on this data
// as well (500 and 3000 steps, see docs section 11).
// ---------------------------------------------------------------------------
template <class Real, class F>
__device__ __forceinline__ void sph_neigh(const SphPtrsT<Real>& s, int p,
                                          Real xp0, Real xp1, Real xp2, F&& body)
{
    const Real r2max = (Real(3.0) * Real(SPH_HSML)) * (Real(3.0) * Real(SPH_HSML));
    const float lim2 = (float)r2max * 1.0001f;
    const float f0p = (float)xp0, f1p = (float)xp1, f2p = (float)xp2;
    const int c = sph_cell(s, xp0, xp1, xp2);
    const int ix = c % s.d.nx;
    const int iy = (c / s.d.nx) % s.d.ny;
    const int iz = c / (s.d.nx * s.d.ny);
    for (int dz = -1; dz <= 1; ++dz)
    {
        const int jz = iz + dz; if (jz < 0 || jz >= s.d.nz) continue;
        for (int dy = -1; dy <= 1; ++dy)
        {
            const int jy = iy + dy; if (jy < 0 || jy >= s.d.ny) continue;
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int jx = ix + dx; if (jx < 0 || jx >= s.d.nx) continue;
                const int c2 = (jz * s.d.ny + jy) * s.d.nx + jx;
                const int a0 = s.cellStart[c2];
                const int a1 = s.cellStart[c2 + 1];
                for (int a = a0; a < a1; ++a)
                {
                    const int q = s.sorted[a];
                    if (q == p) continue;
                    if (sph_reject_far(s, q, f0p, f1p, f2p, lim2)) continue;
                    const Real d0 = xp0 - s.x[q];
                    const Real d1 = xp1 - s.x[N + q];
                    const Real d2 = xp2 - s.x[2 * N + q];
                    // The lattice places many pairs exactly at r = 3*hsml, so the
                    // membership test is knife-edge.  sph_mul_rn/sph_add_rn prevent
                    // FMA contraction and reproduce the reference rounding, which
                    // keeps the fp64 pair set identical to direct_find().
                    const Real r2 = sph_add_rn(sph_add_rn(sph_mul_rn(d0, d0), sph_mul_rn(d1, d1)),
                                               sph_mul_rn(d2, d2));
                    if (r2 >= r2max) continue;
                    body(q, d0, d1, d2, r2);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// generic 27-cell walk: body(q) for every candidate in the neighbourhood of
// cell c, in the same visiting order as sph_neigh.  No distance test -- the
// caller decides.  Used only by the one-off reference-configuration passes.
// ---------------------------------------------------------------------------
template <class Real, class F>
__device__ __forceinline__ void sph_neigh_cells(const SphPtrsT<Real>& s, int c, F&& body)
{
    const int ix = c % s.d.nx;
    const int iy = (c / s.d.nx) % s.d.ny;
    const int iz = c / (s.d.nx * s.d.ny);
    for (int dz = -1; dz <= 1; ++dz)
    {
        const int jz = iz + dz; if (jz < 0 || jz >= s.d.nz) continue;
        for (int dy = -1; dy <= 1; ++dy)
        {
            const int jy = iy + dy; if (jy < 0 || jy >= s.d.ny) continue;
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int jx = ix + dx; if (jx < 0 || jx >= s.d.nx) continue;
                const int c2 = (jz * s.d.ny + jy) * s.d.nx + jx;
                const int a0 = s.cellStart[c2];
                const int a1 = s.cellStart[c2 + 1];
                for (int a = a0; a < a1; ++a) body(s.sorted[a]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// grid bookkeeping
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_cell_hist(SphPtrsT<Real> s, int* __restrict__ cnt,
                            int* __restrict__ csolid, int* __restrict__ cfluid)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.num) return;
    const int c = sph_cell(s, s.x[p], s.x[N + p], s.x[2 * N + p]);
    atomicAdd(&cnt[c], 1);
    if (s.itype[p] < Real(-10.0)) atomicAdd(&csolid[c], 1);
    if (s.itype[p] > Real(0.0))   atomicAdd(&cfluid[c], 1);
}

// ---------------------------------------------------------------------------
// static pack for the solid kernels: {X0[3], A[9], rho0, mass} per solid
// particle, contiguous.  Built once (after GraCorrec), never updated.
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_build_solid_pack(SphPtrsT<Real> s, int base)
{
    const int p = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.nall) return;
    const int k = p - s.d.ntotal;
    Real* d = s.solidPack + (size_t)k * 14;
    d[0] = s.X0[p]; d[1] = s.X0[N + p]; d[2] = s.X0[2 * N + p];
    for (int c = 0; c < 9; ++c) d[3 + c] = s.A[(size_t)c * s.d.num + p];
    d[12] = s.rho0[p];
    d[13] = s.mass[p];
}

// ---------------------------------------------------------------------------
// A3: reference-configuration geometry of one solid/solid pair.
// g0 = dwdr(Dr/h) * (X0_p - X0_q)/Dr with Dr = |X0_p - X0_q|, exactly what
// k_pass1_solid / k_pass2_solid evaluate every step.  Returns false when the
// kernel gradient vanishes, i.e. when the pair contributes nothing at all.
// ---------------------------------------------------------------------------
template <class Real>
__device__ __forceinline__ bool solid_pair_geom(const SphPtrsT<Real>& s, int q,
                                                Real Xp0, Real Xp1, Real Xp2,
                                                Real& g0x, Real& g0y, Real& g0z)
{
    const Real gX0 = Xp0 - s.X0[q];
    const Real gX1 = Xp1 - s.X0[N + q];
    const Real gX2 = Xp2 - s.X0[2 * N + q];
    const Real Dr = sqrt(gX0 * gX0 + gX1 * gX1 + gX2 * gX2);
    // C2: the lattice puts a large part of the solid/solid pairs exactly on
    // r = 3*hsml.  There qr rounds to 2 - 1ulp for some of them, so wq returns
    // a denormal ~1e-40 instead of 0 and the pair is stored only to add a
    // quantity that cannot change a single bit of the result (it is below the
    // ulp of every accumulator).  Rejecting them geometrically shortens the ELL
    // width and the per-step streaming by 15-20 %, with identical output.  A
    // magnitude threshold does not work; in fp32 wq(~2) is ~1e-20, not 0.
    // The margin has to be representable in Real: 1e-9 is below the fp32
    // epsilon, so fp32 needs 1e-6 (still a 3e-8 m band, where wq ~ 1e-17).
    const Real margin = (sizeof(Real) < sizeof(double)) ? Real(1e-6) : Real(1e-9);
    if (!(Dr < Real(3.0) * Real(SPH_HSML) * (Real(1.0) - margin))) return false;
    const Real qr = Dr * sph_inv_h<Real>();
    const Real wc = sph_wq<Real>(qr);
    if (wc == Real(0.0)) return false;          // also catches q == p (Dr = 0)
    const Real pinvDr = Real(1.0) / Dr;
    g0x = wc * (gX0 * pinvDr);
    g0y = wc * (gX1 * pinvDr);
    g0z = wc * (gX2 * pinvDr);
    return true;
}

// pass 1 of 2: number of stored pairs per solid particle
template <class Real>
__global__ void k_solid_pair_count(SphPtrsT<Real> s, int* __restrict__ cnt)
{
    const int p = s.d.ntotal + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.nall) return;

    const Real Xp0 = s.X0[p], Xp1 = s.X0[N + p], Xp2 = s.X0[2 * N + p];
    int n = 0;
    sph_neigh_cells(s, sph_cell(s, Xp0, Xp1, Xp2), [&](int q)
    {
        if (q == p || s.itype[q] >= Real(-10.0)) return;
        Real g0x, g0y, g0z;
        if (solid_pair_geom(s, q, Xp0, Xp1, Xp2, g0x, g0y, g0z)) ++n;
    });
    cnt[p - s.d.ntotal] = n;
}

// pass 2 of 2: write the per-pair constants at slot j*SOLNUM + k (ELL, P2).
// One thread owns one solid particle, so the slot index needs no atomics.
//   G = w0 * g0,  B = w0 * (A_p g0),  D = w0 * (A_q (-g0)),  w0 = m_q/rho0_q
// cur[k] is written back so the host can assert (C3) that this pass stored
// exactly as many entries as k_solid_pair_count counted.
template <class Real>
__global__ void k_solid_pair_fill(SphPtrsT<Real> s, int* __restrict__ cur)
{
    const int p = s.d.ntotal + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.nall) return;

    const int k = p - s.d.ntotal;
    const Real Xp0 = s.X0[p], Xp1 = s.X0[N + p], Xp2 = s.X0[2 * N + p];
    const Real Ap0 = s.A[p],         Ap1 = s.A[N + p],     Ap2 = s.A[2 * N + p];
    const Real Ap3 = s.A[3 * N + p], Ap4 = s.A[4 * N + p], Ap5 = s.A[5 * N + p];
    const Real Ap6 = s.A[6 * N + p], Ap7 = s.A[7 * N + p], Ap8 = s.A[8 * N + p];

    const size_t L = (size_t)s.sw;              // length of one component array
    int j = 0;
    sph_neigh_cells(s, sph_cell(s, Xp0, Xp1, Xp2), [&](int q)
    {
        if (q == p || s.itype[q] >= Real(-10.0)) return;
        Real g0x, g0y, g0z;
        if (!solid_pair_geom(s, q, Xp0, Xp1, Xp2, g0x, g0y, g0z)) return;

        const Real w0 = s.mass[q] / s.rho0[q];
        const Real Gx = w0 * g0x, Gy = w0 * g0y, Gz = w0 * g0z;

        const Real Bx = Ap0 * Gx + Ap1 * Gy + Ap2 * Gz;
        const Real By = Ap3 * Gx + Ap4 * Gy + Ap5 * Gz;
        const Real Bz = Ap6 * Gx + Ap7 * Gy + Ap8 * Gz;

        const Real Aq0 = s.A[q],         Aq1 = s.A[N + q],     Aq2 = s.A[2 * N + q];
        const Real Aq3 = s.A[3 * N + q], Aq4 = s.A[4 * N + q], Aq5 = s.A[5 * N + q];
        const Real Aq6 = s.A[6 * N + q], Aq7 = s.A[7 * N + q], Aq8 = s.A[8 * N + q];
        const Real Dx = -(Aq0 * Gx + Aq1 * Gy + Aq2 * Gz);
        const Real Dy = -(Aq3 * Gx + Aq4 * Gy + Aq5 * Gz);
        const Real Dz = -(Aq6 * Gx + Aq7 * Gy + Aq8 * Gz);

        const size_t e = (size_t)j * s.d.solnum + k;
        s.sQ[e]      = q;
        s.sG[e]      = Gx; s.sG[L + e]      = Gy; s.sG[2 * L + e]      = Gz;
        s.sB[e]      = Bx; s.sB[L + e]      = By; s.sB[2 * L + e]      = Bz;
        s.sD[e]      = Dx; s.sD[L + e]      = Dy; s.sD[2 * L + e]      = Dz;
        ++j;
    });
    cur[k] = j;
}

// exclusive scan over s.d.ncell, then re-arm the cursor array for the scatter
// pass.  One 1024-thread block walks the cells in chunks of 1024*32 and carries
// the running total across chunks, so the grid may be any size (the new deck
// needs 188552 cells; the old case needed 32076, hence the old static_assert).
__global__ void k_cell_scan(int* __restrict__ start, int* __restrict__ cnt, int ncell)
{
    const int T = 512;
    const int C = 32;                       // cells per thread per chunk
    __shared__ int sh[512];
    __shared__ int carry_sh;
    const int t = threadIdx.x;
    int carry = 0;
    for (int base = 0; base < ncell; base += T * C)
    {
        int own = 0;
        for (int k = 0; k < C; ++k)
        {
            const int c = base + t * C + k;
            if (c < ncell) own += cnt[c];
        }
        sh[t] = own;
        __syncthreads();

        for (int off = 1; off < T; off <<= 1)
        {
            const int v = (t >= off) ? sh[t - off] : 0;
            __syncthreads();
            sh[t] += v;
            __syncthreads();
        }

        int run = carry + sh[t] - own;
        for (int k = 0; k < C; ++k)
        {
            const int c = base + t * C + k;
            if (c < ncell)
            {
                const int v = cnt[c];
                start[c] = run;
                cnt[c] = run;
                run += v;
            }
        }
        __syncthreads();
        if (t == T - 1) carry_sh = carry + sh[T - 1];
        __syncthreads();
        carry = carry_sh;
    }
    if (t == 0) start[ncell] = carry;
}

// deterministic order inside each cell (insertion sort by particle index).
// Without it the neighbour sum order follows the atomic scatter arrival order,
// so runs are not bit-identical and an optimisation cannot be verified by bits.
__global__ void k_cell_sort(const int* __restrict__ start, int* __restrict__ sorted, int ncell)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    const int a0 = start[c], a1 = start[c + 1];
    for (int a = a0 + 1; a < a1; ++a)
    {
        const int v = sorted[a];
        int b = a - 1;
        while (b >= a0 && sorted[b] > v) { sorted[b + 1] = sorted[b]; --b; }
        sorted[b + 1] = v;
    }
}

template <class Real>
__global__ void k_cell_scatter(SphPtrsT<Real> s, int* __restrict__ cursor)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.num) return;
    const int c = sph_cell(s, s.x[p], s.x[N + p], s.x[2 * N + p]);
    const int pos = atomicAdd(&cursor[c], 1);
    s.sorted[pos] = p;
}

// per-particle neighbour count (debug)
template <class Real>
__global__ void k_nbr_count(SphPtrsT<Real> s, int* __restrict__ out)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.num) return;
    int n = 0;
    sph_neigh(s, p, s.x[p], s.x[N + p], s.x[2 * N + p],
              [&](int, Real, Real, Real, Real) { n += 1; });
    out[p] = n;
}

// neighbour list of one particle (debug)
template <class Real>
__global__ void k_nbr_list(SphPtrsT<Real> s, int target, int* __restrict__ out, int* __restrict__ n)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int k = 0;
    sph_neigh(s, target, s.x[target], s.x[N + target], s.x[2 * N + target],
              [&](int q, Real, Real, Real, Real) { out[k++] = q; });
    *n = k;
}

// number of unordered pairs (only used for the "particles pairs" console line)
template <class Real>
__global__ void k_count_pairs(SphPtrsT<Real> s, int* __restrict__ niac)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.num) return;
    int n = 0;
    sph_neigh(s, p, s.x[p], s.x[N + p], s.x[2 * N + p],
              [&](int, Real, Real, Real, Real) { n += 1; });
    atomicAdd(niac, n);                     // host divides by 2
}

// ---------------------------------------------------------------------------
// profiling probe: identical neighbour traversal, reads the same 13 fields a
// real pass1_fluid visit reads, but does no arithmetic.  SCATTER=true reads
// the neighbour (the real access pattern); SCATTER=false reads a coalesced
// window across the warp, which isolates the cost of the scattered access.
// ---------------------------------------------------------------------------
template <class Real, bool SCATTER>
__global__ void k_bench_loads(SphPtrsT<Real> s, Real* __restrict__ out)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.ntotal) return;
    const Real xp0 = s.x[p], xp1 = s.x[N + p], xp2 = s.x[2 * N + p];
    Real acc = Real(0.0);
    int it = 0;
    sph_neigh2(s, p, xp0, xp1, xp2, [&](int q, Real, Real, Real, Real)
    {
        int base = p + (it & 7); ++it;
        const int icoal = (base >= s.d.num) ? (base - s.d.num) : base;
        const int idx = SCATTER ? q : icoal;
        acc += s.x[idx] + s.x[N + idx] + s.x[2 * N + idx]
             + s.vx[idx] + s.vx[N + idx] + s.vx[2 * N + idx]
             + s.rho[idx] + s.irho[idx] + s.mass[idx] + s.p[idx]
             + s.itype[idx] + s.SurFD[idx] + s.vecn[idx];
    });
    out[p] = acc;
}
// ---------------------------------------------------------------------------
// ILP variant: exactly the same neighbour set and the same per-particle
// accumulation order, but two neighbours are processed per loop iteration so
// the two independent sqrt/divide chains can overlap in the pipeline.
// ---------------------------------------------------------------------------
template <class Real, class F>
__device__ __forceinline__ void sph_neigh2(const SphPtrsT<Real>& s, int p,
                                           Real xp0, Real xp1, Real xp2, F&& body)
{
    const Real r2max = (Real(3.0) * Real(SPH_HSML)) * (Real(3.0) * Real(SPH_HSML));
    const float lim2 = (float)r2max * 1.0001f;      // A4 pre-filter, see sph_neigh
    const float f0p = (float)xp0, f1p = (float)xp1, f2p = (float)xp2;
    const int c = sph_cell(s, xp0, xp1, xp2);
    const int ix = c % s.d.nx;
    const int iy = (c / s.d.nx) % s.d.ny;
    const int iz = c / (s.d.nx * s.d.ny);
    for (int dz = -1; dz <= 1; ++dz)
    {
        const int jz = iz + dz; if (jz < 0 || jz >= s.d.nz) continue;
        for (int dy = -1; dy <= 1; ++dy)
        {
            const int jy = iy + dy; if (jy < 0 || jy >= s.d.ny) continue;
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int jx = ix + dx; if (jx < 0 || jx >= s.d.nx) continue;
                const int c2 = (jz * s.d.ny + jy) * s.d.nx + jx;
                const int a0 = s.cellStart[c2];
                const int a1 = s.cellStart[c2 + 1];
                int a = a0;
                for (; a + 1 < a1; a += 2)
                {
                    const int q1 = s.sorted[a];
                    const int q2 = s.sorted[a + 1];
                    const bool k1 = (q1 != p) && !sph_reject_far(s, q1, f0p, f1p, f2p, lim2);
                    const bool k2 = (q2 != p) && !sph_reject_far(s, q2, f0p, f1p, f2p, lim2);
                    if (!k1 && !k2) continue;
                    if (k1)
                    {
                        const Real e00 = xp0 - s.x[q1];
                        const Real e01 = xp1 - s.x[N + q1];
                        const Real e02 = xp2 - s.x[2 * N + q1];
                        const Real ra = sph_add_rn(sph_add_rn(sph_mul_rn(e00, e00), sph_mul_rn(e01, e01)), sph_mul_rn(e02, e02));
                        if (ra < r2max) body(q1, e00, e01, e02, ra);
                    }
                    if (k2)
                    {
                        const Real e10 = xp0 - s.x[q2];
                        const Real e11 = xp1 - s.x[N + q2];
                        const Real e12 = xp2 - s.x[2 * N + q2];
                        const Real rb = sph_add_rn(sph_add_rn(sph_mul_rn(e10, e10), sph_mul_rn(e11, e11)), sph_mul_rn(e12, e12));
                        if (rb < r2max) body(q2, e10, e11, e12, rb);
                    }
                }
                for (; a < a1; ++a)
                {
                    const int q = s.sorted[a];
                    if (q == p) continue;
                    if (sph_reject_far(s, q, f0p, f1p, f2p, lim2)) continue;
                    const Real d0 = xp0 - s.x[q];
                    const Real d1 = xp1 - s.x[N + q];
                    const Real d2 = xp2 - s.x[2 * N + q];
                    const Real r2 = sph_add_rn(sph_add_rn(sph_mul_rn(d0, d0), sph_mul_rn(d1, d1)), sph_mul_rn(d2, d2));
                    if (r2 >= r2max) continue;
                    body(q, d0, d1, d2, r2);
                }
            }
        }
    }
}
// ---------------------------------------------------------------------------
// step 2.0 : SurFD := SurF ; p := c0^2 (rho-1000) clamped at 0
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_prepare(SphPtrsT<Real> s)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.d.num) return;
    const Real C0 = Real(SPH_C0);
    s.SurFD[i] = s.SurF[i];
    Real pi_ = C0 * C0 * (s.rho[i] - Real(1000.0));
    if (s.rho[i] < Real(1000.0)) pi_ = Real(0.0);
    s.p[i] = pi_;
    s.irho[i] = Real(1.0) / s.rho[i];
}

// ---------------------------------------------------------------------------
// step 2.1 : fluid side of pass 1
//   fluid/fluid        -> drhodt, dvxdt, viscos, SurF
//   fluid/(solid|wall) -> drhodt, dvxdt, viscos, SurF, using vecn[j]
// (AveV is accumulated by 1.cpp but never read; it is not computed here.)
// ---------------------------------------------------------------------------
template <class Real>
__global__ void __launch_bounds__(256, 4) k_pass1_fluid(SphPtrsT<Real> s)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.ntotal) return;

    const Real C0   = Real(SPH_C0);
    const Real HS   = Real(SPH_HSML);
    const Real GRAV = Real(9.8);

    const Real xp0 = s.x[p], xp1 = s.x[p + N], xp2 = s.x[p + 2 * N];
    const Real vi0 = s.vx[p], vi1 = s.vx[N + p], vi2 = s.vx[2 * N + p];
    const Real rhoi = s.rho[p], irhoi = s.irho[p], pi_ = s.p[p];

    Real adrho = Real(0.0), ad0 = Real(0.0), ad1 = Real(0.0), ad2 = Real(0.0);
    Real av0 = Real(0.0), av1 = Real(0.0), av2 = Real(0.0), asurf = Real(0.0);

    sph_neigh2(s, p, xp0, xp1, xp2, [&](int q, Real d0, Real d1, Real d2, Real r2)
    {
        const Real dr = sqrt(r2 + Real(0.0000000000001));   // 1e-13, as in 1.cpp
        const Real pinv = Real(1.0) / dr;
        const Real qk = dr * sph_inv_h<Real>();
        const Real e0 = -d0 * pinv, e1 = -d1 * pinv, e2 = -d2 * pinv;

        const Real mq = s.mass[q], rhoq = s.rho[q], irhoq = s.irho[q], pq = s.p[q];

        // 1.cpp:
        //     if(fabs(SurFD[i])<2.5){eta = 0.001;}else{eta = 0.00002;}
        //     if(fabs(SurFD[j])<2.5){eta = 0.001;}else{eta = 0.00002;}
        // The second assignment overwrites the first, so BOTH sides of a pair use
        // the SurFD of the *higher indexed* particle j (the original always works
        // with i<j).  Using the neighbour's value would only be right when the
        // current particle happens to be the i side; on the j side it would use
        // eta = 0.001 instead of 2e-5 (a factor of 50 in the viscous term).
        const Real sfd_j = s.SurFD[(p > q) ? p : q];
        const Real eta = (fabs(sfd_j) < Real(2.5)) ? Real(0.001) : Real(0.00002);

        const Real wc = sph_wq<Real>(qk);
        const Real w0 = wc * (d0 * pinv);
        const Real w1 = wc * (d1 * pinv);
        const Real w2 = wc * (d2 * pinv);
        const Real S  = w0 * d0 + w1 * d1 + w2 * d2;
        const Real invfac = irhoq / (dr * dr + Real(0.01) * HS * HS);

        Real UL, UR, PL, PR, rhoL, rhoR;

        if (s.itype[q] > Real(0.0))
        {
            // ---------------- fluid / fluid ----------------
            PL = pi_;  PR = pq;
            rhoL = rhoi; rhoR = rhoq;
            UL = vi0 * e0 + vi1 * e1 + vi2 * e2;
            UR = s.vx[q] * e0 + s.vx[N + q] * e1 + s.vx[2 * N + q] * e2;
        }
        else
        {
            // ---------------- fluid / (solid or wall) ----------------
            PL   = pi_;
            PR   = pi_ - rhoi * GRAV * (-d2);       // x[2][j]-x[2][i]
            rhoL = rhoi;
            rhoR = Real(1000.0) + PR * (Real(1.0) / (C0 * C0));
            UL   = -(vi0 * s.vecn[q] + vi1 * s.vecn[N + q] + vi2 * s.vecn[2 * N + q]);
            UR   = -UL;
        }

        const Real rhoSum = rhoL + rhoR;
        const Real UM = Real(0.5) * (UL + UR)
                      + Real(0.5) * (PL - PR) * (Real(1.0) / (Real(0.5) * rhoSum * C0));
        const Real PM = Real(0.5) * (PL + PR)
                      + Real(0.5) * (Real(0.5) * rhoSum) * C0 * (UL - UR)
                        * sph_min((Real(3.0) / C0) * sph_max(UL - UR, Real(0.0)), Real(1.0));

        adrho += -Real(2.0) * rhoi * (mq * irhoq) * (UL - UM) * wc;
        const Real c = -Real(2.0) * mq * (PM * irhoi * irhoq);
        ad0 += c * w0; ad1 += c * w1; ad2 += c * w2;

        av0 += Real(2.0) * mq * eta * S * (vi0 - s.vx[q]) * invfac;
        av1 += Real(2.0) * mq * eta * S * (vi1 - s.vx[N + q]) * invfac;
        av2 += Real(2.0) * mq * eta * S * (vi2 - s.vx[2 * N + q]) * invfac;

        asurf += S * mq * irhoq;
    });

    s.drhodt[p]   = adrho;
    s.dvxdt[p]    = ad0;
    s.dvxdt[N + p]  = ad1;
    s.dvxdt[2 * N + p] = ad2;
    s.viscos[p]   = av0;
    s.viscos[N + p] = av1;
    s.viscos[2 * N + p] = av2;
    s.SurF[p]     = asurf;
}

// ---------------------------------------------------------------------------
// step 2.1 : fluid side of pass 1  (A5, factored inner loop)
//
// Same physics as k_pass1_fluid, four identities from the review (A5):
//   1. (2-q)^4 - 2(1+2q)(2-q)^3 = -5 q (2-q)^3      -> dW/dr in 4 mults
//   2. only UL - UR is ever used, and it is -(dv.d)/dr for fluid/fluid and
//      -2(vi.n) for a wall
//   3. gradW.d = (dW/dr) * r2 / dr                  -> no component products
//   4. 1/dr, 1/(dr^2+eps2) and 1/rhoSum from ONE division
// About 60 FP64 operations + 1 sqrt + 1 division per neighbour instead of
// about 90 + 1 sqrt + 3 divisions.  Measured deviation against the literal
// version: 6.8e-15 on drhodt, 1.0e-14 on dvxdt, 2.5e-16 on viscos, 8.9e-16 on
// SurF -- the same size as the neighbour-order noise (docs section 11).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// P1: the neighbour list of one fluid particle, in exactly the order the fused
// kernel visits them (same sph_neigh2, so the same cell order and the same
// within-cell order).  fcnt[p] = -1 marks a particle whose list did not fit in
// fwidth slots; the body kernel then falls back to the traversal for it alone,
// so the result is independent of the width.
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_fluid_nbr_build(SphPtrsT<Real> s)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.ntotal) return;
    const Real xp0 = s.x[p], xp1 = s.x[N + p], xp2 = s.x[2 * N + p];
    const int W = s.fwidth;
    int n = 0;
    sph_neigh2(s, p, xp0, xp1, xp2, [&](int q, Real, Real, Real, Real)
    {
        if (n < W) s.fnbr[(size_t)n * s.d.ntotal + p] = q;
        ++n;
    });
    s.fcnt[p] = (n <= W) ? n : -1;
}

// USE_LIST = false: fused traversal (what the port shipped with, and what fp32
// keeps using -- there the body is cheap and the kernel is traversal bound).
// USE_LIST = true : the same body over the recorded list.  The four kernels
// that used to make the fp64 body run under a sparse lane mask are gone.
// ---------------------------------------------------------------------------
template <class Real, bool USE_LIST>
__global__ void __launch_bounds__(256, 4) k_pass1_fluid_body(SphPtrsT<Real> s)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.ntotal) return;

    const Real C0   = Real(SPH_C0);
    const Real HS   = Real(SPH_HSML);
    const Real GRAV = Real(9.8);
    const Real invh = sph_inv_h<Real>();
    const Real C5   = Real(-5.0) * sph_wq_coe<Real>();
    const Real eps2 = Real(0.01) * HS * HS;
    const Real invC0  = Real(1.0) / C0;
    const Real invC02 = Real(1.0) / (C0 * C0);
    const Real b3     = Real(3.0) * invC0;
    const Real qC0    = Real(0.25) * C0;

    const Real xp0 = s.x[p], xp1 = s.x[p + N], xp2 = s.x[p + 2 * N];
    const Real vi0 = s.vx[p], vi1 = s.vx[N + p], vi2 = s.vx[2 * N + p];
    const Real rhoi = s.rho[p], irhoi = s.irho[p], pi_ = s.p[p];

    Real adrho = Real(0.0), ad0 = Real(0.0), ad1 = Real(0.0), ad2 = Real(0.0);
    Real av0 = Real(0.0), av1 = Real(0.0), av2 = Real(0.0), asurf = Real(0.0);

    auto pair = [&](int q, Real d0, Real d1, Real d2, Real r2)
    {
        const Real r2e = r2 + Real(0.0000000000001);        // 1e-13, as in 1.cpp
        const Real dr = sqrt(r2e);
        const Real qk = dr * invh;
        const Real tt = Real(2.0) - qk;
        const Real wc = (qk <= Real(2.0)) ? C5 * qk * (tt * tt * tt) : Real(0.0);

        const Real volq = s.mass[q] * s.irho[q];
        const Real sfd = s.SurFD[(p > q) ? p : q];
        const Real eta = (fabs(sfd) < Real(2.5)) ? Real(0.001) : Real(0.00002);
        const Real dv0 = vi0 - s.vx[q];
        const Real dv1 = vi1 - s.vx[N + q];
        const Real dv2 = vi2 - s.vx[2 * N + q];

        Real PR, rhoR, dUraw;
        const bool fluid = s.itype[q] > Real(0.0);
        if (fluid)
        {
            PR    = s.p[q];
            rhoR  = s.rho[q];
            dUraw = dv0 * d0 + dv1 * d1 + dv2 * d2;         // (vi-vq).d, /dr later
        }
        else
        {
            PR    = pi_ + rhoi * GRAV * d2;                 // pi - rho g (z_q - z_p)
            rhoR  = Real(1000.0) + PR * invC02;
            dUraw = Real(-2.0) * (vi0 * s.vecn[q] + vi1 * s.vecn[N + q]
                                  + vi2 * s.vecn[2 * N + q]);
        }

        // one division gives 1/rhoSum, 1/dr and 1/(dr^2+eps2)
        const Real rhoSum = rhoi + rhoR;
        const Real den_v  = r2e + eps2;
        const Real R      = Real(1.0) / ((dr * den_v) * rhoSum);
        const Real inv_rs = R * (dr * den_v);               // 1/rhoSum
        const Real t1     = R * rhoSum;                     // 1/(dr*den_v)
        const Real pinv   = t1 * den_v;                     // 1/dr
        const Real inv_v  = t1 * dr;                        // 1/(dr^2+eps2)

        const Real dU    = fluid ? -(dUraw * pinv) : dUraw;  // UL - UR
        const Real ULmUM = Real(0.5) * dU - (pi_ - PR) * inv_rs * invC0;
        const Real beta  = sph_min(b3 * sph_max(dU, Real(0.0)), Real(1.0));
        const Real PM    = Real(0.5) * (pi_ + PR) + qC0 * rhoSum * dU * beta;

        const Real gk = wc * pinv;                          // gradW = gk * d
        const Real S  = gk * r2;                            // gradW . d

        adrho += Real(-2.0) * rhoi * volq * ULmUM * wc;
        const Real cg = Real(-2.0) * volq * irhoi * PM * gk;
        ad0 += cg * d0; ad1 += cg * d1; ad2 += cg * d2;

        const Real f = Real(2.0) * volq * eta * S * inv_v;
        av0 += f * dv0; av1 += f * dv1; av2 += f * dv2;

        asurf += S * volq;
    };

    if (USE_LIST)
    {
        const int n = s.fcnt[p];
        if (n >= 0)
        {
            for (int j = 0; j < n; ++j)
            {
                const int q = s.fnbr[(size_t)j * s.d.ntotal + p];
                const Real d0 = xp0 - s.x[q];
                const Real d1 = xp1 - s.x[N + q];
                const Real d2 = xp2 - s.x[2 * N + q];
                // the same expression sph_neigh2 uses, so the same bits
                const Real r2 = sph_add_rn(sph_add_rn(sph_mul_rn(d0, d0), sph_mul_rn(d1, d1)),
                                           sph_mul_rn(d2, d2));
                pair(q, d0, d1, d2, r2);
            }
        }
        else
        {
            sph_neigh2(s, p, xp0, xp1, xp2, pair);          // did not fit: fused
        }
    }
    else
    {
        sph_neigh2(s, p, xp0, xp1, xp2, pair);
    }

    s.drhodt[p]   = adrho;
    s.dvxdt[p]    = ad0;
    s.dvxdt[N + p]  = ad1;
    s.dvxdt[2 * N + p] = ad2;
    s.viscos[p]   = av0;
    s.viscos[N + p] = av1;
    s.viscos[2 * N + p] = av2;
    s.SurF[p]     = asurf;
}

// ---------------------------------------------------------------------------
// step 2.1 : solid side of pass 1  (deformation gradient + friction rate)
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_pass1_solid(SphPtrsT<Real> s, int base)
{
    const Real invh = sph_inv_h<Real>();

    const int p = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.nall) return;

    const Real xp0 = s.x[p], xp1 = s.x[N + p], xp2 = s.x[2 * N + p];
    const Real vi0 = s.vx[p], vi1 = s.vx[N + p], vi2 = s.vx[2 * N + p];
    const Real Xp0 = s.X0[p], Xp1 = s.X0[N + p], Xp2 = s.X0[2 * N + p];

    // fp32 build: accumulate H = F - I from displacement differences u[j]-u[i].
    // sum_j w (X0[j]-X0[i]) (x) (A g0) = I by construction of A, so the result is
    // the same tensor, but the float mantissa is spent on the strain (1e-5..1e-3)
    // instead of on the absolute position (ulp(x) ~ 6e-8 m = 6e-6 in F).
    // Not valid for a particle whose A fell back to the identity in k_gra_invert.
    const bool lowp = (sizeof(Real) < sizeof(double));
    const Real up0 = lowp ? s.u[p] : Real(0.0);
    const Real up1 = lowp ? s.u[N + p] : Real(0.0);
    const Real up2 = lowp ? s.u[2 * N + p] : Real(0.0);

    const Real Ap0 = s.A[p],         Ap1 = s.A[N + p],     Ap2 = s.A[2 * N + p];
    const Real Ap3 = s.A[3 * N + p], Ap4 = s.A[4 * N + p], Ap5 = s.A[5 * N + p];
    const Real Ap6 = s.A[6 * N + p], Ap7 = s.A[7 * N + p], Ap8 = s.A[8 * N + p];

    Real F11 = Real(0.0), F12 = Real(0.0), F13 = Real(0.0);
    Real F21 = Real(0.0), F22 = Real(0.0), F23 = Real(0.0);
    Real F31 = Real(0.0), F32 = Real(0.0), F33 = Real(0.0);
    Real D11 = Real(0.0), D12 = Real(0.0), D13 = Real(0.0);
    Real D21 = Real(0.0), D22 = Real(0.0), D23 = Real(0.0);
    Real D31 = Real(0.0), D32 = Real(0.0), D33 = Real(0.0);

    sph_neigh2(s, p, xp0, xp1, xp2, [&](int q, Real d0, Real d1, Real d2, Real)
    {
        if (s.itype[q] >= Real(-10.0)) return;              // solid/solid only

        // solidPack only covers [NTOTAL, NALL); particles outside that range can
        // still carry itype < -10 (they then have A = 0, exactly as in 1.cpp)
        const int kq = q - s.d.ntotal;
        const bool inpack = (kq >= 0) && (kq < s.d.solnum);
        const Real* pk = s.solidPack + (size_t)(inpack ? kq : 0) * 14;
        const Real qX0 = inpack ? pk[0] : s.X0[q];
        const Real qX1 = inpack ? pk[1] : s.X0[N + q];
        const Real qX2 = inpack ? pk[2] : s.X0[2 * N + q];
        const Real gX0 = Xp0 - qX0;
        const Real gX1 = Xp1 - qX1;
        const Real gX2 = Xp2 - qX2;
        const Real Dr = sqrt(gX0 * gX0 + gX1 * gX1 + gX2 * gX2);
        const Real pinvDr = Real(1.0) / Dr;
        const Real qr = Dr * invh;
        const Real wc = sph_wq<Real>(qr);
        const Real g0x = wc * (gX0 * pinvDr);
        const Real g0y = wc * (gX1 * pinvDr);
        const Real g0z = wc * (gX2 * pinvDr);

        const Real vec0 = Ap0 * g0x + Ap1 * g0y + Ap2 * g0z;
        const Real vec1 = Ap3 * g0x + Ap4 * g0y + Ap5 * g0z;
        const Real vec2 = Ap6 * g0x + Ap7 * g0y + Ap8 * g0z;

        const Real w = inpack ? (pk[13] * (Real(1.0) / pk[12]))
                              : (s.mass[q] * (Real(1.0) / s.rho0[q]));
        const Real Dx0 = lowp ? (s.u[q] - up0)         : -d0;   // x[j]-x[i]  (fp32: u[j]-u[i])
        const Real Dx1 = lowp ? (s.u[N + q] - up1)     : -d1;
        const Real Dx2 = lowp ? (s.u[2 * N + q] - up2) : -d2;

        F11 += w * Dx0 * vec0;  F12 += w * Dx0 * vec1;  F13 += w * Dx0 * vec2;
        F21 += w * Dx1 * vec0;  F22 += w * Dx1 * vec1;  F23 += w * Dx1 * vec2;
        F31 += w * Dx2 * vec0;  F32 += w * Dx2 * vec1;  F33 += w * Dx2 * vec2;

        const Real dv0 = s.vx[q] - vi0;
        const Real dv1 = s.vx[N + q] - vi1;
        const Real dv2 = s.vx[2 * N + q] - vi2;

        D11 += w * dv0 * g0x;   D12 += w * dv0 * g0y;   D13 += w * dv0 * g0z;
        D21 += w * dv1 * g0x;   D22 += w * dv1 * g0y;   D23 += w * dv1 * g0z;
        D31 += w * dv2 * g0x;   D32 += w * dv2 * g0y;   D33 += w * dv2 * g0z;
    });

    s.F[p] = F11; s.F[N + p] = F12; s.F[2 * N + p] = F13;
    s.F[3 * N + p] = F21; s.F[4 * N + p] = F22; s.F[5 * N + p] = F23;
    s.F[6 * N + p] = F31; s.F[7 * N + p] = F32; s.F[8 * N + p] = F33;

    s.DFdt[p] = D11; s.DFdt[N + p] = D12; s.DFdt[2 * N + p] = D13;
    s.DFdt[3 * N + p] = D21; s.DFdt[4 * N + p] = D22; s.DFdt[5 * N + p] = D23;
    s.DFdt[6 * N + p] = D31; s.DFdt[7 * N + p] = D32; s.DFdt[8 * N + p] = D33;
}

// ---------------------------------------------------------------------------
// A3: same physics from the static pair list.  Per pair the cost drops to
//     F    += (x_q - x_p) (x) B
//     DFdt += (v_q - v_p) (x) G
// because G and B already carry the reference-configuration weight, the kernel
// gradient and the gradient-correction matrices.  No distance test, no sqrt.
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_pass1_solid_static(SphPtrsT<Real> s)
{
    const int p = s.d.ntotal + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.nall) return;

    const bool lowp = (sizeof(Real) < sizeof(double));
    const Real xp0 = s.x[p], xp1 = s.x[N + p], xp2 = s.x[2 * N + p];
    const Real vi0 = s.vx[p], vi1 = s.vx[N + p], vi2 = s.vx[2 * N + p];
    const Real up0 = lowp ? s.u[p] : Real(0.0);
    const Real up1 = lowp ? s.u[N + p] : Real(0.0);
    const Real up2 = lowp ? s.u[2 * N + p] : Real(0.0);

    Real F11 = Real(0.0), F12 = Real(0.0), F13 = Real(0.0);
    Real F21 = Real(0.0), F22 = Real(0.0), F23 = Real(0.0);
    Real F31 = Real(0.0), F32 = Real(0.0), F33 = Real(0.0);
    Real D11 = Real(0.0), D12 = Real(0.0), D13 = Real(0.0);
    Real D21 = Real(0.0), D22 = Real(0.0), D23 = Real(0.0);
    Real D31 = Real(0.0), D32 = Real(0.0), D33 = Real(0.0);

    const int k = p - s.d.ntotal;
    const int n = s.sCnt[k];
    const size_t L = (size_t)s.sw;
    for (int j = 0; j < n; ++j)
    {
        const size_t e = (size_t)j * s.d.solnum + k;
        const int q = s.sQ[e];
        const Real bx = s.sB[e], by = s.sB[L + e], bz = s.sB[2 * L + e];
        const Real gx = s.sG[e], gy = s.sG[L + e], gz = s.sG[2 * L + e];

        const Real Dx0 = lowp ? (s.u[q] - up0)         : (s.x[q] - xp0);
        const Real Dx1 = lowp ? (s.u[N + q] - up1)     : (s.x[N + q] - xp1);
        const Real Dx2 = lowp ? (s.u[2 * N + q] - up2) : (s.x[2 * N + q] - xp2);
        const Real dv0 = s.vx[q] - vi0;
        const Real dv1 = s.vx[N + q] - vi1;
        const Real dv2 = s.vx[2 * N + q] - vi2;

        F11 += Dx0 * bx;  F12 += Dx0 * by;  F13 += Dx0 * bz;
        F21 += Dx1 * bx;  F22 += Dx1 * by;  F23 += Dx1 * bz;
        F31 += Dx2 * bx;  F32 += Dx2 * by;  F33 += Dx2 * bz;

        D11 += dv0 * gx;  D12 += dv0 * gy;  D13 += dv0 * gz;
        D21 += dv1 * gx;  D22 += dv1 * gy;  D23 += dv1 * gz;
        D31 += dv2 * gx;  D32 += dv2 * gy;  D33 += dv2 * gz;
    }

    s.F[p] = F11; s.F[N + p] = F12; s.F[2 * N + p] = F13;
    s.F[3 * N + p] = F21; s.F[4 * N + p] = F22; s.F[5 * N + p] = F23;
    s.F[6 * N + p] = F31; s.F[7 * N + p] = F32; s.F[8 * N + p] = F33;

    s.DFdt[p] = D11; s.DFdt[N + p] = D12; s.DFdt[2 * N + p] = D13;
    s.DFdt[3 * N + p] = D21; s.DFdt[4 * N + p] = D22; s.DFdt[5 * N + p] = D23;
    s.DFdt[6 * N + p] = D31; s.DFdt[7 * N + p] = D32; s.DFdt[8 * N + p] = D33;
}

// ---------------------------------------------------------------------------
// step 2.3 : solid constitution (needs the freshly reduced F / DFdt)
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_constitution(SphPtrsT<Real> s, int base)
{
    const int p = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.nall) return;

    // Evaluated in double for BOTH precisions (C = calculation type).  For Real =
    // double nothing changes.  For Real = float: the law is ill conditioned
    // (lambda*log10(J) and F - F^-T with J ~ 1), it is only ~300 operations for each
    // of the 60 k solid particles, and s.F holds F - I there (see k_pass1_solid), so
    // the identity is restored here, in double.
    typedef double C;
    const bool lowp = (sizeof(Real) < sizeof(double));

    const C MU = C(SPH_MUYOU);
    const C LA = C(SPH_LAMD);

    const C F11 = lowp ? C(1.0) + C(s.F[p]) : C(s.F[p]);
    const C F12 = s.F[N + p],       F13 = s.F[2 * N + p];
    const C F21 = s.F[3 * N + p];
    const C F22 = lowp ? C(1.0) + C(s.F[4 * N + p]) : C(s.F[4 * N + p]);
    const C F23 = s.F[5 * N + p];
    const C F31 = s.F[6 * N + p], F32 = s.F[7 * N + p];
    const C F33 = lowp ? C(1.0) + C(s.F[8 * N + p]) : C(s.F[8 * N + p]);

    const C A11 = s.A[p],       A12 = s.A[N + p],       A13 = s.A[2 * N + p];
    const C A21 = s.A[3 * N + p], A22 = s.A[4 * N + p], A23 = s.A[5 * N + p];
    const C A31 = s.A[6 * N + p], A32 = s.A[7 * N + p], A33 = s.A[8 * N + p];

    const C d11 = s.DFdt[p],       d12 = s.DFdt[N + p],       d13 = s.DFdt[2 * N + p];
    const C d21 = s.DFdt[3 * N + p], d22 = s.DFdt[4 * N + p], d23 = s.DFdt[5 * N + p];
    const C d31 = s.DFdt[6 * N + p], d32 = s.DFdt[7 * N + p], d33 = s.DFdt[8 * N + p];

    // DFdtB = DFdt * A
    const C b11 = d11 * A11 + d12 * A21 + d13 * A31;
    const C b12 = d11 * A12 + d12 * A22 + d13 * A32;
    const C b13 = d11 * A13 + d12 * A23 + d13 * A33;
    const C b21 = d21 * A11 + d22 * A21 + d23 * A31;
    const C b22 = d21 * A12 + d22 * A22 + d23 * A32;
    const C b23 = d21 * A13 + d22 * A23 + d23 * A33;
    const C b31 = d31 * A11 + d32 * A21 + d33 * A31;
    const C b32 = d31 * A12 + d32 * A22 + d33 * A32;
    const C b33 = d31 * A13 + d32 * A23 + d33 * A33;

    const C Jacb = F11 * F22 * F33 + F31 * F12 * F23 + F21 * F32 * F13
                    - F31 * F22 * F13 - F11 * F32 * F23 - F12 * F21 * F33 + C(0.0);

    const C rho0 = s.rho0[p];
    s.rho[p] = Real((fabs(Jacb) < C(0.000001)) ? C(1000.0) : rho0 / Jacb);
    s.irho[p] = Real(1.0) / s.rho[p];

    const C paii = C(0.25) * rho0 * C(1000.0) * C(SPH_HSML);

    const C SD11 = paii * (b11 * F11 + b21 * F21 + b31 * F31 + F11 * b11 + F21 * b21 + F31 * b31);
    const C SD12 = paii * (b11 * F12 + b21 * F22 + b31 * F32 + F11 * b12 + F21 * b22 + F31 * b32);
    const C SD13 = paii * (b11 * F13 + b21 * F23 + b31 * F33 + F11 * b13 + F21 * b23 + F31 * b33);
    const C SD21 = paii * (b12 * F11 + b22 * F21 + b32 * F31 + F12 * b11 + F22 * b21 + F32 * b31);
    const C SD22 = paii * (b12 * F12 + b22 * F22 + b32 * F32 + F12 * b12 + F22 * b22 + F32 * b32);
    const C SD23 = paii * (b12 * F13 + b22 * F23 + b32 * F33 + F12 * b13 + F22 * b23 + F32 * b33);
    const C SD31 = paii * (b13 * F11 + b23 * F21 + b33 * F31 + F13 * b11 + F23 * b21 + F33 * b31);
    const C SD32 = paii * (b13 * F12 + b23 * F22 + b33 * F32 + F13 * b12 + F23 * b22 + F33 * b32);
    const C SD33 = paii * (b13 * F13 + b23 * F23 + b33 * F33 + F13 * b13 + F23 * b23 + F33 * b33);

    // F^-1 (cofactors / det)
    C n11 =  (F22 * F33 - F32 * F23) / Jacb;
    C n12 = -(F12 * F33 - F32 * F13) / Jacb;
    C n13 =  (F12 * F23 - F22 * F13) / Jacb;
    C n21 = -(F21 * F33 - F31 * F23) / Jacb;
    C n22 =  (F11 * F33 - F13 * F31) / Jacb;
    C n23 = -(F11 * F23 - F21 * F13) / Jacb;
    C n31 =  (F21 * F32 - F31 * F22) / Jacb;
    C n32 = -(F11 * F32 - F31 * F12) / Jacb;
    C n33 =  (F11 * F22 - F21 * F12) / Jacb;

    // PD = F * SD
    const C PD11 = F11 * SD11 + F12 * SD21 + F13 * SD31;
    const C PD12 = F11 * SD12 + F12 * SD22 + F13 * SD32;
    const C PD13 = F11 * SD13 + F12 * SD23 + F13 * SD33;
    const C PD21 = F21 * SD11 + F22 * SD21 + F23 * SD31;
    const C PD22 = F21 * SD12 + F22 * SD22 + F23 * SD32;
    const C PD23 = F21 * SD13 + F22 * SD23 + F23 * SD33;
    const C PD31 = F31 * SD11 + F32 * SD21 + F33 * SD31;
    const C PD32 = F31 * SD12 + F32 * SD22 + F33 * SD32;
    const C PD33 = F31 * SD13 + F32 * SD23 + F33 * SD33;

    // F^-T
    const C t11 = n11, t12 = n21, t13 = n31;
    const C t21 = n12, t22 = n22, t23 = n32;
    const C t31 = n13, t32 = n23, t33 = n33;

    const C lg = log10(Jacb);
    Real* pb = s.Pia + (size_t)(p - s.d.ntotal) * SPH_PIA_STRIDE;
    pb[0] = Real(MU * (F11 - t11) + LA * lg * t11 + PD11);
    pb[1] = Real(MU * (F12 - t12) + LA * lg * t12 + PD12);
    pb[2] = Real(MU * (F13 - t13) + LA * lg * t13 + PD13);
    pb[3] = Real(MU * (F21 - t21) + LA * lg * t21 + PD21);
    pb[4] = Real(MU * (F22 - t22) + LA * lg * t22 + PD22);
    pb[5] = Real(MU * (F23 - t23) + LA * lg * t23 + PD23);
    pb[6] = Real(MU * (F31 - t31) + LA * lg * t31 + PD31);
    pb[7] = Real(MU * (F32 - t32) + LA * lg * t32 + PD32);
    pb[8] = Real(MU * (F33 - t33) + LA * lg * t33 + PD33);
}

// ---------------------------------------------------------------------------
// step 2.4 : fluid side of pass 2 (fluid/solid coupling force)
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_pass2_fluid(SphPtrsT<Real> s)
{
    const Real invh = sph_inv_h<Real>();

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.ntotal) return;

    const Real xp0 = s.x[p], xp1 = s.x[N + p], xp2 = s.x[2 * N + p];

    // no solid anywhere in the 27-cell neighbourhood -> this pass has nothing to do
    {
        const int c = sph_cell(s, xp0, xp1, xp2);
        const int ix = c % s.d.nx, iy = (c / s.d.nx) % s.d.ny, iz = c / (s.d.nx * s.d.ny);
        int any = 0;
        for (int dz = -1; dz <= 1 && !any; ++dz)
        {
            const int jz = iz + dz; if (jz < 0 || jz >= s.d.nz) continue;
            for (int dy = -1; dy <= 1 && !any; ++dy)
            {
                const int jy = iy + dy; if (jy < 0 || jy >= s.d.ny) continue;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int jx = ix + dx; if (jx < 0 || jx >= s.d.nx) continue;
                    if (s.cellSolid[(jz * s.d.ny + jy) * s.d.nx + jx]) { any = 1; break; }
                }
            }
        }
        if (!any) { s.SForce[p] = Real(0.0); s.SForce[N + p] = Real(0.0); s.SForce[2 * N + p] = Real(0.0); return; }
    }

    const Real pi_ = s.p[p], rhoi = s.rho[p];

    Real a0 = Real(0.0), a1 = Real(0.0), a2 = Real(0.0);

    sph_neigh2(s, p, xp0, xp1, xp2, [&](int q, Real d0, Real d1, Real d2, Real r2)
    {
        if (s.itype[q] >= Real(-10.0)) return;              // needs a solid partner

        const Real dr = sqrt(r2);
        const Real pinv = Real(1.0) / dr;
        const Real qk = dr * invh;
        const Real mq = s.mass[q], rhoq = s.rho[q];

        const Real dotN = d0 * s.vecn[q] + d1 * s.vecn[N + q] + d2 * s.vecn[2 * N + q];
        const Real dotP = (Real(0.0) - s.PIJW[p]) * s.vecn[q]
                        + (Real(0.0) - s.PIJW[N + p]) * s.vecn[N + q]
                        + (Real(0.0) - s.PIJW[2 * N + p]) * s.vecn[2 * N + q];
        const Real PR = pi_ + rhoi * sph_max(Real(0.0), dotP) * dotN * dotN;
        const Real coef = (pi_ * rhoq + PR * rhoi) / (rhoi + rhoq);
        const Real k = -Real(2.0) * (mq / (rhoi * rhoq)) * coef;

        const Real wc = sph_wq<Real>(qk);
        a0 += k * (wc * (d0 * pinv));
        a1 += k * (wc * (d1 * pinv));
        a2 += k * (wc * (d2 * pinv));
    });

    s.SForce[p] = a0;
    s.SForce[N + p] = a1;
    s.SForce[2 * N + p] = a2;
}

// ---------------------------------------------------------------------------
// step 2.4 : solid side of pass 2 (PIJW from solid neighbours, SForce from fluids)
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_pass2_solid(SphPtrsT<Real> s, int base)
{
    const Real invh = sph_inv_h<Real>();

    const int p = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.nall) return;

    const Real xp0 = s.x[p], xp1 = s.x[N + p], xp2 = s.x[2 * N + p];
    const Real rhoi = s.rho[p];
    const Real Xp0 = s.X0[p], Xp1 = s.X0[N + p], Xp2 = s.X0[2 * N + p];

    // Self-side data does not change inside the neighbour loop.  Without this the
    // compiler cannot hoist the loads (it cannot prove Pia/PIJW do not alias), and
    // the loop ended up issuing 64 scattered 64-bit loads per neighbour.
    typedef typename SphVec2T<Real>::type V2;
    const Real* pbp = s.Pia + (size_t)(p - s.d.ntotal) * SPH_PIA_STRIDE;
    const V2* vp = reinterpret_cast<const V2*>(pbp);
    const Real P00 = vp[0].x, P01 = vp[0].y;
    const Real P02 = vp[1].x, P10 = vp[1].y;
    const Real P11 = vp[2].x, P12 = vp[2].y;
    const Real P20 = vp[3].x, P21 = vp[3].y;
    const Real P22 = vp[4].x;
    const Real Ap0 = s.A[p],         Ap1 = s.A[N + p],     Ap2 = s.A[2 * N + p];
    const Real Ap3 = s.A[3 * N + p], Ap4 = s.A[4 * N + p], Ap5 = s.A[5 * N + p];
    const Real Ap6 = s.A[6 * N + p], Ap7 = s.A[7 * N + p], Ap8 = s.A[8 * N + p];
    const Real vn0 = s.vecn[p], vn1 = s.vecn[N + p], vn2 = s.vecn[2 * N + p];

    Real P0 = Real(0.0), P1 = Real(0.0), P2 = Real(0.0);
    Real a0 = Real(0.0), a1 = Real(0.0), a2 = Real(0.0);

    sph_neigh2(s, p, xp0, xp1, xp2, [&](int q, Real d0, Real d1, Real d2, Real r2)
    {
        const Real mq = s.mass[q], rhoq = s.rho[q];

        const int kq = q - s.d.ntotal;
        const bool inpack = (kq >= 0) && (kq < s.d.solnum);
        const Real* pk = s.solidPack + (size_t)(inpack ? kq : 0) * 14;
        const Real qX0 = inpack ? pk[0] : s.X0[q];
        const Real qX1 = inpack ? pk[1] : s.X0[N + q];
        const Real qX2 = inpack ? pk[2] : s.X0[2 * N + q];
        const Real gX0 = Xp0 - qX0;
        const Real gX1 = Xp1 - qX1;
        const Real gX2 = Xp2 - qX2;
        const Real Dr = sqrt(gX0 * gX0 + gX1 * gX1 + gX2 * gX2);
        const Real pinvDr = Real(1.0) / Dr;
        const Real qr = Dr * invh;
        const Real wc = sph_wq<Real>(qr);
        const Real g0x = wc * (gX0 * pinvDr);
        const Real g0y = wc * (gX1 * pinvDr);
        const Real g0z = wc * (gX2 * pinvDr);

        if (s.itype[q] < Real(-10.0))
        {
            // ---------------- solid / solid : PIJW ----------------
            Real vj0, vj1, vj2;
            const Real vi0 = Ap0 * g0x + Ap1 * g0y + Ap2 * g0z;
            const Real vi1 = Ap3 * g0x + Ap4 * g0y + Ap5 * g0z;
            const Real vi2 = Ap6 * g0x + Ap7 * g0y + Ap8 * g0z;
            // A of the neighbour lives in its static pack (9 doubles at +3);
            // outside [NTOTAL, NALL) it is 0 in 1.cpp, so the SoA array is right
            if (inpack)
            {
                const Real* aq = pk + 3;
                vj0 = aq[0] * (-g0x) + aq[1] * (-g0y) + aq[2] * (-g0z);
                vj1 = aq[3] * (-g0x) + aq[4] * (-g0y) + aq[5] * (-g0z);
                vj2 = aq[6] * (-g0x) + aq[7] * (-g0y) + aq[8] * (-g0z);
            }
            else
            {
                sph_matvec(s, s.A, q, -g0x, -g0y, -g0z, vj0, vj1, vj2);
            }

            const Real Pi0 = P00 * vi0 + P01 * vi1 + P02 * vi2;
            const Real Pi1 = P10 * vi0 + P11 * vi1 + P12 * vi2;
            const Real Pi2 = P20 * vi0 + P21 * vi1 + P22 * vi2;

            Real Pj0 = Real(0.0), Pj1 = Real(0.0), Pj2 = Real(0.0);
            if (inpack)
            {
                const V2* vq = reinterpret_cast<const V2*>(s.Pia + (size_t)kq * SPH_PIA_STRIDE);
                const Real q00 = vq[0].x, q01 = vq[0].y;
                const Real q02 = vq[1].x, q10 = vq[1].y;
                const Real q11 = vq[2].x, q12 = vq[2].y;
                const Real q20 = vq[3].x, q21 = vq[3].y;
                const Real q22 = vq[4].x;
                Pj0 = q00 * vj0 + q01 * vj1 + q02 * vj2;
                Pj1 = q10 * vj0 + q11 * vj1 + q12 * vj2;
                Pj2 = q20 * vj0 + q21 * vj1 + q22 * vj2;
            }

            const Real w = mq / (rhoi * rhoq);
            P0 += w * (Pi0 - Pj0);
            P1 += w * (Pi1 - Pj1);
            P2 += w * (Pi2 - Pj2);
        }
        else if (s.itype[q] > Real(0.0))
        {
            // ---------------- fluid / solid : SForce on the solid side -------
            const Real dr = sqrt(r2);
            const Real pinv = Real(1.0) / dr;
            const Real qk = dr * invh;
            const Real pq = s.p[q];

            const Real dotN = -d0 * vn0 - d1 * vn1 - d2 * vn2;
            const Real dotP = (Real(0.0) - s.PIJW[q]) * vn0
                            + (Real(0.0) - s.PIJW[N + q]) * vn1
                            + (Real(0.0) - s.PIJW[2 * N + q]) * vn2;
            const Real PR = pq + rhoq * sph_max(Real(0.0), dotP) * dotN * dotN;
            const Real coef = (pq * rhoi + PR * rhoq) / (rhoq + rhoi);
            const Real k = -Real(2.0) * (mq / (rhoq * rhoi)) * coef;

            const Real wf = sph_wq<Real>(qk);
            a0 += k * (wf * (d0 * pinv));
            a1 += k * (wf * (d1 * pinv));
            a2 += k * (wf * (d2 * pinv));
        }
    });

    s.PIJW[p] = P0;
    s.PIJW[N + p] = P1;
    s.PIJW[2 * N + p] = P2;
    s.SForce[p] = a0;
    s.SForce[N + p] = a1;
    s.SForce[2 * N + p] = a2;
}

// ---------------------------------------------------------------------------
// A3: pass 2 from the static list.
//
// The solid/solid part factors the q-independent terms out of the sum:
//   PIJW_p = (Pia_p A_p (sum_q vol_q g0) - sum_q vol_q Pia_q A_q(-g0)) / rho_p
// with vol_q = m_q/rho_q, and vol_q g0 = (rho0_q/rho_q) G, vol_q A_q(-g0)
// = (rho0_q/rho_q) D.  So the per-pair work is 3 FMA for the first sum and a
// 3x3 matvec for the second.
//
// The fluid/solid part still needs the live cell list (fluid moves), but only
// for solid particles that have fluid in their 27 cells -- cellFluid, the same
// trick pass2_fluid uses with cellSolid.
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_pass2_solid_static(SphPtrsT<Real> s)
{
    const Real invh = sph_inv_h<Real>();

    const int p = s.d.ntotal + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.nall) return;

    const Real xp0 = s.x[p], xp1 = s.x[N + p], xp2 = s.x[2 * N + p];
    const Real rhoi = s.rho[p];
    const Real vn0 = s.vecn[p], vn1 = s.vecn[N + p], vn2 = s.vecn[2 * N + p];

    typedef typename SphVec2T<Real>::type V2;
    const Real* pbp = s.Pia + (size_t)(p - s.d.ntotal) * SPH_PIA_STRIDE;
    const V2* vp = reinterpret_cast<const V2*>(pbp);
    const Real P00 = vp[0].x, P01 = vp[0].y;
    const Real P02 = vp[1].x, P10 = vp[1].y;
    const Real P11 = vp[2].x, P12 = vp[2].y;
    const Real P20 = vp[3].x, P21 = vp[3].y;
    const Real P22 = vp[4].x;
    const Real Ap0 = s.A[p],         Ap1 = s.A[N + p],     Ap2 = s.A[2 * N + p];
    const Real Ap3 = s.A[3 * N + p], Ap4 = s.A[4 * N + p], Ap5 = s.A[5 * N + p];
    const Real Ap6 = s.A[6 * N + p], Ap7 = s.A[7 * N + p], Ap8 = s.A[8 * N + p];

    // ---------------- solid / solid : PIJW from the static list ----------------
    Real u10 = Real(0.0), u11 = Real(0.0), u12 = Real(0.0);   // sum vol_q g0
    Real t0 = Real(0.0), t1 = Real(0.0), t2 = Real(0.0);      // sum vol_q Pia_q A_q(-g0)

    const int k = p - s.d.ntotal;
    const int n = s.sCnt[k];
    const size_t L = (size_t)s.sw;
    for (int j = 0; j < n; ++j)
    {
        const size_t e = (size_t)j * s.d.solnum + k;
        const int q = s.sQ[e];
        const Real kq3 = s.rho0[q] * s.irho[q];               // rho0_q / rho_q
        u10 += kq3 * s.sG[e];
        u11 += kq3 * s.sG[L + e];
        u12 += kq3 * s.sG[2 * L + e];

        const int kq = q - s.d.ntotal;
        if (kq < 0 || kq >= s.d.solnum) continue;             // no Pia for those
        const Real f0 = kq3 * s.sD[e], f1 = kq3 * s.sD[L + e], f2 = kq3 * s.sD[2 * L + e];
        const V2* vq = reinterpret_cast<const V2*>(s.Pia + (size_t)kq * SPH_PIA_STRIDE);
        const Real q00 = vq[0].x, q01 = vq[0].y;
        const Real q02 = vq[1].x, q10 = vq[1].y;
        const Real q11 = vq[2].x, q12 = vq[2].y;
        const Real q20 = vq[3].x, q21 = vq[3].y;
        const Real q22 = vq[4].x;
        t0 += q00 * f0 + q01 * f1 + q02 * f2;
        t1 += q10 * f0 + q11 * f1 + q12 * f2;
        t2 += q20 * f0 + q21 * f1 + q22 * f2;
    }

    const Real sv0 = Ap0 * u10 + Ap1 * u11 + Ap2 * u12;       // A_p * (sum vol_q g0)
    const Real sv1 = Ap3 * u10 + Ap4 * u11 + Ap5 * u12;
    const Real sv2 = Ap6 * u10 + Ap7 * u11 + Ap8 * u12;
    s.PIJW[p]         = (P00 * sv0 + P01 * sv1 + P02 * sv2 - t0) / rhoi;
    s.PIJW[N + p]     = (P10 * sv0 + P11 * sv1 + P12 * sv2 - t1) / rhoi;
    s.PIJW[2 * N + p] = (P20 * sv0 + P21 * sv1 + P22 * sv2 - t2) / rhoi;

    // ---------------- fluid / solid : SForce, live cell list -------------------
    Real a0x = Real(0.0), a1x = Real(0.0), a2x = Real(0.0);
    {
        const int c = sph_cell(s, xp0, xp1, xp2);
        const int ix = c % s.d.nx, iy = (c / s.d.nx) % s.d.ny, iz = c / (s.d.nx * s.d.ny);
        int any = 0;
        for (int dz = -1; dz <= 1 && !any; ++dz)
        {
            const int jz = iz + dz; if (jz < 0 || jz >= s.d.nz) continue;
            for (int dy = -1; dy <= 1 && !any; ++dy)
            {
                const int jy = iy + dy; if (jy < 0 || jy >= s.d.ny) continue;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int jx = ix + dx; if (jx < 0 || jx >= s.d.nx) continue;
                    if (s.cellFluid[(jz * s.d.ny + jy) * s.d.nx + jx]) { any = 1; break; }
                }
            }
        }
        if (!any) { s.SForce[p] = Real(0.0); s.SForce[N + p] = Real(0.0); s.SForce[2 * N + p] = Real(0.0); return; }
    }

    sph_neigh2(s, p, xp0, xp1, xp2, [&](int q, Real d0, Real d1, Real d2, Real r2)
    {
        if (s.itype[q] <= Real(0.0)) return;                  // needs a fluid partner

        const Real dr = sqrt(r2);
        const Real pinv = Real(1.0) / dr;
        const Real qk = dr * invh;
        const Real mq = s.mass[q], rhoq = s.rho[q];
        const Real pq = s.p[q];

        const Real dotN = -d0 * vn0 - d1 * vn1 - d2 * vn2;
        const Real dotP = (Real(0.0) - s.PIJW[q]) * vn0
                        + (Real(0.0) - s.PIJW[N + q]) * vn1
                        + (Real(0.0) - s.PIJW[2 * N + q]) * vn2;
        const Real PR = pq + rhoq * sph_max(Real(0.0), dotP) * dotN * dotN;
        const Real coef = (pq * rhoi + PR * rhoq) / (rhoq + rhoi);
        const Real kk = -Real(2.0) * (mq / (rhoq * rhoi)) * coef;

        const Real wf = sph_wq<Real>(qk);
        a0x += kk * (wf * (d0 * pinv));
        a1x += kk * (wf * (d1 * pinv));
        a2x += kk * (wf * (d2 * pinv));
    });

    s.SForce[p] = a0x;
    s.SForce[N + p] = a1x;
    s.SForce[2 * N + p] = a2x;
}

// ---------------------------------------------------------------------------
// A4: refresh the float position copy the neighbour test pre-filters with.
// Called once after load_input; k_update_pos keeps it current from then on.
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_copy_xf(SphPtrsT<Real> s)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.d.num) return;
    s.xf[i]         = (float)s.x[i];
    s.xf[N + i]     = (float)s.x[N + i];
    s.xf[2 * N + i] = (float)s.x[2 * N + i];
}

// ---------------------------------------------------------------------------
// step 3 : time integration
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_update_fluid(SphPtrsT<Real> s)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.d.ntotal) return;

    const Real dt = Real(SPH_DT);
    const Real GRAV = Real(9.8);

    Real r;
    if (sizeof(Real) < sizeof(double))
    {
        // fp32: dt*drhodt is routinely below half an ulp of rho (3e-5 at rho ~ 1000),
        // a plain sum drops it; carry the rounding error to the next step instead
        const Real y = dt * s.drhodt[i] - s.rhoc[i];
        r = s.rho[i] + y;
        s.rhoc[i] = (r - s.rho[i]) - y;
        if (r < Real(960.0)) { r = Real(960.0); s.rhoc[i] = Real(0.0); }
    }
    else
    {
        r = s.rho[i] + dt * s.drhodt[i];
        if (r < Real(960.0)) r = Real(960.0);
    }
    s.rho[i] = r;

    Real v0 = s.vx[i]         + dt * (s.dvxdt[i]         + s.viscos[i])         + dt * s.SForce[i];
    Real v1 = s.vx[N + i]     + dt * (s.dvxdt[N + i]     + s.viscos[N + i])     + dt * s.SForce[N + i];
    Real v2 = s.vx[2 * N + i] + dt * (s.dvxdt[2 * N + i] + s.viscos[2 * N + i] - GRAV)
                                                                               + dt * s.SForce[2 * N + i];

    const Real absv = sqrt(v0 * v0 + v1 * v1 + v2 * v2 + Real(0.00001));
    if (absv > Real(SPH_MAXV))
    {
        v0 = Real(SPH_MAXV) * v0 / absv;
        v1 = Real(SPH_MAXV) * v1 / absv;
        v2 = Real(SPH_MAXV) * v2 / absv;
    }
    // 1.cpp:186 -- reflect the vertical velocity at the free surface.
    // Same place in the sequence: after the MaxV clamp, before the store.
    if (s.x[2 * N + i] > Real(1.0)) v2 = -v2;
    s.vx[i] = v0; s.vx[N + i] = v1; s.vx[2 * N + i] = v2;
}

template <class Real>
__global__ void k_update_solid(SphPtrsT<Real> s, int base)
{
    const int i = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.d.nall) return;

    const Real dt = Real(SPH_DT);
    const Real GRAV = Real(9.8);

    Real v0 = s.vx[i]         + dt * s.PIJW[i]         + dt * s.SForce[i];
    Real v1 = s.vx[N + i]     + dt * s.PIJW[N + i]     + dt * s.SForce[N + i];
    Real v2 = s.vx[2 * N + i] + dt * s.PIJW[2 * N + i] + dt * s.SForce[2 * N + i] - dt * GRAV;

    // New case (1.cpp): the old test-only box clamp is commented out there
    // (it froze the plate); the solid now deforms freely.
    s.vx[i] = v0; s.vx[N + i] = v1; s.vx[2 * N + i] = v2;
}

template <class Real>
__global__ void k_update_pos(SphPtrsT<Real> s)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.d.num) return;
    const Real dt = Real(SPH_DT);
    if (sizeof(Real) < sizeof(double))
    {
        // fp32: dt*v (1e-9..1e-6 m) is at or below half an ulp of x (1.5e-8..3e-8 m);
        // a plain sum freezes slow particles.  Compensated (Kahan) sum; solid-type
        // particles integrate the displacement u and rebuild x = X0 + u, which is
        // only needed for the neighbour search and the coupling geometry.
        const bool solid = s.itype[i] < Real(-10.0);
        Real* b = solid ? s.u : s.x;
        for (int c = 0; c < 3; ++c)
        {
            const int j = c * N + i;
            const Real y = dt * s.vx[j] - s.xc[j];
            const Real tsum = b[j] + y;
            s.xc[j] = (tsum - b[j]) - y;
            b[j] = tsum;
            if (solid) s.x[j] = s.X0[j] + tsum;
            s.xf[j] = (float)s.x[j];
        }
        return;
    }
    s.x[i]         = s.x[i]         + dt * s.vx[i];
    s.x[N + i]     = s.x[N + i]     + dt * s.vx[N + i];
    s.x[2 * N + i] = s.x[2 * N + i] + dt * s.vx[2 * N + i];
    s.xf[i]         = (float)s.x[i];
    s.xf[N + i]     = (float)s.x[N + i];
    s.xf[2 * N + i] = (float)s.x[2 * N + i];
}

// ---------------------------------------------------------------------------
// initialisation: NorVector (surface normals) and GraCorrec (correction matrix)
// ---------------------------------------------------------------------------
template <class Real>
__global__ void k_nor_sumvecn(SphPtrsT<Real> s, Real* __restrict__ sumvecn)
{
    const Real invh = sph_inv_h<Real>();

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.num) return;
    if (s.itype[p] >= Real(0.0)) return;

    const Real xp0 = s.x[p], xp1 = s.x[N + p], xp2 = s.x[2 * N + p];
    Real a0 = Real(0.0), a1 = Real(0.0), a2 = Real(0.0);

    sph_neigh2(s, p, xp0, xp1, xp2, [&](int q, Real d0, Real d1, Real d2, Real r2)
    {
        if (s.itype[q] >= Real(0.0)) return;
        const Real dr = sqrt(r2 + Real(0.000000000001));    // 1e-12, as in NorVector
        const Real pinv = Real(1.0) / dr;
        const Real qk = dr * invh;
        const Real w = s.mass[q] / s.rho[q];
        const Real wc = sph_wq<Real>(qk);
        a0 -= w * (wc * (d0 * pinv));
        a1 -= w * (wc * (d1 * pinv));
        a2 -= w * (wc * (d2 * pinv));
    });

    sumvecn[p] = a0; sumvecn[N + p] = a1; sumvecn[2 * N + p] = a2;
}

template <class Real>
__global__ void k_nor_normalize(SphPtrsT<Real> s, const Real* __restrict__ sumvecn)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.d.num) return;
    if (i < s.d.ntotal)
    {
        s.vecn[i] = Real(0.0); s.vecn[N + i] = Real(0.0); s.vecn[2 * N + i] = Real(0.0);
    }
    else
    {
        const Real a0 = sumvecn[i], a1 = sumvecn[N + i], a2 = sumvecn[2 * N + i];
        const Real d = sqrt(a0 * a0 + a1 * a1 + a2 * a2 + Real(0.0000001));
        s.vecn[i] = a0 / d; s.vecn[N + i] = a1 / d; s.vecn[2 * N + i] = a2 / d;
    }
}

template <class Real>
__global__ void k_gra_correc(SphPtrsT<Real> s, Real* __restrict__ a, int base)
{
    const Real invh = sph_inv_h<Real>();

    const int p = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= s.d.nall) return;

    const Real Xp0 = s.X0[p], Xp1 = s.X0[N + p], Xp2 = s.X0[2 * N + p];
    Real a11 = Real(0.0), a12 = Real(0.0), a13 = Real(0.0);
    Real a21 = Real(0.0), a22 = Real(0.0), a23 = Real(0.0);
    Real a31 = Real(0.0), a32 = Real(0.0), a33 = Real(0.0);

    sph_neigh(s, p, s.x[p], s.x[N + p], s.x[2 * N + p],
              [&](int q, Real, Real, Real, Real)
    {
        if (s.itype[q] >= Real(-10.0)) return;
        const Real gX0 = Xp0 - s.X0[q];
        const Real gX1 = Xp1 - s.X0[N + q];
        const Real gX2 = Xp2 - s.X0[2 * N + q];
        const Real Dr = sqrt(gX0 * gX0 + gX1 * gX1 + gX2 * gX2);
        const Real pinvDr = Real(1.0) / Dr;
        const Real qr = Dr * invh;
        const Real w = s.mass[q] / s.rho0[q];
        // (X0_r[j]-X0_r[i]) = -gX_r  and dwdx(X0_c[i]-X0_c[j], Dr) = dwdx(gX_c, Dr)
        const Real q0 = -gX0, q1 = -gX1, q2 = -gX2;
        const Real wc = sph_wq<Real>(qr);
        const Real d0 = wc * (gX0 * pinvDr);
        const Real d1 = wc * (gX1 * pinvDr);
        const Real d2 = wc * (gX2 * pinvDr);
        a11 += w * q0 * d0; a12 += w * q0 * d1; a13 += w * q0 * d2;
        a21 += w * q1 * d0; a22 += w * q1 * d1; a23 += w * q1 * d2;
        a31 += w * q2 * d0; a32 += w * q2 * d1; a33 += w * q2 * d2;
    });

    a[p] = a11; a[N + p] = a12; a[2 * N + p] = a13;
    a[3 * N + p] = a21; a[4 * N + p] = a22; a[5 * N + p] = a23;
    a[6 * N + p] = a31; a[7 * N + p] = a32; a[8 * N + p] = a33;
}

template <class Real>
__global__ void k_gra_invert(SphPtrsT<Real> s, const Real* __restrict__ a, int base)
{
    const int i = base + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.d.nall) return;

    const Real a11 = a[i], a12 = a[N + i], a13 = a[2 * N + i];
    const Real a21 = a[3 * N + i], a22 = a[4 * N + i], a23 = a[5 * N + i];
    const Real a31 = a[6 * N + i], a32 = a[7 * N + i], a33 = a[8 * N + i];

    const Real Da = a11 * a22 * a33 + a31 * a12 * a23 + a21 * a32 * a13
                  - a31 * a22 * a13 - a11 * a32 * a23 - a12 * a21 * a33 + Real(0.0);

    Real A11 =  (a22 * a33 - a32 * a23) / Da;
    Real A12 = -(a12 * a33 - a32 * a13) / Da;
    Real A13 =  (a12 * a23 - a22 * a13) / Da;
    Real A21 = -(a21 * a33 - a31 * a23) / Da;
    Real A22 =  (a11 * a33 - a13 * a31) / Da;
    Real A23 = -(a11 * a23 - a21 * a13) / Da;
    Real A31 =  (a21 * a32 - a31 * a22) / Da;
    Real A32 = -(a11 * a32 - a31 * a12) / Da;
    Real A33 =  (a11 * a22 - a21 * a12) / Da;

    if (fabs(Da) < Real(0.000001))
    {
        A11 = Real(1.0); A12 = Real(0.0); A13 = Real(0.0);
        A21 = Real(0.0); A22 = Real(1.0); A23 = Real(0.0);
        A31 = Real(0.0); A32 = Real(0.0); A33 = Real(1.0);
        // C8: the fp32 displacement form (F - I) assumes a * = I, i.e. no
        // fallback anywhere; counted so the host can refuse/warn.
        atomicAdd(&sph_gra_fallbacks, 1);
    }

    s.A[i] = A11; s.A[N + i] = A12; s.A[2 * N + i] = A13;
    s.A[3 * N + i] = A21; s.A[4 * N + i] = A22; s.A[5 * N + i] = A23;
    s.A[6 * N + i] = A31; s.A[7 * N + i] = A32; s.A[8 * N + i] = A33;
}

// ---------------------------------------------------------------------------
// launch wrappers
// ---------------------------------------------------------------------------
static const int BLK = 256;

static inline int nblocks(int n) { return (n + BLK - 1) / BLK; }

// A3 switch, see sph.h: 0 = pre-A3 path (cell list walk every step)
int sph_use_static_solid = 1;
// A5 switch: 1 = factored pass1_fluid loop, 0 = literal transcription
int sph_fast_fluid_loop = 1;
// P1 switch: 0 = fused fp64 pass1_fluid (A/B comparison)
int sph_use_fluid_list = 1;

int gra_fallback_count()
{
    int n = 0;
    cudaMemcpyFromSymbol(&n, sph_gra_fallbacks, sizeof(int));
    return n;
}

template <class Real>
void launch_cell_hist(SphPtrsT<Real> s, int* d_count, int* d_solid, int* d_fluid, cudaStream_t st)
{
    k_cell_hist<Real><<<nblocks(s.d.num), BLK, 0, st>>>(s, d_count, d_solid, d_fluid);
}

template <class Real>
void launch_build_solid_pack(SphPtrsT<Real> s, cudaStream_t st)
{
    k_build_solid_pack<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s, s.d.ntotal);
}

template <class Real>
void launch_copy_xf(SphPtrsT<Real> s, cudaStream_t st)
{
    k_copy_xf<Real><<<nblocks(s.d.num), BLK, 0, st>>>(s);
}

template <class Real>
void launch_solid_pair_count(SphPtrsT<Real> s, int* d_cnt, cudaStream_t st)
{
    k_solid_pair_count<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s, d_cnt);
}

template <class Real>
void launch_solid_pair_fill(SphPtrsT<Real> s, int* d_cur, cudaStream_t st)
{
    k_solid_pair_fill<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s, d_cur);
}

template <class Real>
void launch_fluid_nbr_build(SphPtrsT<Real> s, cudaStream_t st)
{
    k_fluid_nbr_build<Real><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s);
}

template <class Real>
void launch_pass1_fluid_over_list(SphPtrsT<Real> s, cudaStream_t st)
{
    k_pass1_fluid_body<Real, true><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s);
}

void launch_cell_sort(const int* d_cellStart, int* d_sorted, int ncell, cudaStream_t st)
{
    k_cell_sort<<<(ncell + BLK - 1) / BLK, BLK, 0, st>>>(d_cellStart, d_sorted, ncell);
}

void launch_cell_scan(int* d_cellStart, int* d_count, int ncell, cudaStream_t st)
{
    k_cell_scan<<<1, 512, 0, st>>>(d_cellStart, d_count, ncell);
}

template <class Real>
void launch_cell_scatter(SphPtrsT<Real> s, int* d_cursor, cudaStream_t st)
{
    k_cell_scatter<Real><<<nblocks(s.d.num), BLK, 0, st>>>(s, d_cursor);
}

template <class Real>
void launch_count_pairs(SphPtrsT<Real> s, int* d_niac, cudaStream_t st)
{
    k_count_pairs<Real><<<nblocks(s.d.num), BLK, 0, st>>>(s, d_niac);
}

template <class Real>
void launch_nbr_count(SphPtrsT<Real> s, int* d_out, cudaStream_t st)
{
    k_nbr_count<Real><<<nblocks(s.d.num), BLK, 0, st>>>(s, d_out);
}

template <class Real>
void launch_nbr_list(SphPtrsT<Real> s, int target, int* d_out, int* d_n, cudaStream_t st)
{
    k_nbr_list<Real><<<1, 1, 0, st>>>(s, target, d_out, d_n);
}

template <class Real>
void launch_bench_loads(SphPtrsT<Real> s, Real* out, bool scatter, cudaStream_t st)
{
    if (scatter) k_bench_loads<Real, true><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s, out);
    else         k_bench_loads<Real, false><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s, out);
}

template <class Real>
void launch_prepare(SphPtrsT<Real> s, cudaStream_t st)
{
    k_prepare<Real><<<nblocks(s.d.num), BLK, 0, st>>>(s);
}

template <class Real>
void launch_pass1(SphPtrsT<Real> s, cudaStream_t st)
{
    launch_pass1_fluid(s, st);
    launch_pass1_solid(s, st);
}

template <class Real>
void launch_pass1_fluid(SphPtrsT<Real> s, cudaStream_t st)
{
    if (!sph_fast_fluid_loop)
    {
        k_pass1_fluid<Real><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s);
        return;
    }
    // P1: the list pays off only in fp64; fp32 is traversal bound and the fused
    // kernel won on both GPUs measured so far (see docs, P1/P7).
    if constexpr (sizeof(Real) >= sizeof(double))
    {
        if (sph_use_fluid_list && s.fwidth > 0)
        {
            k_fluid_nbr_build<Real><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s);
            k_pass1_fluid_body<Real, true><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s);
            return;
        }
        k_pass1_fluid_body<Real, false><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s);
    }
    else
    {
        k_pass1_fluid_body<Real, false><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s);
    }
}

template <class Real>
void launch_pass1_solid(SphPtrsT<Real> s, cudaStream_t st)
{
    if (sph_use_static_solid && s.npair > 0)
        k_pass1_solid_static<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s);
    else
        k_pass1_solid<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s, s.d.ntotal);
}

template <class Real>
void launch_pass2_fluid(SphPtrsT<Real> s, cudaStream_t st)
{
    k_pass2_fluid<Real><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s);
}

template <class Real>
void launch_pass2_solid(SphPtrsT<Real> s, cudaStream_t st)
{
    if (sph_use_static_solid && s.npair > 0)
        k_pass2_solid_static<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s);
    else
        k_pass2_solid<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s, s.d.ntotal);
}

template <class Real>
void launch_update_fluid(SphPtrsT<Real> s, cudaStream_t st)
{
    k_update_fluid<Real><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s);
}

template <class Real>
void launch_update_solid(SphPtrsT<Real> s, cudaStream_t st)
{
    k_update_solid<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s, s.d.ntotal);
}

template <class Real>
void launch_update_pos(SphPtrsT<Real> s, cudaStream_t st)
{
    k_update_pos<Real><<<nblocks(s.d.num), BLK, 0, st>>>(s);
}

template <class Real>
void launch_constitution(SphPtrsT<Real> s, cudaStream_t st)
{
    k_constitution<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s, s.d.ntotal);
}

template <class Real>
void launch_pass2(SphPtrsT<Real> s, cudaStream_t st)
{
    launch_pass2_fluid(s, st);
    launch_pass2_solid(s, st);
}

template <class Real>
void launch_update(SphPtrsT<Real> s, cudaStream_t st)
{
    k_update_fluid<Real><<<nblocks(s.d.ntotal), BLK, 0, st>>>(s);
    k_update_solid<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s, s.d.ntotal);
    k_update_pos<Real><<<nblocks(s.d.num), BLK, 0, st>>>(s);
}

template <class Real>
void launch_nor_sumvecn(SphPtrsT<Real> s, Real* d_sumvecn, cudaStream_t st)
{
    k_nor_sumvecn<Real><<<nblocks(s.d.num), BLK, 0, st>>>(s, d_sumvecn);
}

template <class Real>
void launch_nor_normalize(SphPtrsT<Real> s, Real* d_sumvecn, cudaStream_t st)
{
    k_nor_normalize<Real><<<nblocks(s.d.num), BLK, 0, st>>>(s, d_sumvecn);
}

template <class Real>
void launch_gra_correc(SphPtrsT<Real> s, Real* d_a, cudaStream_t st)
{
    k_gra_correc<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s, d_a, s.d.ntotal);
}

template <class Real>
void launch_gra_invert(SphPtrsT<Real> s, Real* d_a, cudaStream_t st)
{
    k_gra_invert<Real><<<nblocks(s.d.solnum), BLK, 0, st>>>(s, d_a, s.d.ntotal);
}

// ---------------------------------------------------------------------------
// explicit instantiations (guard with the CMake switches)
// ---------------------------------------------------------------------------
#define SPH_INSTANTIATE(Real)                                                          \
    template void launch_cell_hist<Real>(SphPtrsT<Real>, int*, int*, int*, cudaStream_t); \
    template void launch_build_solid_pack<Real>(SphPtrsT<Real>, cudaStream_t);            \
    template void launch_copy_xf<Real>(SphPtrsT<Real>, cudaStream_t);                    \
    template void launch_solid_pair_count<Real>(SphPtrsT<Real>, int*, cudaStream_t);     \
    template void launch_solid_pair_fill<Real>(SphPtrsT<Real>, int*, cudaStream_t);      \
    template void launch_fluid_nbr_build<Real>(SphPtrsT<Real>, cudaStream_t);            \
    template void launch_pass1_fluid_over_list<Real>(SphPtrsT<Real>, cudaStream_t);      \
    template void launch_cell_scatter<Real>(SphPtrsT<Real>, int*, cudaStream_t);        \
    template void launch_count_pairs<Real>(SphPtrsT<Real>, int*, cudaStream_t);         \
    template void launch_nbr_count<Real>(SphPtrsT<Real>, int*, cudaStream_t);           \
    template void launch_nbr_list<Real>(SphPtrsT<Real>, int, int*, int*, cudaStream_t); \
    template void launch_bench_loads<Real>(SphPtrsT<Real>, Real*, bool, cudaStream_t);   \
    template void launch_prepare<Real>(SphPtrsT<Real>, cudaStream_t);                   \
    template void launch_pass1<Real>(SphPtrsT<Real>, cudaStream_t);                     \
    template void launch_pass1_fluid<Real>(SphPtrsT<Real>, cudaStream_t);               \
    template void launch_pass1_solid<Real>(SphPtrsT<Real>, cudaStream_t);               \
    template void launch_pass2_fluid<Real>(SphPtrsT<Real>, cudaStream_t);               \
    template void launch_pass2_solid<Real>(SphPtrsT<Real>, cudaStream_t);               \
    template void launch_update_fluid<Real>(SphPtrsT<Real>, cudaStream_t);              \
    template void launch_update_solid<Real>(SphPtrsT<Real>, cudaStream_t);              \
    template void launch_update_pos<Real>(SphPtrsT<Real>, cudaStream_t);                \
    template void launch_constitution<Real>(SphPtrsT<Real>, cudaStream_t);              \
    template void launch_pass2<Real>(SphPtrsT<Real>, cudaStream_t);                     \
    template void launch_update<Real>(SphPtrsT<Real>, cudaStream_t);                    \
    template void launch_nor_sumvecn<Real>(SphPtrsT<Real>, Real*, cudaStream_t);        \
    template void launch_nor_normalize<Real>(SphPtrsT<Real>, Real*, cudaStream_t);      \
    template void launch_gra_correc<Real>(SphPtrsT<Real>, Real*, cudaStream_t);         \
    template void launch_gra_invert<Real>(SphPtrsT<Real>, Real*, cudaStream_t);

#if SPH_HAVE_FP64
SPH_INSTANTIATE(double)
#endif
#if SPH_HAVE_FP32
SPH_INSTANTIATE(float)
#endif
