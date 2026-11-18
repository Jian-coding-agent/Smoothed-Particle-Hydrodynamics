// ============================================================================
// Review experiments (appended to a scratch copy of sph_kernels.cu).
//   D: two-phase pass1_fluid -- record the neighbour list, then run the exact
//      same body over it (same order -> same bits, no divergent hit test)
//   E: slot-major (ELL) copy of the static solid pair lists
// ============================================================================
int sph_use_ell = 0;

template <class Real>
__global__ void k_fluid_nbr_build(SphPtrsT<Real> s, int maxn, int* __restrict__ over)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= SPH_NTOTAL) return;
    const Real xp0 = s.x[p], xp1 = s.x[p + N], xp2 = s.x[p + 2 * N];
    int k = 0;
    sph_neigh2(s, p, xp0, xp1, xp2, [&](int q, Real, Real, Real, Real)
    {
        if (k < maxn) s.fnbr[(size_t)k * SPH_NTOTAL + p] = q;
        ++k;
    });
    s.fcnt[p] = k;
    if (k > maxn) atomicAdd(over, 1);
}

// body copied verbatim from k_pass1_fluid_fast; only the neighbour source differs
template <class Real>
__global__ void P1_LB k_pass1_fluid_list(SphPtrsT<Real> s)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= SPH_NTOTAL) return;

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

    auto body = [&](int q, Real d0, Real d1, Real d2, Real r2)
    {
        const Real r2e = r2 + Real(0.0000000000001);
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
            dUraw = dv0 * d0 + dv1 * d1 + dv2 * d2;
        }
        else
        {
            PR    = pi_ + rhoi * GRAV * d2;
            rhoR  = Real(1000.0) + PR * invC02;
            dUraw = Real(-2.0) * (vi0 * s.vecn[q] + vi1 * s.vecn[N + q]
                                  + vi2 * s.vecn[2 * N + q]);
        }

        const Real rhoSum = rhoi + rhoR;
        const Real den_v  = r2e + eps2;
        const Real R      = Real(1.0) / ((dr * den_v) * rhoSum);
        const Real inv_rs = R * (dr * den_v);
        const Real t1     = R * rhoSum;
        const Real pinv   = t1 * den_v;
        const Real inv_v  = t1 * dr;

        const Real dU    = fluid ? -(dUraw * pinv) : dUraw;
        const Real ULmUM = Real(0.5) * dU - (pi_ - PR) * inv_rs * invC0;
        const Real beta  = sph_min(b3 * sph_max(dU, Real(0.0)), Real(1.0));
        const Real PM    = Real(0.5) * (pi_ + PR) + qC0 * rhoSum * dU * beta;

        const Real gk = wc * pinv;
        const Real S  = gk * r2;

        adrho += Real(-2.0) * rhoi * volq * ULmUM * wc;
        const Real cg = Real(-2.0) * volq * irhoi * PM * gk;
        ad0 += cg * d0; ad1 += cg * d1; ad2 += cg * d2;

        const Real f = Real(2.0) * volq * eta * S * inv_v;
        av0 += f * dv0; av1 += f * dv1; av2 += f * dv2;

        asurf += S * volq;
    };

    const int n = s.fcnt[p];
    for (int k = 0; k < n; ++k)
    {
        const int q = s.fnbr[(size_t)k * SPH_NTOTAL + p];
        const Real d0 = xp0 - s.x[q];
        const Real d1 = xp1 - s.x[N + q];
        const Real d2 = xp2 - s.x[2 * N + q];
        const Real r2 = sph_add_rn(sph_add_rn(sph_mul_rn(d0, d0), sph_mul_rn(d1, d1)),
                                   sph_mul_rn(d2, d2));
        body(q, d0, d1, d2, r2);
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

// ---- E: ELL versions of the A3 kernels (same per-particle order as CSR) ----
template <class Real>
__global__ void k_pass1_solid_ell(SphPtrsT<Real> s)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= SPH_SOLNUM) return;
    const int p = SPH_NTOTAL + k;
    const size_t S = SPH_SOLNUM, M = (size_t)s.ellMax * S;

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

    const int n = s.eCnt[k];
    for (int a = 0; a < n; ++a)
    {
        const size_t e = (size_t)a * S + k;
        const int q = s.eQ[e];
        const Real bx = s.eB[e], by = s.eB[M + e], bz = s.eB[2 * M + e];
        const Real gx = s.eG[e], gy = s.eG[M + e], gz = s.eG[2 * M + e];

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

template <class Real>
__global__ void k_pass2_solid_ell(SphPtrsT<Real> s)
{
    const Real invh = sph_inv_h<Real>();
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= SPH_SOLNUM) return;
    const int p = SPH_NTOTAL + k;
    const size_t S = SPH_SOLNUM, M = (size_t)s.ellMax * S;

    const Real xp0 = s.x[p], xp1 = s.x[N + p], xp2 = s.x[2 * N + p];
    const Real rhoi = s.rho[p];
    const Real vn0 = s.vecn[p], vn1 = s.vecn[N + p], vn2 = s.vecn[2 * N + p];

    typedef typename SphVec2T<Real>::type V2;
    const Real* pbp = s.Pia + (size_t)(p - SPH_NTOTAL) * SPH_PIA_STRIDE;
    const V2* vp = reinterpret_cast<const V2*>(pbp);
    const Real P00 = vp[0].x, P01 = vp[0].y;
    const Real P02 = vp[1].x, P10 = vp[1].y;
    const Real P11 = vp[2].x, P12 = vp[2].y;
    const Real P20 = vp[3].x, P21 = vp[3].y;
    const Real P22 = vp[4].x;
    const Real Ap0 = s.A[p],         Ap1 = s.A[N + p],     Ap2 = s.A[2 * N + p];
    const Real Ap3 = s.A[3 * N + p], Ap4 = s.A[4 * N + p], Ap5 = s.A[5 * N + p];
    const Real Ap6 = s.A[6 * N + p], Ap7 = s.A[7 * N + p], Ap8 = s.A[8 * N + p];

    Real u10 = Real(0.0), u11 = Real(0.0), u12 = Real(0.0);
    Real t0 = Real(0.0), t1 = Real(0.0), t2 = Real(0.0);

    const int n = s.eCnt[k];
    for (int a = 0; a < n; ++a)
    {
        const size_t e = (size_t)a * S + k;
        const int q = s.eQ[e];
        const Real kq3 = s.rho0[q] * s.irho[q];
        u10 += kq3 * s.eG[e];
        u11 += kq3 * s.eG[M + e];
        u12 += kq3 * s.eG[2 * M + e];

        const int kq = q - SPH_NTOTAL;
        if (kq < 0 || kq >= SPH_SOLNUM) continue;
        const Real f0 = kq3 * s.eD[e], f1 = kq3 * s.eD[M + e], f2 = kq3 * s.eD[2 * M + e];
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

    const Real sv0 = Ap0 * u10 + Ap1 * u11 + Ap2 * u12;
    const Real sv1 = Ap3 * u10 + Ap4 * u11 + Ap5 * u12;
    const Real sv2 = Ap6 * u10 + Ap7 * u11 + Ap8 * u12;
    s.PIJW[p]         = (P00 * sv0 + P01 * sv1 + P02 * sv2 - t0) / rhoi;
    s.PIJW[N + p]     = (P10 * sv0 + P11 * sv1 + P12 * sv2 - t1) / rhoi;
    s.PIJW[2 * N + p] = (P20 * sv0 + P21 * sv1 + P22 * sv2 - t2) / rhoi;

    // fluid/solid part unchanged from k_pass2_solid_static
    Real a0x = Real(0.0), a1x = Real(0.0), a2x = Real(0.0);
    {
        const int c = sph_cell(xp0, xp1, xp2);
        const int ix = c % SPH_NX, iy = (c / SPH_NX) % SPH_NY, iz = c / (SPH_NX * SPH_NY);
        int any = 0;
        for (int dz = -1; dz <= 1 && !any; ++dz)
        {
            const int jz = iz + dz; if (jz < 0 || jz >= SPH_NZ) continue;
            for (int dy = -1; dy <= 1 && !any; ++dy)
            {
                const int jy = iy + dy; if (jy < 0 || jy >= SPH_NY) continue;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int jx = ix + dx; if (jx < 0 || jx >= SPH_NX) continue;
                    if (s.cellFluid[(jz * SPH_NY + jy) * SPH_NX + jx]) { any = 1; break; }
                }
            }
        }
        if (!any) { s.SForce[p] = Real(0.0); s.SForce[N + p] = Real(0.0); s.SForce[2 * N + p] = Real(0.0); return; }
    }
    sph_neigh2(s, p, xp0, xp1, xp2, [&](int q, Real d0, Real d1, Real d2, Real r2)
    {
        if (s.itype[q] <= Real(0.0)) return;
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

template <class Real>
void launch_fluid_nbr_build(SphPtrsT<Real> s, int maxn, int* d_over, cudaStream_t st)
{
    k_fluid_nbr_build<Real><<<nblocks(SPH_NTOTAL), BLK, 0, st>>>(s, maxn, d_over);
}
template <class Real>
void launch_pass1_fluid_list(SphPtrsT<Real> s, cudaStream_t st)
{
    k_pass1_fluid_list<Real><<<nblocks(SPH_NTOTAL), BLK, 0, st>>>(s);
}
template <class Real>
void launch_pass1_solid_ell(SphPtrsT<Real> s, cudaStream_t st)
{
    k_pass1_solid_ell<Real><<<nblocks(SPH_SOLNUM), BLK, 0, st>>>(s);
}
template <class Real>
void launch_pass2_solid_ell(SphPtrsT<Real> s, cudaStream_t st)
{
    k_pass2_solid_ell<Real><<<nblocks(SPH_SOLNUM), BLK, 0, st>>>(s);
}

#define SPH_INSTANTIATE_EXT(Real)                                                      \
    template void launch_fluid_nbr_build<Real>(SphPtrsT<Real>, int, int*, cudaStream_t); \
    template void launch_pass1_fluid_list<Real>(SphPtrsT<Real>, cudaStream_t);           \
    template void launch_pass1_solid_ell<Real>(SphPtrsT<Real>, cudaStream_t);            \
    template void launch_pass2_solid_ell<Real>(SphPtrsT<Real>, cudaStream_t);
#if SPH_HAVE_FP64
SPH_INSTANTIATE_EXT(double)
#endif
#if SPH_HAVE_FP32
SPH_INSTANTIATE_EXT(float)
#endif
