// ============================================================================
//  sph_host.cu -- host driver for the CUDA port of 1.cpp
//
//  The solver lives in a template on Real so the fp64 / fp32 paths share one
//  implementation; `main` picks the instantiation at start-up:
//
//      sph_cuda.exe                 input.dat, fp64 (default)
//      sph_cuda.exe --fp32          fp32
//      set SPH_PRECISION=fp32       fp32
//      sph_cuda.exe other.dat       alternative input file
//
//  Environment: SPH_MAXSTEP (max steps), SPH_PROFILE (per-phase timing),
//  plus the debug switches at the bottom of this file.
// ============================================================================
#include "sph.h"

#include <vector>
#include <chrono>
#include <cstring>
#include <cctype>
#include <charconv>
#include <thread>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err_ = (x);                                                 \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

enum { PREC_FP64 = 0, PREC_FP32 = 1 };

#ifdef SPH_DEFAULT_FP32
#define SPH_PREC_DEFAULT PREC_FP32
#else
#define SPH_PREC_DEFAULT PREC_FP64
#endif

// ---------------------------------------------------------------------------
// output staging: everything is converted to double before printing so the
// reference comparison tool keeps working unchanged for both precisions
// ---------------------------------------------------------------------------
enum { HX0 = 0, HX1, HX2, HP, HVX0, HVX1, HVX2, HRHO, HMASS, HITYPE,
       HX00, HX01, HX02, HRHO0, HPIA11, HPIA22, HPIA33, HNARR };

template <class Real>
struct Solver
{
    SphPtrsT<Real> g = SphPtrsT<Real>();
    int*  d_niac = nullptr;
    int*  d_nbr  = nullptr;
    Real* d_sumvecn = nullptr;
    Real* d_a = nullptr;

    std::vector<double> hout;
    bool dump_step = false;
    bool dump_a = false;

    // ---------------- allocation ----------------
    void* dev_alloc(size_t bytes)
    {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, bytes));
        CUDA_CHECK(cudaMemset(p, 0, bytes));
        return p;
    }

    // no memset: for buffers whose first write covers every byte
    void* dev_alloc_raw(size_t bytes)
    {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, bytes));
        return p;
    }

    Real* dev_alloc_real(size_t count) { return (Real*)dev_alloc(count * sizeof(Real)); }
    Real* dev_alloc_real_raw(size_t count) { return (Real*)dev_alloc_raw(count * sizeof(Real)); }

    void alloc_state()
    {
        const size_t n1 = (size_t)SPH_NUM;
        const size_t n3 = 3 * n1;
        const size_t n9 = 9 * n1;

        g.x      = dev_alloc_real(n3);
        g.vx     = dev_alloc_real(n3);
        g.X0     = dev_alloc_real(n3);
        g.vecn   = dev_alloc_real(n3);
        g.PIJW   = dev_alloc_real(n3);
        g.SForce = dev_alloc_real(n3);
        g.dvxdt  = dev_alloc_real(n3);
        g.viscos = dev_alloc_real(n3);

        g.rho    = dev_alloc_real(n1);
        g.irho   = dev_alloc_real(n1);
        g.rho0   = dev_alloc_real(n1);
        g.mass   = dev_alloc_real(n1);
        g.p      = dev_alloc_real(n1);
        g.itype  = dev_alloc_real(n1);
        g.SurF   = dev_alloc_real(n1);
        g.SurFD  = dev_alloc_real(n1);
        g.drhodt = dev_alloc_real(n1);

        g.A      = dev_alloc_real(n9);
        g.F      = dev_alloc_real(n9);
        g.DFdt   = dev_alloc_real(n9);
        g.Pia    = dev_alloc_real((size_t)SPH_PIA_STRIDE * SPH_SOLNUM);   // AoS, see sph.h

        g.solidPack = dev_alloc_real((size_t)14 * SPH_SOLNUM);

        g.npair  = 0;                  // A3 list, built by build_solid_lists()
        g.swidth = 0;
        g.sw     = 0;
        g.sCnt   = nullptr;
        g.sQ     = nullptr;
        g.sG     = nullptr;
        g.sB     = nullptr;
        g.sD     = nullptr;

        // P1: the fp64 fluid neighbour list.  fp32 keeps the fused kernel, so
        // the array is not even allocated there.
        g.fnbr = nullptr;
        g.fcnt = nullptr;
        g.fwidth = 0;
        if constexpr (sizeof(Real) >= sizeof(double))
        {
            g.fwidth = SPH_FLIST_SLOTS;
            g.fnbr = (int*)dev_alloc_raw((size_t)g.fwidth * SPH_NTOTAL * sizeof(int));
            g.fcnt = (int*)dev_alloc((size_t)SPH_NTOTAL * sizeof(int));
        }

        g.cellStart = (int*)dev_alloc((SPH_NCELL + 1) * sizeof(int));
        g.cellCur   = (int*)dev_alloc(SPH_NCELL * sizeof(int));
        g.cellSolid = (int*)dev_alloc(SPH_NCELL * sizeof(int));
        g.cellFluid = (int*)dev_alloc(SPH_NCELL * sizeof(int));
        g.sorted    = (int*)dev_alloc(SPH_NUM * sizeof(int));

        g.xf = (float*)dev_alloc_raw(n3 * sizeof(float));   // A4, filled below

        g.u    = dev_alloc_real(n3);       // fp32 accuracy support, see sph.h
        g.xc   = dev_alloc_real(n3);
        g.rhoc = dev_alloc_real(n1);

        d_niac    = (int*)dev_alloc(sizeof(int));
        d_nbr     = (int*)dev_alloc(SPH_NUM * sizeof(int));
        d_sumvecn = dev_alloc_real(n3);
        d_a       = dev_alloc_real(n9);
    }

    // ---------------- input ----------------
    bool load_input(const char* path)
    {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
        fseek(f, 0, SEEK_END);
        const long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<char> buf((size_t)sz + 1);
        const size_t got = fread(buf.data(), 1, (size_t)sz, f);
        buf[got] = '\0';
        fclose(f);

        std::vector<double> hx(3 * SPH_NUM), hvx(3 * SPH_NUM), hX0(3 * SPH_NUM);
        std::vector<double> hr(SPH_NUM), hr0(SPH_NUM), hm(SPH_NUM),
                            hp(SPH_NUM), hit(SPH_NUM);

        const char* s = buf.data();
        char* end = nullptr;
        for (int i = 0; i < SPH_NUM; ++i)
        {
            double v[14];
            for (int k = 0; k < 14; ++k)
            {
                v[k] = strtod(s, &end);
                if (end == s) { fprintf(stderr, "input parse error at particle %d col %d\n", i, k); return false; }
                s = end;
            }
            hx[i] = v[0]; hx[SPH_NUM + i] = v[1]; hx[2 * SPH_NUM + i] = v[2];
            hp[i] = v[3];
            hvx[i] = v[4]; hvx[SPH_NUM + i] = v[5]; hvx[2 * SPH_NUM + i] = v[6];
            hr[i] = v[7];
            hm[i] = v[8];
            hit[i] = v[9];
            hX0[i] = v[10]; hX0[SPH_NUM + i] = v[11]; hX0[2 * SPH_NUM + i] = v[12];
            hr0[i] = v[13];
            if (hr[i] < 960.0) hr[i] = 960.0;
        }
        while (*s && isspace((unsigned char)*s)) ++s;
        if (*s)
            fprintf(stderr, "[input] warning: data after the %d expected records ignored\n", SPH_NUM);

        auto up = [](Real* dst, const std::vector<double>& src)
        {
            std::vector<Real> tmp(src.size());
            for (size_t k = 0; k < src.size(); ++k) tmp[k] = (Real)src[k];
            CUDA_CHECK(cudaMemcpy(dst, tmp.data(), src.size() * sizeof(Real), cudaMemcpyHostToDevice));
        };
        // C4: every kernel depends on this layout, and nothing else checks it.
        //   [0, NTOTAL)      fluid   itype > 0
        //   [NTOTAL, NALL)   solid   itype < -10
        //   [NALL, NUM)      wall    -10 <= itype <= 0
        // plus: all positions inside the fixed grid (outside, sph_cell() clamps
        // them into a boundary cell and the neighbour search silently degrades),
        // and no NaN.  The eta rule's max(p,q) also needs the original index
        // order, so a permuted deck must be rejected rather than run.
        int bad_type = 0, bad_pos = 0;
        for (int i = 0; i < SPH_NUM; ++i)
        {
            const double t = hit[i];
            const bool ok = (i < SPH_NTOTAL)
                              ? (t > 0.0)
                              : (i < SPH_NALL) ? (t < -10.0)
                                               : (t >= -10.0 && t <= 0.0);
            if (!ok && bad_type++ < 5)
                fprintf(stderr, "[input] particle %d: itype %g not allowed in this index range\n", i, t);
            for (int c = 0; c < 3; ++c)
            {
                const double v = hx[c * SPH_NUM + i];
                const bool in = (c == 0) ? (v >= SPH_GX0 && v <= SPH_GX1)
                              : (c == 1) ? (v >= SPH_GY0 && v <= SPH_GY1)
                                         : (v >= SPH_GZ0 && v <= SPH_GZ1);
                if (!in && bad_pos++ < 5)
                    fprintf(stderr, "[input] particle %d: x[%d] = %g outside the grid [%g, %g]\n",
                            i, c, v, (c == 0) ? SPH_GX0 : (c == 1) ? SPH_GY0 : SPH_GZ0,
                            (c == 0) ? SPH_GX1 : (c == 1) ? SPH_GY1 : SPH_GZ1);
            }
        }
        if (bad_type || bad_pos)
        {
            fprintf(stderr, "[input] %s: %d particle type(s) and %d position(s) rejected, "
                            "refusing to run\n", path, bad_type, bad_pos);
            return false;
        }

        up(g.x,   hx);
        up(g.vx,  hvx);
        up(g.X0,  hX0);
        up(g.rho, hr);
        up(g.rho0, hr0);
        up(g.mass, hm);
        up(g.p,   hp);
        up(g.itype, hit);
        {
            // displacement of the solid-type particles, formed in double BEFORE the
            // conversion to Real: x - X0 keeps only 3-4 digits if it is formed in float
            std::vector<double> hu(3 * SPH_NUM, 0.0);
            for (int i = 0; i < SPH_NUM; ++i)
                if (hit[i] < -10.0)
                    for (int c = 0; c < 3; ++c)
                        hu[c * SPH_NUM + i] = hx[c * SPH_NUM + i] - hX0[c * SPH_NUM + i];
            up(g.u, hu);
        }
        return true;
    }

    // ---------------- grid ----------------
    // build_cells_for() bins whatever positions `s.x` points at; build_cells()
    // is the live configuration.  build_solid_lists() calls it with x = X0.
    void build_cells_for(const SphPtrsT<Real>& s)
    {
        CUDA_CHECK(cudaMemset(g.cellCur, 0, SPH_NCELL * sizeof(int)));
        CUDA_CHECK(cudaMemset(g.cellSolid, 0, SPH_NCELL * sizeof(int)));
        CUDA_CHECK(cudaMemset(g.cellFluid, 0, SPH_NCELL * sizeof(int)));
        launch_cell_hist(s, g.cellCur, g.cellSolid, g.cellFluid, 0);
        launch_cell_scan(g.cellStart, g.cellCur, 0);
        launch_cell_scatter(s, g.cellCur, 0);
        launch_cell_sort(g.cellStart, g.sorted, 0);
    }

    void build_cells() { build_cells_for(g); }

    // ---------------- A3: static solid pair list ----------------
    // count -> host reduction (gives the ELL width) -> allocate -> fill.
    // Everything is derived from the reference configuration (X0, A), which
    // never changes, so this runs exactly once per process.
    //
    // C1: the search must run in a cell list binned from X0 as well.  Binning it
    // from the *current* x -- what the port did until this fix -- loses pairs as
    // soon as the solid has moved about one cell, silently and permanently: the
    // list is built once, so a restart from a deformed state would carry wrong
    // stresses for the whole run.  Measured on this deck: no error up to 15 mm
    // of rigid offset, 5.3 MPa at 20 mm.
    void build_solid_lists()
    {
        const int n = SPH_SOLNUM;
        if (n <= 0) return;

        {
            SphPtrsT<Real> ref = g;
            ref.x = g.X0;                       // reference configuration
            build_cells_for(ref);
        }

        int* d_cnt = (int*)dev_alloc_raw((size_t)n * sizeof(int));
        launch_solid_pair_count(g, d_cnt, 0);
        std::vector<int> h(n), cur(n);
        CUDA_CHECK(cudaMemcpy(h.data(), d_cnt, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_cnt));

        long long total = 0;
        int width = 0;
        for (int i = 0; i < n; ++i) { total += h[i]; if (h[i] > width) width = h[i]; }
        if (total > 2147483647LL || (long long)width * n > 2147483647LL)
        {
            fprintf(stderr, "[A3] list too large for 32-bit slot indices "
                            "(%lld pairs, width %d, %d solids)\n", total, width, n);
            exit(3);
        }
        g.npair  = (int)total;
        g.swidth = width;
        g.sw     = width * n;                   // slots; one component array is this long
        if (width <= 0) { build_cells(); return; }

        g.sCnt = (int*)dev_alloc_raw((size_t)n * sizeof(int));
        g.sQ   = (int*)dev_alloc_raw((size_t)g.sw * sizeof(int));
        g.sG   = dev_alloc_real_raw((size_t)3 * g.sw);
        g.sB   = dev_alloc_real_raw((size_t)3 * g.sw);
        g.sD   = dev_alloc_real_raw((size_t)3 * g.sw);

        int* d_cur = (int*)dev_alloc_raw((size_t)n * sizeof(int));
        CUDA_CHECK(cudaMemcpy(g.sCnt, h.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
        launch_solid_pair_fill(g, d_cur, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        // C3: the two passes are compiled separately and agree only if the
        // knife-edge geometry test decides identically in both.  Verify that
        // instead of hoping: a mismatch would silently shift the slot range of
        // every following particle.
        CUDA_CHECK(cudaMemcpy(cur.data(), d_cur, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_cur));
        long long bad = 0;
        int first = -1;
        for (int i = 0; i < n; ++i)
            if (cur[i] != h[i]) { ++bad; if (first < 0) first = i; }
        if (bad != 0)
        {
            fprintf(stderr, "[A3] FATAL: fill/count mismatch on %lld of %d solids "
                            "(first k=%d: counted %d, filled %d)\n",
                    bad, n, first, h[first], cur[first]);
            exit(3);
        }

        build_cells();                          // restore the live configuration

        const double mb = (double)g.sw * (double)(sizeof(int) + 9 * sizeof(Real)) / 1048576.0;
        printf("[A3] solid pair list: %d pairs, %.1f per solid particle, "
               "ELL width %d (padding %.1f%%), %.1f MB\n",
               g.npair, (double)g.npair / (double)n, g.swidth,
               100.0 * ((double)g.sw / (double)(g.npair > 0 ? g.npair : 1) - 1.0), mb);
    }

    void init_reference_state()
    {
        CUDA_CHECK(cudaMemset(d_sumvecn, 0, 3 * SPH_NUM * sizeof(Real)));
        launch_nor_sumvecn(g, d_sumvecn, 0);
        launch_nor_normalize(g, d_sumvecn, 0);

        CUDA_CHECK(cudaMemset(d_a, 0, 9 * SPH_NUM * sizeof(Real)));
        launch_gra_correc(g, d_a, 0);
        launch_gra_invert(g, d_a, 0);
    }

    // ---------------- debug dumps (always written as double) ----------------
    void dump_real(const char* nm, const Real* d)
    {
        std::vector<Real> h(SPH_NUM);
        CUDA_CHECK(cudaMemcpy(h.data(), d, SPH_NUM * sizeof(Real), cudaMemcpyDeviceToHost));
        std::vector<double> o(SPH_NUM);
        for (int i = 0; i < SPH_NUM; ++i) o[i] = (double)h[i];
        char fn[256]; snprintf(fn, sizeof(fn), "gpu_%s.bin", nm);
        FILE* f = fopen(fn, "wb"); if (!f) return;
        fwrite(o.data(), sizeof(double), SPH_NUM, f); fclose(f);
    }

    void dump_pia()
    {
        const int SD = SPH_PIA_STRIDE;
        std::vector<Real> tmp((size_t)SD * SPH_SOLNUM);
        CUDA_CHECK(cudaMemcpy(tmp.data(), g.Pia, tmp.size() * sizeof(Real), cudaMemcpyDeviceToHost));
        const char* nm[9] = {"Pia11","Pia12","Pia13","Pia21","Pia22","Pia23","Pia31","Pia32","Pia33"};
        std::vector<double> out(SPH_NUM);
        for (int c = 0; c < 9; ++c)
        {
            for (int i = 0; i < SPH_NUM; ++i) out[i] = 0.0;
            for (int k = 0; k < SPH_SOLNUM; ++k) out[SPH_NTOTAL + k] = (double)tmp[(size_t)k * SD + c];
            char fn[256]; snprintf(fn, sizeof(fn), "gpu_%s.bin", nm[c]);
            FILE* fh = fopen(fn, "wb"); if (!fh) continue;
            fwrite(out.data(), sizeof(double), SPH_NUM, fh); fclose(fh);
        }
    }

    void dump_reference_arrays()
    {
        const char* nm[18] = {"a11","a12","a13","a21","a22","a23","a31","a32","a33",
                              "A11","A12","A13","A21","A22","A23","A31","A32","A33"};
        const Real* src[18];
        for (int k = 0; k < 9; ++k) src[k] = d_a + (size_t)k * SPH_NUM;
        for (int k = 0; k < 9; ++k) src[9 + k] = g.A + (size_t)k * SPH_NUM;
        for (int k = 0; k < 18; ++k) dump_real(nm[k], src[k]);
        dump_real("vecn0", g.vecn); dump_real("vecn1", g.vecn + SPH_NUM); dump_real("vecn2", g.vecn + 2 * SPH_NUM);
        printf("[dumpa] 21 arrays written\n");
    }

    void load_A_from_dump(const char* dir)
    {
        const char* nm[9] = {"A11","A12","A13","A21","A22","A23","A31","A32","A33"};
        std::vector<double> hb(SPH_NUM);
        std::vector<Real> rb(SPH_NUM);
        for (int k = 0; k < 9; ++k)
        {
            char fn[512]; snprintf(fn, sizeof(fn), "%s/cpu_%s.bin", dir, nm[k]);
            FILE* ff = fopen(fn, "rb");
            if (!ff) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
            size_t n = fread(hb.data(), sizeof(double), SPH_NUM, ff); fclose(ff);
            if (n != (size_t)SPH_NUM) { fprintf(stderr, "short read %s\n", fn); exit(1); }
            for (int i = 0; i < SPH_NUM; ++i) rb[i] = (Real)hb[i];
            CUDA_CHECK(cudaMemcpy(g.A + (size_t)k * SPH_NUM, rb.data(), SPH_NUM * sizeof(Real), cudaMemcpyHostToDevice));
        }
        printf("[loadA] gradient-correction matrix taken from CPU dump (%s)\n", dir);
    }

    // same expression and the same scalar type as the device-side sph_cell(), so a
    // particle sitting on a cell face is binned identically in the fp32 build too
    static int host_cell(Real x, Real y, Real z)
    {
        const Real inv = Real(1.0) / Real(SPH_CS);
        int ix = (int)floor((x - Real(SPH_GX0)) * inv); if (ix < 0) ix = 0; if (ix >= SPH_NX) ix = SPH_NX - 1;
        int iy = (int)floor((y - Real(SPH_GY0)) * inv); if (iy < 0) iy = 0; if (iy >= SPH_NY) iy = SPH_NY - 1;
        int iz = (int)floor((z - Real(SPH_GZ0)) * inv); if (iz < 0) iz = 0; if (iz >= SPH_NZ) iz = SPH_NZ - 1;
        return (iz * SPH_NY + iy) * SPH_NX + ix;
    }

    void verify_cells()
    {
        std::vector<int> st(SPH_NCELL + 1), so(SPH_NUM);
        std::vector<Real> hx(3 * SPH_NUM);
        CUDA_CHECK(cudaMemcpy(st.data(), g.cellStart, (SPH_NCELL + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(so.data(), g.sorted, SPH_NUM * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hx.data(), g.x, 3 * SPH_NUM * sizeof(Real), cudaMemcpyDeviceToHost));

        std::vector<int> seen(SPH_NUM, 0);
        int dup = 0, outOfCell = 0, missing = 0;
        for (int c = 0; c < SPH_NCELL; ++c)
        {
            if (st[c] < 0 || st[c] > st[c + 1] || st[c + 1] > SPH_NUM) { printf("  cellStart bad at %d\n", c); break; }
            for (int a = st[c]; a < st[c + 1]; ++a)
            {
                const int p = so[a];
                if (p < 0 || p >= SPH_NUM) continue;
                seen[p]++;
                const int cc = host_cell(hx[p], hx[SPH_NUM + p], hx[2 * SPH_NUM + p]);
                if (cc != c) outOfCell++;
            }
        }
        for (int p = 0; p < SPH_NUM; ++p) { if (seen[p] == 0) missing++; if (seen[p] > 1) dup += seen[p] - 1; }
        printf("[verify] start[0]=%d start[N]=%d dup=%d missing=%d wrongCell=%d\n",
               st[0], st[SPH_NCELL], dup, missing, outOfCell);
    }

    // ---------------- one time step ----------------
    void step(int itime)
    {
        const bool dump = (dump_step && itime == 0);

        launch_prepare(g, 0);

        CUDA_CHECK(cudaMemsetAsync(g.cellCur, 0, SPH_NCELL * sizeof(int), 0));
        CUDA_CHECK(cudaMemsetAsync(g.cellSolid, 0, SPH_NCELL * sizeof(int), 0));
        CUDA_CHECK(cudaMemsetAsync(g.cellFluid, 0, SPH_NCELL * sizeof(int), 0));
        launch_cell_hist(g, g.cellCur, g.cellSolid, g.cellFluid, 0);
        launch_cell_scan(g.cellStart, g.cellCur, 0);
        launch_cell_scatter(g, g.cellCur, 0);
        launch_cell_sort(g.cellStart, g.sorted, 0);

        launch_pass1(g, 0);
        if (dump)
        {
            const char* fn[9] = {"F11","F12","F13","F21","F22","F23","F31","F32","F33"};
            const char* dn[9] = {"DFdt11","DFdt12","DFdt13","DFdt21","DFdt22","DFdt23","DFdt31","DFdt32","DFdt33"};
            for (int k = 0; k < 9; ++k) { dump_real(fn[k], g.F + (size_t)k * SPH_NUM); dump_real(dn[k], g.DFdt + (size_t)k * SPH_NUM); }
            dump_real("drhodt", g.drhodt);
            dump_real("dvxdt0", g.dvxdt); dump_real("dvxdt1", g.dvxdt + SPH_NUM); dump_real("dvxdt2", g.dvxdt + 2 * SPH_NUM);
            dump_real("viscos0", g.viscos); dump_real("viscos1", g.viscos + SPH_NUM); dump_real("viscos2", g.viscos + 2 * SPH_NUM);
            dump_real("SurF", g.SurF);
            printf("[dumpstep] stage1 written\n");
        }

        launch_constitution(g, 0);
        if (dump)
        {
            dump_real("rho_s", g.rho);
            dump_pia();
            printf("[dumpstep] stage2 written\n");
        }

        launch_pass2(g, 0);
        if (dump)
        {
            dump_real("SForce0", g.SForce); dump_real("SForce1", g.SForce + SPH_NUM); dump_real("SForce2", g.SForce + 2 * SPH_NUM);
            dump_real("PIJW0", g.PIJW); dump_real("PIJW1", g.PIJW + SPH_NUM); dump_real("PIJW2", g.PIJW + 2 * SPH_NUM);
            printf("[dumpstep] stage3 written\n");
        }

        launch_update(g, 0);
        if (dump)
        {
            dump_real("x0", g.x); dump_real("x1", g.x + SPH_NUM); dump_real("x2", g.x + 2 * SPH_NUM);
            dump_real("vx0", g.vx); dump_real("vx1", g.vx + SPH_NUM); dump_real("vx2", g.vx + 2 * SPH_NUM);
            dump_real("rho_f", g.rho);
        }

        // C5: cudaDeviceSynchronize() reports execution errors, but a launch
        // configuration failure (too many registers, bad grid, ...) only shows
        // up in cudaGetLastError(), and the kernel would silently not have run.
        CUDA_CHECK(cudaGetLastError());
    }

    // ---------------- result<N>.dat ----------------
    void fetch_state()
    {
        if (hout.empty()) hout.resize((size_t)HNARR * SPH_NUM);
        auto pull = [&](int slot, const Real* src)
        {
            std::vector<Real> tmp(SPH_NUM);
            CUDA_CHECK(cudaMemcpy(tmp.data(), src, SPH_NUM * sizeof(Real), cudaMemcpyDeviceToHost));
            double* dst = hout.data() + (size_t)slot * SPH_NUM;
            for (int i = 0; i < SPH_NUM; ++i) dst[i] = (double)tmp[i];
        };
        pull(HX0, g.x); pull(HX1, g.x + SPH_NUM); pull(HX2, g.x + 2 * SPH_NUM);
        pull(HP, g.p);
        pull(HVX0, g.vx); pull(HVX1, g.vx + SPH_NUM); pull(HVX2, g.vx + 2 * SPH_NUM);
        pull(HRHO, g.rho); pull(HMASS, g.mass); pull(HITYPE, g.itype);
        pull(HX00, g.X0); pull(HX01, g.X0 + SPH_NUM); pull(HX02, g.X0 + 2 * SPH_NUM);
        pull(HRHO0, g.rho0);
        // Pia is AoS (12 Reals per solid particle): scatter its diagonal into
        // the output staging arrays; everything outside the solid range is 0
        {
            std::vector<Real> tmp((size_t)SPH_PIA_STRIDE * SPH_SOLNUM);
            CUDA_CHECK(cudaMemcpy(tmp.data(), g.Pia, tmp.size() * sizeof(Real), cudaMemcpyDeviceToHost));
            double* d11 = hout.data() + (size_t)HPIA11 * SPH_NUM;
            double* d22 = hout.data() + (size_t)HPIA22 * SPH_NUM;
            double* d33 = hout.data() + (size_t)HPIA33 * SPH_NUM;
            for (int i = 0; i < SPH_NUM; ++i) { d11[i] = 0.0; d22[i] = 0.0; d33[i] = 0.0; }
            for (int k = 0; k < SPH_SOLNUM; ++k)
            {
                const int i = SPH_NTOTAL + k;
                d11[i] = (double)tmp[(size_t)k * SPH_PIA_STRIDE + 0];
                d22[i] = (double)tmp[(size_t)k * SPH_PIA_STRIDE + 4];
                d33[i] = (double)tmp[(size_t)k * SPH_PIA_STRIDE + 8];
            }
        }
    }

    // ------------------------------------------------------------------
    // result<N>.dat: 14 columns, tab separated, "%.6g".
    //
    // A2: the reference format is reproduced with std::to_chars instead of
    // snprintf (same text, ~2x faster) and the body is formatted in a few
    // contiguous chunks that are written in order, so the file is byte for byte
    // what the single-threaded snprintf version produced.  At the 50-step
    // cadence this used to cost more wall clock than the 50 steps themselves in
    // the fp32 build (0.34 s per file against 1.2 s of compute for 500 steps).
    // ------------------------------------------------------------------
    void write_result(int idx)
    {
        char name[64];
        snprintf(name, sizeof(name), "result%d.dat", idx);
        FILE* f = fopen(name, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", name); return; }

        const double* x0 = hout.data() + (size_t)HX0 * SPH_NUM;
        const double* x1 = hout.data() + (size_t)HX1 * SPH_NUM;
        const double* x2 = hout.data() + (size_t)HX2 * SPH_NUM;
        const double* pp = hout.data() + (size_t)HP * SPH_NUM;
        const double* v0 = hout.data() + (size_t)HVX0 * SPH_NUM;
        const double* v1 = hout.data() + (size_t)HVX1 * SPH_NUM;
        const double* v2 = hout.data() + (size_t)HVX2 * SPH_NUM;
        const double* rho = hout.data() + (size_t)HRHO * SPH_NUM;
        const double* mass = hout.data() + (size_t)HMASS * SPH_NUM;
        const double* it = hout.data() + (size_t)HITYPE * SPH_NUM;
        const double* X0 = hout.data() + (size_t)HX00 * SPH_NUM;
        const double* X1 = hout.data() + (size_t)HX01 * SPH_NUM;
        const double* X2 = hout.data() + (size_t)HX02 * SPH_NUM;
        const double* rho0 = hout.data() + (size_t)HRHO0 * SPH_NUM;
        const double* qa = hout.data() + (size_t)HPIA11 * SPH_NUM;
        const double* qb = hout.data() + (size_t)HPIA22 * SPH_NUM;
        const double* qc = hout.data() + (size_t)HPIA33 * SPH_NUM;

        auto fmt = [&](int i0, int i1, std::vector<char>& out)
        {
            out.reserve((size_t)(i1 - i0) * 96 + 256);  // worst case is ~240 B/line
            for (int i = i0; i < i1; ++i)
            {
                const double ppp = (i < SPH_NTOTAL) ? pp[i] : (qa[i] + qb[i] + qc[i]);
                const double col[14] = { x0[i], x1[i], x2[i], ppp, v0[i], v1[i], v2[i], rho[i],
                                         mass[i], it[i], X0[i], X1[i], X2[i], rho0[i] };
                for (int c = 0; c < 14; ++c)
                {
                    char tmp[64];
                    const auto r = std::to_chars(tmp, tmp + sizeof(tmp), col[c],
                                                 std::chars_format::general, 6);
                    out.insert(out.end(), tmp, r.ptr);
                    out.push_back(c == 13 ? '\n' : '\t');
                }
            }
        };

        int nth = (int)std::thread::hardware_concurrency();
        if (nth < 1) nth = 1;
        if (nth > 8) nth = 8;
        if (SPH_NUM / nth < 10000) nth = 1;
        if (nth == 1)
        {
            std::vector<char> out;
            fmt(0, SPH_NUM, out);
            if (!out.empty()) fwrite(out.data(), 1, out.size(), f);
        }
        else
        {
            std::vector<std::vector<char>> part((size_t)nth);
            std::vector<std::thread> th;
            th.reserve((size_t)nth - 1);
            int done = 0;
            const int per = SPH_NUM / nth;
            for (int k = 0; k < nth; ++k)
            {
                const int i0 = done;
                const int i1 = (k == nth - 1) ? SPH_NUM : (i0 + per);
                done = i1;
                if (k == nth - 1) { fmt(i0, i1, part[(size_t)k]); }
                else th.emplace_back(fmt, i0, i1, std::ref(part[(size_t)k]));
            }
            for (auto& t : th) t.join();
            for (int k = 0; k < nth; ++k)
                if (!part[(size_t)k].empty())
                    fwrite(part[(size_t)k].data(), 1, part[(size_t)k].size(), f);
        }
        fclose(f);
    }
};

// ---------------------------------------------------------------------------
// driver for one precision
// ---------------------------------------------------------------------------
template <class Real>
static int run_case(const char* input_path, int iMaxStep, int profile, const char* prec_name)
{
    Solver<Real> S;
    S.dump_step = (getenv("SPH_DUMPSTEP") != nullptr);
    S.dump_a    = (getenv("SPH_DUMPA") != nullptr);

    auto t0 = std::chrono::steady_clock::now();
    S.alloc_state();
    if (!S.load_input(input_path)) return 1;
    launch_copy_xf(S.g, 0);                     // A4: float copy before the first search

    S.build_cells();
    if (getenv("SPH_VERIFY")) S.verify_cells();
    S.init_reference_state();
    CUDA_CHECK(cudaDeviceSynchronize());
    // C8: the fp32 solid kernels accumulate F - I, which is only the same tensor
    // as F when every A is a true inverse (a * = I).  A particle whose A fell
    // back to the identity breaks that, so refuse fp32 rather than run wrong.
    {
        int nfb = gra_fallback_count();
        if (nfb != 0)
        {
            fprintf(stderr, "[A] warning: %d particle(s) have a singular gradient-correction "
                            "matrix and fell back to A = I (as 1.cpp does)\n", nfb);
            if constexpr (sizeof(Real) < sizeof(double))
            {
                fprintf(stderr, "[A] fp32: the F - I displacement form is invalid for this deck; "
                                "re-run in fp64\n");
                return 4;
            }
        }
    }
    if (const char* pa = getenv("SPH_LOADA")) S.load_A_from_dump(pa);
    launch_build_solid_pack(S.g, 0);            // after SPH_LOADA: the pack holds a copy of A
    if (sph_use_static_solid) S.build_solid_lists();   // A3, must follow GraCorrec + LOADA
    CUDA_CHECK(cudaDeviceSynchronize());
    if (S.dump_a) S.dump_reference_arrays();
    if (const char* pt = getenv("SPH_NBRLIST"))
    {
        const int target = atoi(pt);
        if (target < 0 || target >= SPH_NUM) { fprintf(stderr, "SPH_NBRLIST: particle %d out of range\n", target); return 2; }
        int hn = 0;
        launch_nbr_list(S.g, target, S.d_nbr, S.d_niac, 0);
        CUDA_CHECK(cudaMemcpy(&hn, S.d_niac, sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> hl((size_t)(hn > 0 ? hn : 1));
        CUDA_CHECK(cudaMemcpy(hl.data(), S.d_nbr, (size_t)hn * sizeof(int), cudaMemcpyDeviceToHost));
        printf("[nbrlist] p=%d n=%d :", target, hn);
        for (int k = 0; k < hn; ++k) printf(" %d", hl[k]);
        printf("\n");
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("[init] %s, %.3f s, %zu bytes/particle\n", prec_name,
           std::chrono::duration<double>(t1 - t0).count(),
           sizeof(Real) * (size_t)69);

    // ------------------------------------------------------------------
    // SPH_BENCH: time every kernel separately (20 iterations each)
    // ------------------------------------------------------------------
    if (getenv("SPH_BENCH"))
    {
        auto bench = [&](const char* name, auto&& fn)
        {
            fn(); CUDA_CHECK(cudaDeviceSynchronize());
            const int R = 20;
            auto b0 = std::chrono::steady_clock::now();
            for (int r = 0; r < R; ++r) fn();
            CUDA_CHECK(cudaDeviceSynchronize());
            auto b1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double>(b1 - b0).count() * 1000.0 / R;
            printf("[bench] %-26s %8.3f ms/iter\n", name, ms);
        };
        S.build_cells();
        bench("cell(hist+scan+scatter)", [&] { S.build_cells(); });
        bench("prepare",                 [&] { launch_prepare(S.g, 0); });
        bench("pass1_fluid",             [&] { launch_pass1_fluid(S.g, 0); });
        if (sph_fast_fluid_loop)
        {
            sph_fast_fluid_loop = 0;
            bench("pass1_fluid (literal)", [&] { launch_pass1_fluid(S.g, 0); });
            sph_fast_fluid_loop = 1;
        }
        if constexpr (sizeof(Real) >= sizeof(double))
        {
            if (S.g.fwidth > 0)
            {
                bench("fluid nbr build",       [&] { launch_fluid_nbr_build(S.g, 0); });
                bench("pass1_fluid over list", [&] { launch_pass1_fluid_over_list(S.g, 0); });
                launch_fluid_nbr_build(S.g, 0);
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<int> fc(SPH_NTOTAL);
                CUDA_CHECK(cudaMemcpy(fc.data(), S.g.fcnt, (size_t)SPH_NTOTAL * sizeof(int),
                                      cudaMemcpyDeviceToHost));
                long long tot = 0; int mx = 0, mn = 1 << 30, ov = 0;
                for (int v : fc)
                {
                    if (v < 0) { ++ov; continue; }
                    tot += v; if (v > mx) mx = v; if (v < mn) mn = v;
                }
                printf("[bench] fluid neighbours: mean=%.1f min=%d max=%d overflow=%d (width %d)\n",
                       (double)tot / (double)(SPH_NTOTAL - ov), mn, mx, ov, S.g.fwidth);
                sph_use_fluid_list = 0;
                bench("pass1_fluid (fused)",   [&] { launch_pass1_fluid(S.g, 0); });
                sph_use_fluid_list = 1;
            }
        }
        bench("pass1_solid",             [&] { launch_pass1_solid(S.g, 0); });
        bench("constitution",            [&] { launch_constitution(S.g, 0); });
        bench("pass2_fluid",             [&] { launch_pass2_fluid(S.g, 0); });
        bench("pass2_solid",             [&] { launch_pass2_solid(S.g, 0); });
        // A/B: the pre-A3 solid path (cell list walk every step) in the same run
        if (sph_use_static_solid && S.g.npair > 0)
        {
            sph_use_static_solid = 0;
            bench("pass1_solid (pre-A3)",  [&] { launch_pass1_solid(S.g, 0); });
            bench("pass2_solid (pre-A3)",  [&] { launch_pass2_solid(S.g, 0); });
            sph_use_static_solid = 1;
        }
        bench("update_fluid",            [&] { launch_update_fluid(S.g, 0); });
        bench("update_solid",            [&] { launch_update_solid(S.g, 0); });
        bench("update_pos",              [&] { launch_update_pos(S.g, 0); });
        bench("count_pairs (traversal)", [&] { launch_count_pairs(S.g, S.d_niac, 0); });
        bench("loads x13 (scattered)",   [&] { launch_bench_loads(S.g, S.d_a, true, 0); });
        bench("loads x13 (coalesced)",   [&] { launch_bench_loads(S.g, S.d_a, false, 0); });
        return 0;
    }

    if (profile && S.dump_step)
        fprintf(stderr, "note: SPH_DUMPSTEP is ignored while SPH_PROFILE is set (the profiled loop does not dump)\n");

    double step_seconds = 0.0;
    double prof[6] = {0, 0, 0, 0, 0, 0};
    cudaEvent_t ev[8];
    if (profile) for (int k = 0; k < 8; ++k) CUDA_CHECK(cudaEventCreate(&ev[k]));

    for (int itime = 0; itime < iMaxStep; ++itime)
    {
        if (itime % 50 == 0)
        {
            // the list built inside the previous step predates that step's position
            // update; rebuild it so the printed count belongs to the current positions
            if (itime > 0) S.build_cells();
            CUDA_CHECK(cudaMemset(S.d_niac, 0, sizeof(int)));
            launch_count_pairs(S.g, S.d_niac, 0);
            int niac = 0;
            CUDA_CHECK(cudaMemcpy(&niac, S.d_niac, sizeof(int), cudaMemcpyDeviceToHost));
            printf("T= %d\tpairs=%d\n", itime / 50, niac / 2);
        }

        if (profile)
        {
            CUDA_CHECK(cudaEventRecord(ev[0], 0));
            launch_prepare(S.g, 0);
            CUDA_CHECK(cudaEventRecord(ev[1], 0));
            CUDA_CHECK(cudaMemsetAsync(S.g.cellCur, 0, SPH_NCELL * sizeof(int), 0));
            CUDA_CHECK(cudaMemsetAsync(S.g.cellSolid, 0, SPH_NCELL * sizeof(int), 0));
            CUDA_CHECK(cudaMemsetAsync(S.g.cellFluid, 0, SPH_NCELL * sizeof(int), 0));
            launch_cell_hist(S.g, S.g.cellCur, S.g.cellSolid, S.g.cellFluid, 0);
            launch_cell_scan(S.g.cellStart, S.g.cellCur, 0);
            launch_cell_scatter(S.g, S.g.cellCur, 0);
        launch_cell_sort(S.g.cellStart, S.g.sorted, 0);
            CUDA_CHECK(cudaEventRecord(ev[2], 0));
            launch_pass1(S.g, 0);
            CUDA_CHECK(cudaEventRecord(ev[3], 0));
            launch_constitution(S.g, 0);
            CUDA_CHECK(cudaEventRecord(ev[4], 0));
            launch_pass2(S.g, 0);
            CUDA_CHECK(cudaEventRecord(ev[5], 0));
            launch_update(S.g, 0);
            CUDA_CHECK(cudaEventRecord(ev[6], 0));
            CUDA_CHECK(cudaEventSynchronize(ev[6]));
            for (int k = 0; k < 6; ++k)
            {
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev[k], ev[k + 1]));
                prof[k] += ms * 1e-3;
            }
        }
        else
        {
            auto s0 = std::chrono::steady_clock::now();
            S.step(itime);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto s1 = std::chrono::steady_clock::now();
            step_seconds += std::chrono::duration<double>(s1 - s0).count();
        }

        if (itime % 50 == 0)
        {
            S.fetch_state();
            S.write_result(itime / 50);
        }
    }

    if (profile)
    {
        const double n = (double)(iMaxStep > 0 ? iMaxStep : 1);
        printf("[prof] prepare=%.4f cell=%.4f pass1=%.4f cons=%.4f pass2=%.4f update=%.4f s/step\n",
               prof[0] / n, prof[1] / n, prof[2] / n, prof[3] / n, prof[4] / n, prof[5] / n);
        step_seconds = prof[0] + prof[1] + prof[2] + prof[3] + prof[4] + prof[5];
    }
    if (iMaxStep > 0)
        printf("[time] %s, %d steps, %.3f s total, %.3f ms/step\n",
               prec_name, iMaxStep, step_seconds, 1000.0 * step_seconds / iMaxStep);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static void usage()
{
    printf("usage: sph_cuda [input.dat] [--fp64|--fp32] [--precision=fp64|fp32]\n"
           "  env: SPH_MAXSTEP, SPH_PRECISION, SPH_PROFILE\n");
}

// PREC_FP32 / PREC_FP64 for a precision name (case-insensitive), -1 if the
// name is not recognised.  Used for both --precision=<name> and SPH_PRECISION.
static int parse_precision(const char* v)
{
    static const char* const f32[] = {"fp32", "f32", "float", "single", "32"};
    static const char* const f64[] = {"fp64", "f64", "double", "64"};
    char lo[16];
    size_t n = 0;
    for (; v[n] != '\0' && n + 1 < sizeof(lo); ++n) lo[n] = (char)tolower((unsigned char)v[n]);
    lo[n] = '\0';
    if (v[n] != '\0') return -1;                     // longer than any known name
    for (const char* s : f32) if (!strcmp(lo, s)) return PREC_FP32;
    for (const char* s : f64) if (!strcmp(lo, s)) return PREC_FP64;
    return -1;
}

int main(int argc, char** argv)
{
    const char* input = "input.dat";
    int prec = -1;

    for (int i = 1; i < argc; ++i)
    {
        const char* a = argv[i];
        if (!strcmp(a, "--fp32") || !strcmp(a, "-f32")) prec = PREC_FP32;
        else if (!strcmp(a, "--fp64") || !strcmp(a, "-f64")) prec = PREC_FP64;
        else if (!strncmp(a, "--precision=", 12))
        {
            prec = parse_precision(a + 12);
            if (prec < 0) { fprintf(stderr, "unknown precision '%s'\n", a + 12); usage(); return 2; }
        }
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(); return 0; }
        else if (a[0] != '-') input = a;
        else { fprintf(stderr, "unknown option %s\n", a); usage(); return 2; }
    }

    if (prec < 0)
    {
        if (const char* e = getenv("SPH_PRECISION"))
        {
            prec = parse_precision(e);
            if (prec < 0) { fprintf(stderr, "unknown SPH_PRECISION '%s'\n", e); usage(); return 2; }
        }
    }
    if (prec < 0) prec = SPH_PREC_DEFAULT;

    int iMaxStep = 10000001;
    if (const char* e = getenv("SPH_MAXSTEP")) iMaxStep = atoi(e);
    const int profile = (getenv("SPH_PROFILE") != nullptr) ? 1 : 0;
    if (const char* e = getenv("SPH_SOLDDYN")) sph_use_static_solid = (atoi(e) == 0) ? 1 : 0;
    if (const char* e = getenv("SPH_LITERALFLUID")) sph_fast_fluid_loop = (atoi(e) == 0) ? 1 : 0;
    if (const char* e = getenv("SPH_FLUIDLIST")) sph_use_fluid_list = (atoi(e) == 0) ? 0 : 1;

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    {
        const int nx = (int)floor((SPH_GX1 - SPH_GX0) / SPH_CS) + 1;
        const int ny = (int)floor((SPH_GY1 - SPH_GY0) / SPH_CS) + 1;
        const int nz = (int)floor((SPH_GZ1 - SPH_GZ0) / SPH_CS) + 1;
        if (nx != SPH_NX || ny != SPH_NY || nz != SPH_NZ)
        {
            fprintf(stderr, "grid mismatch: %dx%dx%d != %dx%dx%d\n", nx, ny, nz, SPH_NX, SPH_NY, SPH_NZ);
            return 1;
        }
    }

    const bool want_f32 = (prec == PREC_FP32);
#if !SPH_HAVE_FP32
    if (want_f32) { fprintf(stderr, "this build has no fp32 path (SPH_ENABLE_FP32=OFF)\n"); return 2; }
#endif
#if !SPH_HAVE_FP64
    if (!want_f32) { fprintf(stderr, "this build has no fp64 path (SPH_ENABLE_FP64=OFF)\n"); return 2; }
#endif

    printf("[CUDA] %s  sm_%d%d  SMs=%d  %.1f GB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1073741824.0);

#if SPH_HAVE_FP32
    if (want_f32) return run_case<float>(input, iMaxStep, profile, "fp32");
#endif
#if SPH_HAVE_FP64
    return run_case<double>(input, iMaxStep, profile, "fp64");
#else
    return 0;
#endif
}
