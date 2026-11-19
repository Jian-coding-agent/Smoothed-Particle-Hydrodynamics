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
#include <cstdint>
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
       HX00, HX01, HX02, HRHO0, HPIA11, HPIA22, HPIA33, HSURFD, HNARR };

template <class Real>
struct Solver
{
    SphPtrsT<Real> g = SphPtrsT<Real>();
    int*  d_niac = nullptr;
    int*  d_nbr  = nullptr;
    Real* d_sumvecn = nullptr;
    Real* d_a = nullptr;

    std::vector<double> ox, oy, oz, op;   // output staging, filled per result file
    std::vector<double> itype_host;       // itype of every record (constant)
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
        const size_t n1 = (size_t)g.d.num;
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
        g.Pia    = dev_alloc_real((size_t)SPH_PIA_STRIDE * g.d.solnum);   // AoS, see sph.h

        g.solidPack = dev_alloc_real((size_t)14 * g.d.solnum);

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
            g.fnbr = (int*)dev_alloc_raw((size_t)g.fwidth * g.d.ntotal * sizeof(int));
            g.fcnt = (int*)dev_alloc((size_t)g.d.ntotal * sizeof(int));
        }

        g.cellStart = (int*)dev_alloc((g.d.ncell + 1) * sizeof(int));
        g.cellCur   = (int*)dev_alloc(g.d.ncell * sizeof(int));
        g.cellSolid = (int*)dev_alloc(g.d.ncell * sizeof(int));
        g.cellFluid = (int*)dev_alloc(g.d.ncell * sizeof(int));
        g.sorted    = (int*)dev_alloc(g.d.num * sizeof(int));

        g.xf = (float*)dev_alloc_raw(n3 * sizeof(float));   // A4, filled below

        g.u    = dev_alloc_real(n3);       // fp32 accuracy support, see sph.h
        g.xc   = dev_alloc_real(n3);
        g.rhoc = dev_alloc_real(n1);

        d_niac    = (int*)dev_alloc(sizeof(int));
        d_nbr     = (int*)dev_alloc(g.d.num * sizeof(int));
        d_sumvecn = dev_alloc_real(n3);
        d_a       = dev_alloc_real(n9);
    }

    // ---------------- input ----------------
    // ---------------- deck input ----------------
    // Reads either the self-describing .sphbin container or the legacy
    // 14-column text deck.  The file name is arbitrary: argv[1].
    bool load_deck(const char* path)
    {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
        unsigned char hdr[128] = {0};
        const size_t got = fread(hdr, 1, sizeof(hdr), f);
        const bool binary = (got == sizeof(hdr) && memcmp(hdr, "SPHBIN\0\0", 8) == 0);

        std::vector<double> rows;          // AoS, num * 14
        if (binary)
        {
            uint32_t ver, hs, num, ntotal, solnum, nf, step, flags;
            memcpy(&ver, hdr + 8, 4);   memcpy(&hs, hdr + 12, 4);
            memcpy(&num, hdr + 16, 4);  memcpy(&ntotal, hdr + 20, 4);
            memcpy(&solnum, hdr + 24, 4); memcpy(&nf, hdr + 28, 4);
            memcpy(&step, hdr + 32, 4); memcpy(&flags, hdr + 36, 4);
            if (ver != 1 || hs != 128 || nf != 14)
            {
                fprintf(stderr, "[input] %s: unsupported container (version=%u header=%u "
                                "nfields=%u)\n", path, ver, hs, nf);
                fclose(f); return false;
            }
            g.d.num = (int)num; g.d.ntotal = (int)ntotal; g.d.solnum = (int)solnum;
            rows.resize((size_t)num * 14);
            if (fread(rows.data(), sizeof(double), rows.size(), f) != rows.size())
            {
                fprintf(stderr, "[input] %s: short read\n", path); fclose(f); return false;
            }
        }
        else
        {
            fseek(f, 0, SEEK_END);
            const long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            std::vector<char> buf((size_t)sz + 1);
            const size_t rd = fread(buf.data(), 1, (size_t)sz, f);
            buf[rd] = '\0';
            const char* q = buf.data();
            char* end = nullptr;
            rows.reserve(((size_t)rd / 60) * 14 + 14);
            for (;;)
            {
                double v[14];
                bool ok = true;
                for (int k = 0; k < 14; ++k)
                {
                    v[k] = strtod(q, &end);
                    if (end == q) { ok = false; break; }
                    q = end;
                }
                if (!ok) break;
                rows.insert(rows.end(), v, v + 14);
            }
        }
        fclose(f);
        return finish_deck(rows, path);
    }

    // Validates the block layout, derives the neighbour grid from the data (so a
    // new deck can never be binned into a stale box), allocates and uploads.
    bool finish_deck(const std::vector<double>& rows, const char* path)
    {
        const int nf = 14;
        const int num = (int)(rows.size() / nf);
        if (num <= 0 || (size_t)num * nf != rows.size())
        {
            fprintf(stderr, "[input] %s: %zu doubles is not num*14\n", path, rows.size());
            return false;
        }
        int nfl = 0, ns = 0, nw = 0;
        for (int i = 0; i < num; ++i)
        {
            const double t = rows[(size_t)i * nf + 9];
            if (t > 0.0) ++nfl;
            else if (t < -10.0) ++ns;
            else ++nw;
        }
        // layout: [0,ntotal) fluid, [ntotal,nall) solid, [nall,num) wall
        for (int i = 0; i < num; ++i)
        {
            const double t = rows[(size_t)i * nf + 9];
            const bool ok = (i < nfl)      ? (t > 0.0)
                          : (i < nfl + ns) ? (t < -10.0)
                                           : (t >= -10.0 && t <= 0.0);
            if (!ok)
            {
                fprintf(stderr, "[input] %s: particle %d: itype %g is outside its block "
                                "(fluid=%d solid=%d wall=%d); the deck must be ordered "
                                "fluid|solid|wall\n", path, i, t, nfl, ns, nw);
                return false;
            }
        }
        if (g.d.num != 0 && g.d.num != num)
        {
            fprintf(stderr, "[input] %s: header says %d records, file holds %d\n",
                    path, g.d.num, num);
            return false;
        }
        g.d.num = num; g.d.ntotal = nfl; g.d.solnum = ns;
        g.d.nall = nfl + ns; g.d.nwall = nw;

        double mn[3] = { 1e300, 1e300, 1e300 }, mx[3] = { -1e300, -1e300, -1e300 };
        for (int i = 0; i < num; ++i)
            for (int c = 0; c < 3; ++c)
            {
                const double v = rows[(size_t)i * nf + c];
                if (v < mn[c]) mn[c] = v;
                if (v > mx[c]) mx[c] = v;
            }
        const double cs = SPH_CS;
        const double margin = cs + 0.02;
        g.d.gx0 = mn[0] - margin; g.d.gy0 = mn[1] - margin; g.d.gz0 = mn[2] - margin;
        const double gx1 = mx[0] + margin, gy1 = mx[1] + margin, gz1 = mx[2] + margin;
        g.d.nx = (int)floor((gx1 - g.d.gx0) / cs) + 1;
        g.d.ny = (int)floor((gy1 - g.d.gy0) / cs) + 1;
        g.d.nz = (int)floor((gz1 - g.d.gz0) / cs) + 1;
        g.d.ncell = g.d.nx * g.d.ny * g.d.nz;
        fprintf(stderr, "[input] %s: num=%d fluid=%d solid=%d wall=%d grid=%dx%dx%d\n",
                path, g.d.num, g.d.ntotal, g.d.solnum, g.d.nwall,
                g.d.nx, g.d.ny, g.d.nz);

        alloc_state();

        std::vector<Real> t3((size_t)3 * num);
        auto up3 = [&](Real* dst, int c)
        {
            for (int i = 0; i < num; ++i)
                for (int k = 0; k < 3; ++k)
                    t3[(size_t)k * num + i] = (Real)rows[(size_t)i * nf + c + k];
            CUDA_CHECK(cudaMemcpy(dst, t3.data(), t3.size() * sizeof(Real), cudaMemcpyHostToDevice));
        };
        up3(g.x, 0); up3(g.vx, 4); up3(g.X0, 10);

        std::vector<Real> t1(num);
        auto up1 = [&](Real* dst, int c)
        {
            for (int i = 0; i < num; ++i) t1[i] = (Real)rows[(size_t)i * nf + c];
            CUDA_CHECK(cudaMemcpy(dst, t1.data(), t1.size() * sizeof(Real), cudaMemcpyHostToDevice));
        };
        up1(g.p, 3); up1(g.mass, 8); up1(g.itype, 9); up1(g.rho0, 13);

        // rho clamp of the reference solver
        for (int i = 0; i < num; ++i)
        {
            double r = rows[(size_t)i * nf + 7];
            if (r < 960.0) r = 960.0;
            t1[i] = (Real)r;
        }
        CUDA_CHECK(cudaMemcpy(g.rho, t1.data(), t1.size() * sizeof(Real), cudaMemcpyHostToDevice));

        // fp32 displacement x - X0 of the solid block, formed in double first
        std::vector<double> hu((size_t)3 * num, 0.0);
        for (int i = 0; i < num; ++i)
            if (rows[(size_t)i * nf + 9] < -10.0)
                for (int k = 0; k < 3; ++k)
                    hu[(size_t)k * num + i] = rows[(size_t)i * nf + k] - rows[(size_t)i * nf + 10 + k];
        {
            std::vector<Real> t(hu.size());
            for (size_t k = 0; k < hu.size(); ++k) t[k] = (Real)hu[k];
            CUDA_CHECK(cudaMemcpy(g.u, t.data(), t.size() * sizeof(Real), cudaMemcpyHostToDevice));
        }

        itype_host.resize(num);
        for (int i = 0; i < num; ++i) itype_host[i] = rows[(size_t)i * nf + 9];
        return true;
    }

    // ---------------- grid ----------------
    // build_cells_for() bins whatever positions `s.x` points at; build_cells()
    // is the live configuration.  build_solid_lists() calls it with x = X0.
    void build_cells_for(const SphPtrsT<Real>& s)
    {
        CUDA_CHECK(cudaMemset(g.cellCur, 0, g.d.ncell * sizeof(int)));
        CUDA_CHECK(cudaMemset(g.cellSolid, 0, g.d.ncell * sizeof(int)));
        CUDA_CHECK(cudaMemset(g.cellFluid, 0, g.d.ncell * sizeof(int)));
        launch_cell_hist(s, g.cellCur, g.cellSolid, g.cellFluid, 0);
        launch_cell_scan(g.cellStart, g.cellCur, g.d.ncell, 0);
        launch_cell_scatter(g, g.cellCur, 0);
        launch_cell_sort(g.cellStart, g.sorted, g.d.ncell, 0);
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
        const int n = g.d.solnum;
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
        CUDA_CHECK(cudaMemset(d_sumvecn, 0, 3 * g.d.num * sizeof(Real)));
        launch_nor_sumvecn(g, d_sumvecn, 0);
        launch_nor_normalize(g, d_sumvecn, 0);

        CUDA_CHECK(cudaMemset(d_a, 0, 9 * g.d.num * sizeof(Real)));
        launch_gra_correc(g, d_a, 0);
        launch_gra_invert(g, d_a, 0);
    }

    // ---------------- debug dumps (always written as double) ----------------
    void dump_real(const char* nm, const Real* d)
    {
        std::vector<Real> h(g.d.num);
        CUDA_CHECK(cudaMemcpy(h.data(), d, g.d.num * sizeof(Real), cudaMemcpyDeviceToHost));
        std::vector<double> o(g.d.num);
        for (int i = 0; i < g.d.num; ++i) o[i] = (double)h[i];
        char fn[256]; snprintf(fn, sizeof(fn), "gpu_%s.bin", nm);
        FILE* f = fopen(fn, "wb"); if (!f) return;
        fwrite(o.data(), sizeof(double), g.d.num, f); fclose(f);
    }

    void dump_pia()
    {
        const int SD = SPH_PIA_STRIDE;
        std::vector<Real> tmp((size_t)SD * g.d.solnum);
        CUDA_CHECK(cudaMemcpy(tmp.data(), g.Pia, tmp.size() * sizeof(Real), cudaMemcpyDeviceToHost));
        const char* nm[9] = {"Pia11","Pia12","Pia13","Pia21","Pia22","Pia23","Pia31","Pia32","Pia33"};
        std::vector<double> out(g.d.num);
        for (int c = 0; c < 9; ++c)
        {
            for (int i = 0; i < g.d.num; ++i) out[i] = 0.0;
            for (int k = 0; k < g.d.solnum; ++k) out[g.d.ntotal + k] = (double)tmp[(size_t)k * SD + c];
            char fn[256]; snprintf(fn, sizeof(fn), "gpu_%s.bin", nm[c]);
            FILE* fh = fopen(fn, "wb"); if (!fh) continue;
            fwrite(out.data(), sizeof(double), g.d.num, fh); fclose(fh);
        }
    }

    void dump_reference_arrays()
    {
        const char* nm[18] = {"a11","a12","a13","a21","a22","a23","a31","a32","a33",
                              "A11","A12","A13","A21","A22","A23","A31","A32","A33"};
        const Real* src[18];
        for (int k = 0; k < 9; ++k) src[k] = d_a + (size_t)k * g.d.num;
        for (int k = 0; k < 9; ++k) src[9 + k] = g.A + (size_t)k * g.d.num;
        for (int k = 0; k < 18; ++k) dump_real(nm[k], src[k]);
        dump_real("vecn0", g.vecn); dump_real("vecn1", g.vecn + g.d.num); dump_real("vecn2", g.vecn + 2 * g.d.num);
        printf("[dumpa] 21 arrays written\n");
    }

    void load_A_from_dump(const char* dir)
    {
        const char* nm[9] = {"A11","A12","A13","A21","A22","A23","A31","A32","A33"};
        std::vector<double> hb(g.d.num);
        std::vector<Real> rb(g.d.num);
        for (int k = 0; k < 9; ++k)
        {
            char fn[512]; snprintf(fn, sizeof(fn), "%s/cpu_%s.bin", dir, nm[k]);
            FILE* ff = fopen(fn, "rb");
            if (!ff) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
            size_t n = fread(hb.data(), sizeof(double), g.d.num, ff); fclose(ff);
            if (n != (size_t)g.d.num) { fprintf(stderr, "short read %s\n", fn); exit(1); }
            for (int i = 0; i < g.d.num; ++i) rb[i] = (Real)hb[i];
            CUDA_CHECK(cudaMemcpy(g.A + (size_t)k * g.d.num, rb.data(), g.d.num * sizeof(Real), cudaMemcpyHostToDevice));
        }
        printf("[loadA] gradient-correction matrix taken from CPU dump (%s)\n", dir);
    }

    // same expression and the same scalar type as the device-side sph_cell(), so a
    // particle sitting on a cell face is binned identically in the fp32 build too
    int host_cell(Real x, Real y, Real z)
    {
        const Real inv = Real(1.0) / Real(SPH_CS);
        int ix = (int)floor((x - Real(g.d.gx0)) * inv); if (ix < 0) ix = 0; if (ix >= g.d.nx) ix = g.d.nx - 1;
        int iy = (int)floor((y - Real(g.d.gy0)) * inv); if (iy < 0) iy = 0; if (iy >= g.d.ny) iy = g.d.ny - 1;
        int iz = (int)floor((z - Real(g.d.gz0)) * inv); if (iz < 0) iz = 0; if (iz >= g.d.nz) iz = g.d.nz - 1;
        return (iz * g.d.ny + iy) * g.d.nx + ix;
    }

    void verify_cells()
    {
        std::vector<int> st(g.d.ncell + 1), so(g.d.num);
        std::vector<Real> hx(3 * g.d.num);
        CUDA_CHECK(cudaMemcpy(st.data(), g.cellStart, (g.d.ncell + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(so.data(), g.sorted, g.d.num * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hx.data(), g.x, 3 * g.d.num * sizeof(Real), cudaMemcpyDeviceToHost));

        std::vector<int> seen(g.d.num, 0);
        int dup = 0, outOfCell = 0, missing = 0;
        for (int c = 0; c < g.d.ncell; ++c)
        {
            if (st[c] < 0 || st[c] > st[c + 1] || st[c + 1] > g.d.num) { printf("  cellStart bad at %d\n", c); break; }
            for (int a = st[c]; a < st[c + 1]; ++a)
            {
                const int p = so[a];
                if (p < 0 || p >= g.d.num) continue;
                seen[p]++;
                const int cc = host_cell(hx[p], hx[g.d.num + p], hx[2 * g.d.num + p]);
                if (cc != c) outOfCell++;
            }
        }
        for (int p = 0; p < g.d.num; ++p) { if (seen[p] == 0) missing++; if (seen[p] > 1) dup += seen[p] - 1; }
        printf("[verify] start[0]=%d start[N]=%d dup=%d missing=%d wrongCell=%d\n",
               st[0], st[g.d.ncell], dup, missing, outOfCell);
    }

    // ---------------- one time step ----------------
    void step(int itime)
    {
        const bool dump = (dump_step && itime == 0);

        launch_prepare(g, 0);

        CUDA_CHECK(cudaMemsetAsync(g.cellCur, 0, g.d.ncell * sizeof(int), 0));
        CUDA_CHECK(cudaMemsetAsync(g.cellSolid, 0, g.d.ncell * sizeof(int), 0));
        CUDA_CHECK(cudaMemsetAsync(g.cellFluid, 0, g.d.ncell * sizeof(int), 0));
        launch_cell_hist(g, g.cellCur, g.cellSolid, g.cellFluid, 0);
        launch_cell_scan(g.cellStart, g.cellCur, g.d.ncell, 0);
        launch_cell_scatter(g, g.cellCur, 0);
        launch_cell_sort(g.cellStart, g.sorted, g.d.ncell, 0);

        launch_pass1(g, 0);
        if (dump)
        {
            const char* fn[9] = {"F11","F12","F13","F21","F22","F23","F31","F32","F33"};
            const char* dn[9] = {"DFdt11","DFdt12","DFdt13","DFdt21","DFdt22","DFdt23","DFdt31","DFdt32","DFdt33"};
            for (int k = 0; k < 9; ++k) { dump_real(fn[k], g.F + (size_t)k * g.d.num); dump_real(dn[k], g.DFdt + (size_t)k * g.d.num); }
            dump_real("drhodt", g.drhodt);
            dump_real("dvxdt0", g.dvxdt); dump_real("dvxdt1", g.dvxdt + g.d.num); dump_real("dvxdt2", g.dvxdt + 2 * g.d.num);
            dump_real("viscos0", g.viscos); dump_real("viscos1", g.viscos + g.d.num); dump_real("viscos2", g.viscos + 2 * g.d.num);
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
            dump_real("SForce0", g.SForce); dump_real("SForce1", g.SForce + g.d.num); dump_real("SForce2", g.SForce + 2 * g.d.num);
            dump_real("PIJW0", g.PIJW); dump_real("PIJW1", g.PIJW + g.d.num); dump_real("PIJW2", g.PIJW + 2 * g.d.num);
            printf("[dumpstep] stage3 written\n");
        }

        launch_update(g, 0);
        if (dump)
        {
            dump_real("x0", g.x); dump_real("x1", g.x + g.d.num); dump_real("x2", g.x + 2 * g.d.num);
            dump_real("vx0", g.vx); dump_real("vx1", g.vx + g.d.num); dump_real("vx2", g.vx + 2 * g.d.num);
            dump_real("rho_f", g.rho);
        }

        // C5: cudaDeviceSynchronize() reports execution errors, but a launch
        // configuration failure (too many registers, bad grid, ...) only shows
        // up in cudaGetLastError(), and the kernel would silently not have run.
        CUDA_CHECK(cudaGetLastError());
    }

    // ---------------- result<N>.dat ----------------
    // ---------------- result<N>.sphbin ----------------
    void fetch_state()
    {
        const int n = g.d.num;
        ox.assign(n, 0.0); oy.assign(n, 0.0); oz.assign(n, 0.0); op.assign(n, 0.0);
        auto pull = [&](double* dst, const Real* src)
        {
            std::vector<Real> tmp((size_t)n);
            CUDA_CHECK(cudaMemcpy(tmp.data(), src, (size_t)n * sizeof(Real), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) dst[i] = (double)tmp[i];
        };
        pull(ox.data(), g.x);
        pull(oy.data(), g.x + n);
        pull(oz.data(), g.x + 2 * n);
        std::vector<double> pp((size_t)n, 0.0);
        pull(pp.data(), g.p);
        // Pia is AoS (12 Reals per solid): the writer stores the diagonal sum.
        std::vector<Real> tmp((size_t)SPH_PIA_STRIDE * g.d.solnum);
        CUDA_CHECK(cudaMemcpy(tmp.data(), g.Pia, tmp.size() * sizeof(Real), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i)
        {
            if (i < g.d.ntotal) op[i] = pp[i];
            else if (i < g.d.nall)
            {
                const size_t k = (size_t)(i - g.d.ntotal) * SPH_PIA_STRIDE;
                op[i] = (double)(tmp[k] + tmp[k + 4] + tmp[k + 8]);
            }
            else op[i] = 0.0;
        }
    }

    // Binary result: 128-byte header + num * 4 values (x, y, z, ppp).
    // itype is implied by the block split stored in the header
    // (fluid [0,ntotal) | solid [ntotal,nall) | wall [nall,num)).
    // SPH_OUT_F32=1 stores the four values as float32 (half the size).
    void write_result(int idx)
    {
        const bool f32 = (getenv("SPH_OUT_F32") != nullptr);
        char name[128];
        snprintf(name, sizeof(name), "result%d.sphbin", idx);
        FILE* f = fopen(name, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", name); return; }

        unsigned char h[128] = {0};
        const char magic[8] = { 'S','P','H','B','I','N','\0','\0' };
        memcpy(h, magic, 8);
        const uint32_t ver = 1, hs = 128, num = (uint32_t)g.d.num,
                       ntotal = (uint32_t)g.d.ntotal, solnum = (uint32_t)g.d.solnum,
                       nfields = 4, step = (uint32_t)(idx * 500),
                       flags = 2u | (f32 ? 4u : 0u);
        memcpy(h + 8, &ver, 4);      memcpy(h + 12, &hs, 4);
        memcpy(h + 16, &num, 4);     memcpy(h + 20, &ntotal, 4);
        memcpy(h + 24, &solnum, 4);  memcpy(h + 28, &nfields, 4);
        memcpy(h + 32, &step, 4);    memcpy(h + 36, &flags, 4);
        const double hsml = SPH_HSML, dt = SPH_DT;
        memcpy(h + 40, &hsml, 8); memcpy(h + 48, &dt, 8);
        const double grid[6] = { g.d.gx0, g.d.gy0, g.d.gz0,
                                 g.d.gx0 + g.d.nx * SPH_CS,
                                 g.d.gy0 + g.d.ny * SPH_CS,
                                 g.d.gz0 + g.d.nz * SPH_CS };
        memcpy(h + 56, grid, 48);
        fwrite(h, 1, 128, f);

        const size_t rec = 4;
        if (f32)
        {
            std::vector<float> buf(rec * 65536);
            size_t used = 0;
            for (int i = 0; i < g.d.num; ++i)
            {
                if (used + rec > buf.size()) { fwrite(buf.data(), sizeof(float), used, f); used = 0; }
                buf[used++] = (float)ox[i]; buf[used++] = (float)oy[i];
                buf[used++] = (float)oz[i]; buf[used++] = (float)op[i];
            }
            if (used) fwrite(buf.data(), sizeof(float), used, f);
        }
        else
        {
            std::vector<double> buf(rec * 65536);
            size_t used = 0;
            for (int i = 0; i < g.d.num; ++i)
            {
                if (used + rec > buf.size()) { fwrite(buf.data(), sizeof(double), used, f); used = 0; }
                buf[used++] = ox[i]; buf[used++] = oy[i];
                buf[used++] = oz[i]; buf[used++] = op[i];
            }
            if (used) fwrite(buf.data(), sizeof(double), used, f);
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
    if (!S.load_deck(input_path)) return 1;   // parse + size + allocate + upload
    launch_copy_xf(S.g, 0);                    // A4: float copy before the first search

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
        if (target < 0 || target >= S.g.d.num) { fprintf(stderr, "SPH_NBRLIST: particle %d out of range\n", target); return 2; }
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
                std::vector<int> fc(S.g.d.ntotal);
                CUDA_CHECK(cudaMemcpy(fc.data(), S.g.fcnt, (size_t)S.g.d.ntotal * sizeof(int),
                                      cudaMemcpyDeviceToHost));
                long long tot = 0; int mx = 0, mn = 1 << 30, ov = 0;
                for (int v : fc)
                {
                    if (v < 0) { ++ov; continue; }
                    tot += v; if (v > mx) mx = v; if (v < mn) mn = v;
                }
                printf("[bench] fluid neighbours: mean=%.1f min=%d max=%d overflow=%d (width %d)\n",
                       (double)tot / (double)(S.g.d.ntotal - ov), mn, mx, ov, S.g.fwidth);
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
        if (itime % 500 == 0)
        {
            // the list built inside the previous step predates that step's position
            // update; rebuild it so the printed count belongs to the current positions
            if (itime > 0) S.build_cells();
            CUDA_CHECK(cudaMemset(S.d_niac, 0, sizeof(int)));
            launch_count_pairs(S.g, S.d_niac, 0);
            int niac = 0;
            CUDA_CHECK(cudaMemcpy(&niac, S.d_niac, sizeof(int), cudaMemcpyDeviceToHost));
            printf("T= %d\tpairs=%d\n", itime / 500, niac / 2);
        }

        if (profile)
        {
            CUDA_CHECK(cudaEventRecord(ev[0], 0));
            launch_prepare(S.g, 0);
            CUDA_CHECK(cudaEventRecord(ev[1], 0));
        CUDA_CHECK(cudaMemsetAsync(S.g.cellCur, 0, S.g.d.ncell * sizeof(int), 0));
        CUDA_CHECK(cudaMemsetAsync(S.g.cellSolid, 0, S.g.d.ncell * sizeof(int), 0));
        CUDA_CHECK(cudaMemsetAsync(S.g.cellFluid, 0, S.g.d.ncell * sizeof(int), 0));
            launch_cell_hist(S.g, S.g.cellCur, S.g.cellSolid, S.g.cellFluid, 0);
            launch_cell_scan(S.g.cellStart, S.g.cellCur, S.g.d.ncell, 0);
            launch_cell_scatter(S.g, S.g.cellCur, 0);
        launch_cell_sort(S.g.cellStart, S.g.sorted, S.g.d.ncell, 0);
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

        if (itime % 500 == 0)
        {
            S.fetch_state();
            S.write_result(itime / 500);
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
