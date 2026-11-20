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
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <string>
#include <future>

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

// ---------------------------------------------------------------------------
// Deck container checks shared by both drivers.
//
// M2: the header records the smoothing length and the time step the deck was
// converted for, but the solver's h and dt are compile-time constants, so a
// deck made with --hsml 0.005 would otherwise run silently with 0.01 -- every
// kernel distance wrong by a factor of two.  Refuse the mismatch instead.
// M3: the single-GPU loader validates the fluid|solid|wall block order; the
// decomposed one only looked at each rank's own rows, so it trusted the header
// counts.  Both use these helpers now.
// ---------------------------------------------------------------------------
static int deck_class_of(double itype)
{
    return (itype > 0.0) ? 0 : ((itype < -10.0) ? 1 : 2);
}

static bool header_metrics_ok(const unsigned char* hdr, const char* path)
{
    double hsml = 0.0, dt = 0.0;
    memcpy(&hsml, hdr + 40, 8);
    memcpy(&dt, hdr + 48, 8);
    if (hsml == 0.0 && dt == 0.0) return true;     // writer had no metrics
    const double hrel = std::fabs(hsml - SPH_HSML) / SPH_HSML;
    const double drel = std::fabs(dt - SPH_DT) / SPH_DT;
    if (hrel > 1e-9 || drel > 1e-9)
    {
        fprintf(stderr, "[input] %s: deck was written for hsml=%g dt=%g, this build "
                        "uses hsml=%g dt=%g (compile-time constants); rebuild the deck "
                        "with --hsml %g --dt %g or recompile\n",
                path, hsml, dt, (double)SPH_HSML, (double)SPH_DT,
                (double)SPH_HSML, (double)SPH_DT);
        return false;
    }
    return true;
}

// Validates the global block layout ([fluid][solid][wall]) of the first `num`
// records and returns the per-block counts.
static bool deck_blocks_ok(const std::vector<double>& rows, int num, const char* path,
                           int& nfl, int& ns, int& nw)
{
    const int nf = 14;
    nfl = ns = nw = 0;
    for (int i = 0; i < num; ++i)
    {
        const int c = deck_class_of(rows[(size_t)i * nf + 9]);
        if (c == 0) ++nfl; else if (c == 1) ++ns; else ++nw;
    }
    for (int i = 0; i < num; ++i)
    {
        const double t = rows[(size_t)i * nf + 9];
        const int c = deck_class_of(t);
        const int want = (i < nfl) ? 0 : ((i < nfl + ns) ? 1 : 2);
        if (c != want)
        {
            fprintf(stderr, "[input] %s: particle %d: itype %g is outside its block "
                            "(fluid=%d solid=%d wall=%d); the deck must be ordered "
                            "fluid|solid|wall\n", path, i, t, nfl, ns, nw);
            return false;
        }
    }
    return true;
}

template <class Real>
struct Solver
{
    SphPtrsT<Real> g = SphPtrsT<Real>();
    int*  d_niac = nullptr;
    int*  d_nbr  = nullptr;
    int*  d_scan_tmp = nullptr;      // P15: cub scan scratch
    size_t d_scan_bytes = 0;
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
        const size_t n1 = (size_t)g.d.numP;   // owned + ghosts (D1)
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
        // D1: one Pia slot per particle index >= ntotal (owned solids, owned
        // walls and the ghost block), so the kernels keep indexing by q - ntotal.
        // D1 / P13: one slot per *solid* particle -- owned solids, then the ghost
        // solids (`sph_sol_slot`).  Walls and ghost fluids take no slot, which is
        // where the 92 MiB (1203k deck) / 1.47 GiB (19.3 M) of the review went.
        const size_t nsol = (size_t)g.d.solnum + (size_t)g.d.ngs;
        g.Pia    = dev_alloc_real((size_t)SPH_PIA_STRIDE * nsol);   // AoS, see sph.h

        g.solidPack = dev_alloc_real((size_t)14 * nsol);
        g.gid = nullptr;               // D1 only; set by the decomposed driver

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
        g.cellCur   = (int*)dev_alloc((size_t)(g.d.ncell + 1) * sizeof(int));
        g.cellSolid = (int*)dev_alloc(g.d.ncell * sizeof(int));
        g.cellFluid = (int*)dev_alloc(g.d.ncell * sizeof(int));
        g.sorted    = (int*)dev_alloc((size_t)g.d.numP * sizeof(int));

        g.xf = (float*)dev_alloc_raw(n3 * sizeof(float));   // A4, filled below

        g.u    = dev_alloc_real(n3);       // fp32 accuracy support, see sph.h
        g.xc   = dev_alloc_real(n3);
        g.rhoc = dev_alloc_real(n1);

        // P15: the cell scan reads ncell + 1 counts (the trailing one stays 0, so
        // the total lands in start[ncell]); cellCur is the cursor the scatter re-arms.
        d_niac    = (int*)dev_alloc(sizeof(int));
        d_nbr     = (int*)dev_alloc(g.d.num * sizeof(int));
        d_sumvecn = dev_alloc_real(n3);
        d_a       = dev_alloc_real(n9);
    }
    // ------------------------------------------------------------------
    // Repartition: drop every device buffer so the next finish_deck() can
    // rebuild the state for the new own/ghost sets at the exact new size.
    // (cudaFree + cudaMalloc once per repartition is ~50 ms; a capacity
    // scheme with high-water marks would avoid it but this is smaller and
    // there is no fragmentation risk at these sizes.)
    // ------------------------------------------------------------------
    void free_state()
    {
        void* ptrs[] = {
            g.x, g.vx, g.X0, g.vecn, g.PIJW, g.SForce, g.dvxdt, g.viscos,
            g.rho, g.irho, g.rho0, g.mass, g.p, g.itype, g.SurF, g.SurFD, g.drhodt,
            g.A, g.F, g.DFdt, g.Pia, g.solidPack,
            g.sCnt, g.sQ, g.sG, g.sB, g.sD,
            g.fnbr, g.fcnt,
            g.cellStart, g.cellCur, g.cellSolid, g.cellFluid, g.sorted,
            g.xf, g.u, g.xc, g.rhoc, g.gid,
            d_niac, d_nbr, d_sumvecn, d_a, d_scan_tmp
        };
        for (void* p : ptrs) if (p) cudaFree(p);
        g = SphPtrsT<Real>();
        d_niac = nullptr; d_nbr = nullptr; d_sumvecn = nullptr; d_a = nullptr;
        d_scan_tmp = nullptr; d_scan_bytes = 0;
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
            if (!header_metrics_ok(hdr, path)) { fclose(f); return false; }
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
    // ownNum = 0 means "this rank owns every row" (single GPU).  Otherwise the
    // rows hold ownNum owned particles followed by ghost copies, each group
    // ordered fluid|solid|wall (D1).
    bool finish_deck(const std::vector<double>& rows, const char* path, int ownNum = 0,
                     const SphGridBox* fixed = nullptr, int ngs1 = -1)
    {
        const int nf = 14;
        const int numAll = (int)(rows.size() / nf);
        const int num = (ownNum > 0 && ownNum <= numAll) ? ownNum : numAll;
        // rows holds ALL of them (own + ghosts); `num` is what this rank owns
        if (num <= 0 || (size_t)numAll * nf != rows.size())
        {
            fprintf(stderr, "[input] %s: %zu doubles is not num*14\n", path, rows.size());
            return false;
        }
        int nfl = 0, ns = 0, nw = 0;
        if (!deck_blocks_ok(rows, num, path, nfl, ns, nw)) return false;
        if (g.d.num != 0 && g.d.num != num)
        {
            fprintf(stderr, "[input] %s: header says %d records, file holds %d\n",
                    path, g.d.num, num);
            return false;
        }
        // M2/M3: for a subdivided deck the header counts describe the *whole*
        // deck; each rank's own block is checked here, the global order and the
        // header counts by the driver (d1::read_deck).
        g.d.num = num; g.d.ntotal = nfl; g.d.solnum = ns;
        // D1: single-rank decks have no ghosts, so numP == num and every stride
        // and loop bound is exactly what it was before the decomposition work.
        g.d.numP = numAll;
        g.d.gs0 = numAll;
        g.d.ngs = 0;
        for (int i = num; i < numAll; ++i)
            if (rows[(size_t)i * nf + 9] < -10.0)
            {
                if (g.d.ngs == 0) g.d.gs0 = i;
                ++g.d.ngs;
            }
        g.d.nall = nfl + ns; g.d.nwall = nw;
        // P11: how many of the ghost solids compute (layer 1) and how many A3
        // slots exist in total.  A single-GPU deck has no ghosts at all.
        g.d.ngs1 = (ngs1 >= 0 && ngs1 <= g.d.ngs) ? ngs1 : g.d.ngs;
        g.d.nsolC = ns + g.d.ngs1;

        // V2: the decomposed driver hands every rank the grid derived from the
        // whole deck, so a particle is binned identically everywhere (and the
        // grid no longer drifts when a repartition re-derives it).
        const bool fixed_grid = (fixed != nullptr && fixed->ok());
        double mn[3] = { 1e300, 1e300, 1e300 }, mx[3] = { -1e300, -1e300, -1e300 };
        for (int i = 0; i < numAll; ++i)
            for (int c = 0; c < 3; ++c)
            {
                const double v = rows[(size_t)i * nf + c];
                if (v < mn[c]) mn[c] = v;
                if (v > mx[c]) mx[c] = v;
            }
        const double cs = SPH_CS;
        const double margin = cs + 0.02;
        if (fixed_grid)
        {
            g.d.gx0 = fixed->org[0]; g.d.gy0 = fixed->org[1]; g.d.gz0 = fixed->org[2];
            g.d.nx = fixed->nx; g.d.ny = fixed->ny; g.d.nz = fixed->nz;
        }
        else
        {
            g.d.gx0 = mn[0] - margin; g.d.gy0 = mn[1] - margin; g.d.gz0 = mn[2] - margin;
            const double gx1 = mx[0] + margin, gy1 = mx[1] + margin, gz1 = mx[2] + margin;
            g.d.nx = (int)floor((gx1 - g.d.gx0) / cs) + 1;
            g.d.ny = (int)floor((gy1 - g.d.gy0) / cs) + 1;
            g.d.nz = (int)floor((gz1 - g.d.gz0) / cs) + 1;
        }
        g.d.ncell = g.d.nx * g.d.ny * g.d.nz;
        fprintf(stderr, "[input] %s: num=%d fluid=%d solid=%d wall=%d grid=%dx%dx%d%s "
                        "slots=%d ghosts=%d layer1=%d\n",
                path, g.d.num, g.d.ntotal, g.d.solnum, g.d.nwall,
                g.d.nx, g.d.ny, g.d.nz, fixed_grid ? " (global, V2)" : "",
                g.d.nsolC, g.d.ngs, g.d.ngs1);

        alloc_state();

        std::vector<Real> t3((size_t)3 * numAll);
        auto up3 = [&](Real* dst, int c)
        {
            for (int i = 0; i < numAll; ++i)
                for (int k = 0; k < 3; ++k)
                    t3[(size_t)k * numAll + i] = (Real)rows[(size_t)i * nf + c + k];
            CUDA_CHECK(cudaMemcpy(dst, t3.data(), t3.size() * sizeof(Real), cudaMemcpyHostToDevice));
        };
        up3(g.x, 0); up3(g.vx, 4); up3(g.X0, 10);

        std::vector<Real> t1(numAll);
        auto up1 = [&](Real* dst, int c)
        {
            for (int i = 0; i < numAll; ++i) t1[i] = (Real)rows[(size_t)i * nf + c];
            CUDA_CHECK(cudaMemcpy(dst, t1.data(), t1.size() * sizeof(Real), cudaMemcpyHostToDevice));
        };
        up1(g.p, 3); up1(g.mass, 8); up1(g.itype, 9); up1(g.rho0, 13);

        // rho clamp of the reference solver
        for (int i = 0; i < numAll; ++i)
        {
            double r = rows[(size_t)i * nf + 7];
            if (r < 960.0) r = 960.0;
            t1[i] = (Real)r;
        }
        CUDA_CHECK(cudaMemcpy(g.rho, t1.data(), t1.size() * sizeof(Real), cudaMemcpyHostToDevice));

        // fp32 displacement x - X0 of the solid block, formed in double first
        std::vector<double> hu((size_t)3 * numAll, 0.0);
        for (int i = 0; i < numAll; ++i)
            if (rows[(size_t)i * nf + 9] < -10.0)
                for (int k = 0; k < 3; ++k)
                    hu[(size_t)k * numAll + i] = rows[(size_t)i * nf + k] - rows[(size_t)i * nf + 10 + k];
        {
            std::vector<Real> t(hu.size());
            for (size_t k = 0; k < hu.size(); ++k) t[k] = (Real)hu[k];
            CUDA_CHECK(cudaMemcpy(g.u, t.data(), t.size() * sizeof(Real), cudaMemcpyHostToDevice));
        }

        itype_host.resize(numAll);
        for (int i = 0; i < numAll; ++i) itype_host[i] = rows[(size_t)i * nf + 9];
        return true;
    }

    // ---------------- grid ----------------
    // build_cells_for() bins whatever positions `s.x` points at; build_cells()
    // is the live configuration.  build_solid_lists() calls it with x = X0.
    void build_cells_for(const SphPtrsT<Real>& s)
    {
        CUDA_CHECK(cudaMemset(g.cellCur, 0, (size_t)(g.d.ncell + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemset(g.cellSolid, 0, g.d.ncell * sizeof(int)));
        CUDA_CHECK(cudaMemset(g.cellFluid, 0, g.d.ncell * sizeof(int)));
        launch_cell_hist(s, g.cellCur, g.cellSolid, g.cellFluid, 0);
        launch_cell_scan(g.cellStart, g.cellCur, g.d.ncell, d_scan_tmp, d_scan_bytes, 0);
        // C1: the *same* positions as the histogram -- with the live x this list
        // would disagree with the counts whenever the solid has moved away from
        // its reference position (the A3 rebuild after a repartition hits this:
        // the pair count then drifts and the ELL width changes).
        launch_cell_scatter(s, g.cellCur, 0);
        launch_cell_sort(g.cellStart, g.sorted, g.d.ncell, g.gid, 0);
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
        // a repartition rebuilds the list for the new own-solid set
        if (g.sCnt) { cudaFree(g.sCnt); g.sCnt = nullptr; }
        if (g.sQ)   { cudaFree(g.sQ);   g.sQ   = nullptr; }
        if (g.sG)   { cudaFree(g.sG);   g.sG   = nullptr; }
        if (g.sB)   { cudaFree(g.sB);   g.sB   = nullptr; }
        if (g.sD)   { cudaFree(g.sD);   g.sD   = nullptr; }
        g.npair = 0; g.swidth = 0; g.sw = 0;
        // P11: the list covers every solid slot that computes -- the owned solids
        // plus the layer-1 ghost solids -- because a ghost runs its own
        // constitution and therefore needs exactly the partners its owner has.
        const int n = g.d.nsolC;

        // C12: the A6 layout is a global and every rank has to choose the same
        // one, so a rank that owns no solid at all still has to take part in the
        // decision (contributing a fat size of 0) -- otherwise it walks past the
        // barrier the other ranks are waiting at and the whole run desyncs.
        if (n <= 0)
        {
            if (sph_a3_lean < 0)
            {
                double limit_mb = 4096.0;
                if (const char* e = getenv("SPH_A3_LIMIT_MB")) limit_mb = atof(e);
                const double fat_mb = sph_a3_fat_hook ? sph_a3_fat_hook(0.0) : 0.0;
                sph_a3_lean = (fat_mb > limit_mb) ? 1 : 0;
            }
            return;
        }

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

        // A6 / C12: the fat list stores 9 Reals per slot; the lean one stores only
        // the partner index.  Both are bit-identical (see sph.h), so the choice is
        // a pure memory/speed trade.
        //
        // C16: this decision is collective, so it has to happen on *every* rank --
        // including one whose computing solids have no pair at all.  It used to sit
        // after the `width <= 0` return below, and such a rank then walked past the
        // hook's barriers and arrived at the next one (the E1 exchange) out of phase.
        if (sph_a3_lean < 0)
        {
            double limit_mb = 4096.0;
            if (const char* e = getenv("SPH_A3_LIMIT_MB")) limit_mb = atof(e);
            double fat_mb = (double)g.sw * 9.0 * (double)sizeof(Real) / 1048576.0;
            if (sph_a3_fat_hook) fat_mb = sph_a3_fat_hook(fat_mb);
            sph_a3_lean = (fat_mb > limit_mb) ? 1 : 0;
            printf("[A3] list mode: %s (fat would be %.0f MB, limit %.0f MB)\n",
                   sph_a3_lean ? "lean" : "fat", fat_mb, limit_mb);
        }

        // C19: the per-solid counts exist even when no pair was found, so a rank
        // with an empty list can still run the *static* kernels (over an empty
        // list) instead of silently switching to the cell-walk path, which is not
        // bit-identical to it.
        g.sCnt = (int*)dev_alloc_raw((size_t)n * sizeof(int));
        CUDA_CHECK(cudaMemcpy(g.sCnt, h.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
        if (width <= 0) { build_cells(); return; }      // nothing to store

        g.sQ   = (int*)dev_alloc_raw((size_t)g.sw * sizeof(int));
        // A6: in lean mode the per-pair weights are recomputed from the pack, so
        // the three 3*sw arrays (72 B/pair) are never allocated.
        g.sG   = sph_a3_lean ? nullptr : dev_alloc_real_raw((size_t)3 * g.sw);
        g.sB   = sph_a3_lean ? nullptr : dev_alloc_real_raw((size_t)3 * g.sw);
        g.sD   = sph_a3_lean ? nullptr : dev_alloc_real_raw((size_t)3 * g.sw);

        int* d_cur = (int*)dev_alloc_raw((size_t)n * sizeof(int));
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

        const double per_slot = (double)sizeof(int) + (sph_a3_lean ? 0.0 : 9.0 * sizeof(Real));
        const double mb = (double)g.sw * per_slot / 1048576.0;
        printf("[A3] solid pair list: %d pairs, %.1f per computing slot, "
               "ELL width %d (padding %.1f%%), %.1f MB%s\n",
               g.npair, (double)g.npair / (double)n, g.swidth,
               100.0 * ((double)g.sw / (double)(g.npair > 0 ? g.npair : 1) - 1.0), mb,
               sph_a3_lean ? " (lean: index only, weights recomputed)" : "");
    }

    void init_reference_state()
    {
        // C15: the arrays are numP-strided (owned + ghosts), so the clear has to
        // cover numP slots.  Harmless before only because dev_alloc() zeroes the
        // whole allocation already.
        CUDA_CHECK(cudaMemset(d_sumvecn, 0, 3 * g.d.numP * sizeof(Real)));
        launch_nor_sumvecn(g, d_sumvecn, 0);
        launch_nor_normalize(g, d_sumvecn, 0);

        CUDA_CHECK(cudaMemset(d_a, 0, 9 * g.d.numP * sizeof(Real)));
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

        CUDA_CHECK(cudaMemsetAsync(g.cellCur, 0, (size_t)(g.d.ncell + 1) * sizeof(int), 0));
        CUDA_CHECK(cudaMemsetAsync(g.cellSolid, 0, g.d.ncell * sizeof(int), 0));
        CUDA_CHECK(cudaMemsetAsync(g.cellFluid, 0, g.d.ncell * sizeof(int), 0));
        launch_cell_hist(g, g.cellCur, g.cellSolid, g.cellFluid, 0);
        launch_cell_scan(g.cellStart, g.cellCur, g.d.ncell, d_scan_tmp, d_scan_bytes, 0);
        launch_cell_scatter(g, g.cellCur, 0);
        launch_cell_sort(g.cellStart, g.sorted, g.d.ncell, g.gid, 0);

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

// ===========================================================================
//  D1: multi-GPU driver
//
//  One process, K host threads, one CUDA device per thread.  The deck is read
//  once and cut into an nx*ny*nz block grid whose *compute* weight is balanced
//  (fluid 1.0, solid 0.5, wall 0.0 -- a wall costs no per-step kernel work, it
//  only takes memory), and every rank then runs the ordinary Solver on
//
//      rows_r = [ own particles (deck order) | ghost copies (deck order) ]
//
//  Ghosts are the neighbours' particles within 3*hsml + skin of the block, plus
//  (C14) the solids whose *reference* position is in the reference shell, so
//  the A3 pairs of the static solid list survive too.  Owned particles keep the
//  deck's fluid|solid|wall order, so every `p < s.d.ntotal` style bound in the
//  kernels still means "my fluid" and ghost slots are only ever read.
//
//  Per step a rank exchanges two record blocks (E1 fluid+solid, E2 solid) with
//  each of its up to 26 peers; see sph_kernels.cu for the layouts.  Packing is
//  a gather kernel into a device staging block, ONE cudaMemcpyAsync of the whole
//  block into a pinned host block, a barrier over all K threads, then a pinned
//  -> device copy and a scatter kernel into the ghost slots on the receiving
//  side.  A second barrier closes the exchange, because the sender reuses the
//  same pinned block at the next exchange point.
//
//  The single-GPU path above (run_case) is untouched: it never allocates a
//  gid table, never sets numP != num and never enters this namespace.
// ===========================================================================
#if SPH_HAVE_FP64
namespace d1
{

static inline double now_s()
{
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// reusable barrier over the K rank threads
// ---------------------------------------------------------------------------
// Adaptive barrier over the K rank threads: spin for a short while (the ranks
// are normally within a few hundred us, and a real block+unblock costs
// 0.4-1.4 ms on Windows), then fall back to a condition variable so that a
// genuinely slow rank does not keep the others burning CPU.  A pure
// condition-variable barrier measured 4.20 ms/step at K=8, a pure spin
// 3.48 ms/step on an idle machine but unstable when another process competes
// for the display GPU; this hybrid keeps the fast path and stays well behaved.
// barrier variant, from SPH_BARRIER (measurement switch; see Barrier::wait)
static int g_barrier_mode = 0;      // 0 = adaptive (default), 1 = pure spin, 2 = block

struct Barrier
{
    std::atomic<int> arrived{ 0 };
    std::atomic<int> gen{ 0 };
    std::mutex m;
    std::condition_variable cv;
    int K = 1;

    void wait()
    {
        if (K <= 1) return;
        const int g = gen.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == K)
        {
            arrived.store(0, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(m);   // pairs with the waiter check
                gen.fetch_add(1, std::memory_order_release);
            }
            cv.notify_all();
            return;
        }
        const int mode = g_barrier_mode;
        if (mode == 1)
        {
            while (gen.load(std::memory_order_acquire) == g) std::this_thread::yield();
            return;
        }
        if (mode == 0)
        {
            for (int i = 0; i < 3000; ++i)
            {
                if (gen.load(std::memory_order_acquire) != g) return;
                std::this_thread::yield();
            }
        }
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return gen.load(std::memory_order_acquire) != g; });
    }
};

// ---------------------------------------------------------------------------
// one (sender, receiver) pair.  send* lists are local indices of particles
// this rank owns, recv* lists are the ghost slots that arrive from `peer`.
// Both sides are built in the same scan of the receiver's ghost block, so the
// two lists are in the same order.
//
// All the records of all peers live in a few contiguous blocks per rank, so a
// whole exchange point is ONE host<->device copy; off[] is where this link's
// block starts in my send block, recvOff[] where the incoming block goes in my
// receive block, peerOff[] where the peer keeps the block it sends me.
// ---------------------------------------------------------------------------
struct DLink
{
    int peer = -1;
    std::vector<int> sendF, sendS, sendW;
    std::vector<int> recvF, recvS, recvW;
    int off[5] = { 0, 0, 0, 0, 0 };
    int recvOff[5] = { 0, 0, 0, 0, 0 };
    size_t peerE1 = 0;                 // P17: the peer's E1 block length (doubles)
    int peerOff[5] = { 0, 0, 0, 0, 0 };
};

struct Rank
{
    int id = 0, gpu = 0, ownNum = 0, numP = 0, ngs1 = 0, ngs2 = 0;
    std::vector<int>    gid;       // [numP] deck index of every local slot
    std::vector<double> rows;      // 14 * numP
    std::vector<DLink>  links;
    std::string         label;
    Solver<double>      S;

    // Exchange buffers, allocated once: two pinned send blocks (P10: consecutive
    // exchange points alternate between them, so a peer can still be relaying
    // from one while the next point fills the other), one pinned receive block
    // (each holds every peer's records side by side) and one device staging
    // block used for both directions.  Between the two copies the receiving rank
    // relays the peers' blocks into its own receive block with plain CPU memcpy
    // -- much cheaper than one cudaMemcpyAsync per peer, which on Windows/WDDM
    // costs ~150 us even for a few kB (3-D blocks have up to 26 peers, and that
    // per-copy cost used to dominate the step).
    double* hSendB[2] = { nullptr, nullptr };   // pinned, double buffered
    int     sendSlot = 0;                       // half this rank writes
    int     xchSeq = 0;                         // exchange points done so far
    double* hRecv = nullptr;       // pinned
    double* dStage = nullptr;      // device
    size_t  sendCap = 0, recvCap = 0;      // doubles
    // per exchange point (0 E1-fluid, 1 E1-solid, 2 E2, 3 E3, 4 static)
    int  nSend[5] = { 0, 0, 0, 0, 0 }, nRecv[5] = { 0, 0, 0, 0, 0 };
    int* dSendIdx[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
    int* dSendOff[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
    int* dRecvIdx[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
    int* dRecvOff[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };

    double step_s = 0.0, xch_s = 0.0, t_start = 0.0, t_end = 0.0;
    long long reparts = 0;         // repartition events this rank took part in
    double repart_s = 0.0;         // wall spent in them
    // C9: the ghost set is only complete at the instant it is chosen, so every
    // rank tracks how far its owned particles have moved since then and rebuilds
    // before a pair can be lost.  dXBuild is [3*num], component-major; dMaxDisp
    // is the reduced Chebyshev distance as a positive double bit pattern.
    double* dXBuild = nullptr;
    unsigned long long* dMaxDisp = nullptr;
    bool rebuild_due = false;      // agreed across ranks, set by the step
    size_t nSendE1 = 0;            // P17: doubles of this rank's E1 block
    double disp = 0.0;             // exact global max displacement (E1 tail)
    double last_disp = 0.0;        // C9: what triggered the pending rebuild
    // P17: the check is exact every step and costs nothing extra -- the value rides
    // in the tail of the E1 send block, so it is copied and relayed with the
    // records (no synchronous 8-byte readback, no rate extrapolation, no C18
    // overshoot).
    // exchange breakdown (profiling only)
    double x_bar = 0.0;      // time parked on the exchange barriers
    double x_g = 0.0, x_c = 0.0, x_r = 0.0, x_h = 0.0;   // gather / copy / relay / copy-back
    double ph[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, ph_t = 0.0;
};

struct Job
{
    int K = 1, maxStep = 0;
    bool prof = false;
    int gnx = 1, gny = 1, gnz = 1;      // chosen block grid
    long long ghost_total = 0;          // sum of ghost slots over the ranks
    std::vector<Rank> ranks;
    Barrier bar;
    // P14: two staging buffers so the ranks can keep stepping while a result file
    // is being written.  outBusy[s] says stage s still has a writer running.
    struct OutBuf { std::vector<double> gx, gy, gz, gp; };
    OutBuf out[2];
    bool outBusy[2] = { false, false };
    std::future<void> outFut[2];
    int num = 0, ntotal = 0, solnum = 0;
    double grid[6] = { 0, 0, 0, 0, 0, 0 };   // global box, for the result header
    std::string deck;
    std::mutex print_m;

    // ---- repartition (SPH_D1_REPART=N) --------------------------------
    std::vector<double> deckRows;       // the whole deck, kept for rebuilds
    std::vector<int>    owner;          // [num] current ownership
    std::vector<double> box;            // [6*K] current block bounds
    std::vector<double> pos;            // [3*num] current positions (gathered)
    std::vector<double> mig;            // [20*num] dynamic state, by gid
    int  repart = 0;                    // 0 = never
    bool regrid = false;                // re-search the candidate grids too
    long long repart_count = 0;
    double repart_s = 0.0;              // wall spent in repartition (rank 0)

    // V2: the neighbour grid every rank bins into (derived once from the deck)
    SphGridBox cellgrid;

    // ---- C9: skin + displacement trigger -----------------------------
    // skin   = how far the ghost shell is widened past 3h (metres, 0 = the old
    //          behaviour: ghosts chosen once, never revalidated).
    // disp   = [2*K] per-rank max displacement, double buffered by step parity
    //          so a rank can never overwrite a slot a slower rank still reads.
    double skin = 0.0;
    std::vector<double> disp;
    std::vector<double> a3_fat;         // [K] C12: per-rank fat list size (MB)
};

// ---------------------------------------------------------------------------
// C12: the A6 lean/fat layout flag is a global (the kernels read it), so every
// rank must reach the same answer or the ranks disagree on what sG/sB/sD mean.
// The rank thread announces its own fat size, the ranks meet at the barrier,
// and each of them gets the maximum back -- the same decision everywhere.
// ---------------------------------------------------------------------------
static thread_local Job* tl_job = nullptr;
static thread_local int  tl_rank = -1;

static double a3_fat_agreed(double mb)
{
    Job* J = tl_job;
    if (!J || J->K <= 1 || tl_rank < 0) return mb;
    J->a3_fat[(size_t)tl_rank] = mb;
    J->bar.wait();
    double mx = mb;
    for (int r = 0; r < J->K; ++r) mx = std::max(mx, J->a3_fat[(size_t)r]);
    J->bar.wait();
    return mx;
}

// ---------------------------------------------------------------------------
// deck reader for the driver (the full file, all particles)
// ---------------------------------------------------------------------------
static bool read_deck(const char* path, std::vector<double>& rows,
                      int& num, int& ntotal, int& solnum)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[D1] cannot open %s\n", path); return false; }
    unsigned char hdr[128] = {0};
    const size_t got = fread(hdr, 1, sizeof(hdr), f);
    const bool binary = (got == sizeof(hdr) && memcmp(hdr, "SPHBIN\0\0", 8) == 0);
    if (binary)
    {
        uint32_t ver, hs, n, nf;
        memcpy(&ver, hdr + 8, 4);   memcpy(&hs, hdr + 12, 4);
        memcpy(&n, hdr + 16, 4);    memcpy(&nf, hdr + 28, 4);
        memcpy(&ntotal, hdr + 20, 4); memcpy(&solnum, hdr + 24, 4);
        if (ver != 1 || hs != 128 || nf != 14)
        {
            fprintf(stderr, "[D1] %s: unsupported container\n", path);
            fclose(f); return false;
        }
        if (!header_metrics_ok(hdr, path)) { fclose(f); return false; }
        num = (int)n;
        rows.resize((size_t)num * 14);
        if (fread(rows.data(), sizeof(double), rows.size(), f) != rows.size())
        {
            fprintf(stderr, "[D1] %s: short read\n", path);
            fclose(f); return false;
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
        rows.clear();
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
        num = (int)(rows.size() / 14);
        ntotal = 0; solnum = 0;
        for (int i = 0; i < num; ++i)
        {
            const double t = rows[(size_t)i * 14 + 9];
            if (t > 0.0) ++ntotal; else if (t < -10.0) ++solnum;
        }
    }
    fclose(f);
    if (!(num > 0 && (size_t)num * 14 == rows.size())) return false;

    // M3: validate the *whole* deck once, here.  The single-GPU loader has always
    // done this; the decomposed one only checked each rank's own rows and took
    // the header counts on trust, so a hand-made deck with a misplaced record
    // could still be cut up and run.
    int nfl = 0, ns = 0, nw = 0;
    if (!deck_blocks_ok(rows, num, path, nfl, ns, nw)) return false;
    if (binary && (ntotal != nfl || solnum != ns))
    {
        fprintf(stderr, "[D1] %s: header counts do not match the records "
                        "(header: %d fluid %d solid; data: %d fluid %d solid)\n",
                path, ntotal, solnum, nfl, ns);
        return false;
    }
    ntotal = nfl; solnum = ns;
    return true;
}

// ---------------------------------------------------------------------------
// partition + per-rank decks + link lists.  Runs on the host before any CUDA
// work; `rows` is the whole deck, `ownerRank`/`ownerLocal` are the global
// tables the links are built from.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// N-D block decomposition (TASK 1)
//
// The deck is cut into an nx*ny*nz grid of blocks.  Each axis is cut at equal
// *cumulative compute* (fluid 1.0, solid 0.5, wall 0.0 -- a wall costs no
// per-step kernel work, only memory), and a cut is never placed inside a group
// of equal coordinate.  The y cuts are computed inside each x slab, the z cuts
// inside each (x,y) column, so every rank owns a rectangular box.
//
// A rank's ghost set is every particle inside that box expanded by 3*hsml plus
// the skin (C9) on every face.  The expanded box is complete at the instant it
// is built: for an owned p and any q with |x_q-x_p| < 3h the x coordinate alone
// puts q inside the expanded box, and the same holds for y and z.  The skin is
// what the displacement trigger in step_d1() spends before the next rebuild --
// without it, and without a rebuild, the set silently decays.
//
// (nx,ny,nz) is chosen by evaluating the real coordinates: for every ordered
// factorisation of K the ghosts are counted and the winner is the one with the
// smallest max_r(own_weight[r] + ghost_weight[r]), i.e. the smallest predicted
// per-step load of the slowest rank.
// ---------------------------------------------------------------------------
struct Deck
{
    int num = 0;
    std::vector<double> x, y, z, w;
    std::vector<double> x0, y0, z0;      // reference positions (C14; solids only)
    // C17: an explicit type flag.  Identifying solids by comparing their weight
    // (w == 0.5) silently broke as soon as the weight became configurable.
    std::vector<char>   solid;
    std::vector<int> ordX, ordY, ordZ;   // deck sorted along each axis
};

struct AxisSplit
{
    std::vector<double> cut;       // [parts-1] lowest coordinate of groups 1..parts-1
    std::vector<double> lo, hi;    // [parts]   lowest / highest coordinate of each group
};

// Cut `order` (sorted along the axis) into `parts` weight-balanced groups; only
// particles accepted by `accept` (the parent cell) take part.  `acc` is scratch
// reused across calls.  Empty groups get an inverted box so they attract no
// ghosts.
template <class Pred>
static void split_axis(const std::vector<double>& c, const std::vector<double>& w,
                       const std::vector<int>& order, int parts, Pred accept,
                       std::vector<int>& acc, AxisSplit& S)
{
    acc.clear();
    double total = 0.0;
    for (int g : order)
        if (accept(g)) { acc.push_back(g); total += w[g]; }
    const int n = (int)acc.size();
    S.cut.assign(parts > 1 ? (size_t)(parts - 1) : 0, 0.0);
    S.lo.assign(parts, 1e300);
    S.hi.assign(parts, -1e300);
    if (n == 0) return;

    std::vector<int> cut((size_t)parts + 1, 0);
    cut[parts] = n;
    double cum = 0.0;
    int oi = 0;
    for (int p = 1; p < parts; ++p)
    {
        const double want = total * (double)p / (double)parts;
        while (oi < n && cum < want) { cum += w[acc[oi]]; ++oi; }
        // never split a group of equal coordinates (keeps ownership rectangular)
        while (oi < n && oi > 0 && c[acc[oi]] == c[acc[oi - 1]]) { cum += w[acc[oi]]; ++oi; }
        while (oi < p) { cum += w[acc[oi]]; ++oi; }
        while (oi > n - (parts - p)) { --oi; cum -= w[acc[oi]]; }
        if (oi <= cut[p - 1]) oi = cut[p - 1] + 1;
        cut[p] = oi;
    }
    for (int p = 0; p < parts; ++p)
    {
        S.lo[p] = c[acc[cut[p]]];
        S.hi[p] = c[acc[cut[p + 1] - 1]];
        if (p > 0) S.cut[p - 1] = S.lo[p];
    }
}

// L2/L3: the compute weight of a particle.  The review measured, on one GPU, that
// a solid particle costs 0.41 of a fluid one with the fat list and 0.97 with the
// lean one, and that a ghost costs about 0.02 (its cell-list slot) plus its share
// of the exchange.  Both are knobs: the defaults sit between the two list modes,
// and SPH_D1_W_SOLID / SPH_D1_W_GHOST override them.
static double g_w_solid = 0.45;
static double g_w_ghost = 0.06;

struct Part
{
    int nx = 1, ny = 1, nz = 1;
    std::vector<int> owner;      // [num]
    std::vector<double> box;     // [6*K] xlo,xhi,ylo,yhi,zlo,zhi
    std::vector<double> ownw, ghw;   // [K]
    std::vector<int> ghost;      // [K] ghost count
    std::vector<int> owncnt;     // [K] owned particle count
    long long ghosts = 0;
    double maxload = 0.0;
    bool empty = false;
};

static Part make_part(const Deck& D, int nx, int ny, int nz, double skin,
                      std::vector<int>& ixOf, std::vector<int>& iyOf,
                      std::vector<int>& izOf, std::vector<int>& acc)
{
    const int num = D.num, K = nx * ny * nz;
    const double band = 3.0 * SPH_HSML + skin;
    Part P;
    P.nx = nx; P.ny = ny; P.nz = nz;
    P.owner.assign(num, -1);
    P.box.assign((size_t)6 * K, 0.0);

    AxisSplit sx;
    split_axis(D.x, D.w, D.ordX, nx, [](int) { return true; }, acc, sx);
    for (int g = 0; g < num; ++g)
        ixOf[g] = (int)(std::upper_bound(sx.cut.begin(), sx.cut.end(), D.x[g]) - sx.cut.begin());

    std::vector<AxisSplit> sy(nx);
    for (int ix = 0; ix < nx; ++ix)
        split_axis(D.y, D.w, D.ordY, ny, [&](int g) { return ixOf[g] == ix; }, acc, sy[ix]);
    for (int g = 0; g < num; ++g)
        iyOf[g] = (int)(std::upper_bound(sy[ixOf[g]].cut.begin(), sy[ixOf[g]].cut.end(), D.y[g])
                        - sy[ixOf[g]].cut.begin());

    std::vector<AxisSplit> sz((size_t)nx * ny);
    for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy)
            split_axis(D.z, D.w, D.ordZ, nz,
                       [&](int g) { return ixOf[g] == ix && iyOf[g] == iy; },
                       acc, sz[(size_t)ix * ny + iy]);
    for (int g = 0; g < num; ++g)
    {
        const int cell = ixOf[g] * ny + iyOf[g];
        izOf[g] = (int)(std::upper_bound(sz[cell].cut.begin(), sz[cell].cut.end(), D.z[g])
                        - sz[cell].cut.begin());
        P.owner[g] = cell * nz + izOf[g];
    }

    for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy)
            for (int iz = 0; iz < nz; ++iz)
            {
                const int r = (ix * ny + iy) * nz + iz;
                P.box[6 * r + 0] = sx.lo[ix];
                P.box[6 * r + 1] = sx.hi[ix];
                P.box[6 * r + 2] = sy[ix].lo[iy];
                P.box[6 * r + 3] = sy[ix].hi[iy];
                P.box[6 * r + 4] = sz[(size_t)ix * ny + iy].lo[iz];
                P.box[6 * r + 5] = sz[(size_t)ix * ny + iy].hi[iz];
            }

    P.ownw.assign(K, 0.0); P.ghw.assign(K, 0.0);
    P.ghost.assign(K, 0); P.owncnt.assign(K, 0);
    for (int g = 0; g < num; ++g) { P.ownw[P.owner[g]] += D.w[g]; ++P.owncnt[P.owner[g]]; }

    // C14: reference box of each rank's owned solids, so the ghost estimate
    // counts the A3 partners the selection below will add (see build_ranks)
    std::vector<double> rbox((size_t)6 * K);
    const bool have_ref = !D.x0.empty();
    for (int r = 0; r < K; ++r)
    {
        rbox[6 * r + 0] = rbox[6 * r + 2] = rbox[6 * r + 4] = 1e300;
        rbox[6 * r + 1] = rbox[6 * r + 3] = rbox[6 * r + 5] = -1e300;
    }
    if (have_ref)
        for (int g = 0; g < num; ++g)
        {
            if (!D.solid[g]) continue;                 // C17: solids only
            const int r = P.owner[g];
            const double v[3] = { D.x0[g], D.y0[g], D.z0[g] };
            for (int c = 0; c < 3; ++c)
            {
                if (v[c] < rbox[6 * r + 2 * c])     rbox[6 * r + 2 * c] = v[c];
                if (v[c] > rbox[6 * r + 2 * c + 1]) rbox[6 * r + 2 * c + 1] = v[c];
            }
        }

    for (int r = 0; r < K; ++r)
    {
        const double x0 = P.box[6 * r + 0] - band, x1 = P.box[6 * r + 1] + band;
        const double y0 = P.box[6 * r + 2] - band, y1 = P.box[6 * r + 3] + band;
        const double z0 = P.box[6 * r + 4] - band, z1 = P.box[6 * r + 5] + band;
        const bool has_solid = have_ref && rbox[6 * r + 1] >= rbox[6 * r + 0];
        const double rx0 = rbox[6 * r + 0] - band, rx1 = rbox[6 * r + 1] + band;
        const double ry0 = rbox[6 * r + 2] - band, ry1 = rbox[6 * r + 3] + band;
        const double rz0 = rbox[6 * r + 4] - band, rz1 = rbox[6 * r + 5] + band;
        long long ng = 0;
        double gw = 0.0;
        for (int g = 0; g < num; ++g)
        {
            if (P.owner[g] == r) continue;
            bool in = D.x[g] >= x0 && D.x[g] <= x1 && D.y[g] >= y0 && D.y[g] <= y1 &&
                      D.z[g] >= z0 && D.z[g] <= z1;
            if (!in && has_solid && D.solid[g])
                in = D.x0[g] >= rx0 && D.x0[g] <= rx1 && D.y0[g] >= ry0 &&
                     D.y0[g] <= ry1 && D.z0[g] >= rz0 && D.z0[g] <= rz1;
            if (in) { ++ng; gw += D.w[g]; }
        }
        P.ghost[r] = (int)ng;
        P.ghw[r] = gw;
        P.ghosts += ng;
        // L3: a ghost is not a particle that computes -- it costs a cell-list slot
        // and its share of the exchange, not a full owned particle
        const double load = P.ownw[r] + g_w_ghost * gw;
        if (load > P.maxload) P.maxload = load;
        if (P.owncnt[r] == 0) P.empty = true;
    }
    return P;
}

static bool build_ranks(Job& J, const std::vector<double>& rows,
                        const std::vector<int>& ownerRank,
                        const std::vector<double>& box,
                        const double* pos = nullptr);

static bool build_job(Job& J, const std::vector<double>& rows)
{
    const int nf = 14;
    const int num = J.num, K = J.K;
    if (K < 1 || K > num) { fprintf(stderr, "[D1] bad SPH_RANKS=%d\n", K); return false; }

    Deck D;
    D.num = num;
    D.x.resize(num); D.y.resize(num); D.z.resize(num); D.w.resize(num);
    D.x0.resize(num); D.y0.resize(num); D.z0.resize(num); D.solid.resize(num);
    double total = 0.0;
    for (int i = 0; i < num; ++i)
    {
        const double* r = &rows[(size_t)i * nf];
        D.x[i] = r[0]; D.y[i] = r[1]; D.z[i] = r[2];
        const double t = r[9];
        D.w[i] = (t > 0.0) ? 1.0 : ((t < -10.0) ? g_w_solid : 0.0);
        D.solid[i] = (t < -10.0) ? 1 : 0;
        D.x0[i] = r[10]; D.y0[i] = r[11]; D.z0[i] = r[12];
        total += D.w[i];
    }
    if (!(total > 0.0)) { fprintf(stderr, "[D1] the deck has no weighted particle\n"); return false; }

    // one sorted order per axis (ties by deck index, so the cuts are reproducible)
    auto sort_axis = [&](const std::vector<double>& c)
    {
        std::vector<int> o(num);
        for (int i = 0; i < num; ++i) o[i] = i;
        std::sort(o.begin(), o.end(), [&](int a, int b)
        {
            return (c[a] < c[b]) || (c[a] == c[b] && a < b);
        });
        return o;
    };
    D.ordX = sort_axis(D.x);
    D.ordY = sort_axis(D.y);
    D.ordZ = sort_axis(D.z);

    // candidates: every ordered factorisation nx*ny*nz == K (SPH_D1_SPLIT=nx,ny,nz
    // forces one of them; that is only there to A/B the decomposition itself)
    struct Cand { int nx, ny, nz; };
    std::vector<Cand> cand;
    if (const char* s = getenv("SPH_D1_SPLIT"))
    {
        int a = 1, b = 1, d = 1;
        if (sscanf(s, "%d,%d,%d", &a, &b, &d) == 3 && a > 0 && b > 0 && d > 0 && a * b * d == K)
            cand.push_back({ a, b, d });
    }
    if (cand.empty())
        for (int nx = 1; nx <= K; ++nx)
            for (int ny = 1; ny <= K / nx; ++ny)
                if (K % (nx * ny) == 0) cand.push_back({ nx, ny, K / (nx * ny) });

    std::vector<int> ixOf(num), iyOf(num), izOf(num), acc;
    Part best;
    bool have = false;
    for (const Cand& c : cand)
    {
        Part P = make_part(D, c.nx, c.ny, c.nz, J.skin, ixOf, iyOf, izOf, acc);
        const double dup = (double)P.ghosts / (double)num;
        std::printf("[D1] split (%d,%d,%d): max load %.0f of %.0f weight, "
                    "ghosts %lld (+%.1f%% of the deck)%s\n",
                    c.nx, c.ny, c.nz, P.maxload, total, P.ghosts, 100.0 * dup,
                    P.empty ? "  [empty rank, rejected]" : "");
        if (P.empty) continue;
        if (!have || P.maxload < best.maxload ||
            (P.maxload == best.maxload && P.ghosts < best.ghosts))
        {
            best = std::move(P);
            have = true;
        }
    }
    if (!have)
    {
        fprintf(stderr, "[D1] no block decomposition of K=%d keeps every rank non-empty\n", K);
        return false;
    }
    J.gnx = best.nx; J.gny = best.ny; J.gnz = best.nz;
    // P16: keep the ownership map.  A halo refresh reuses it (only the boxes are
    // recomputed from the live positions), so a rebuild no longer has to re-cut the
    // domain and re-sort the whole deck.
    J.owner = best.owner;
    J.box   = best.box;
    return build_ranks(J, rows, best.owner, best.box);
}

// ---------------------------------------------------------------------------
// Everything a rank needs from an ownership map: the per-rank deck rows
// (own, then ghosts, both in deck order), gid, the peer links and the exchange
// block offsets.  Rank 0 runs it at start-up and at every repartition while the
// other rank threads wait on the barrier.
// ---------------------------------------------------------------------------
static bool build_ranks(Job& J, const std::vector<double>& rows,
                        const std::vector<int>& ownerRank,
                        const std::vector<double>& box,
                        const double* pos)
{
    const int nf = 14;
    const int num = J.num, K = J.K;
    // C9: the ghost band is 3h plus the skin; the displacement trigger rebuilds
    // before any particle can spend more than half of the skin, so a particle
    // that was not selected is still more than 3h away when it is needed.
    const double band = 3.0 * SPH_HSML + J.skin;

    // ownership, then the owned lists in *deck* order (so every rank's own
    // block comes out fluid|solid|wall and the local index is the deck order)
    std::vector<int> ownerLocal(num, -1);
    std::vector<std::vector<int>> ownGids(K), ghostGids(K);
    for (int g = 0; g < num; ++g)
    {
        const int r = ownerRank[g];
        ownerLocal[g] = (int)ownGids[r].size();
        ownGids[r].push_back(g);
    }

    // C14: the solid-solid pairs come from the *reference* configuration (A3),
    // so a partner can be inside the reference shell of a rank's solids while
    // being outside the shell of its current box.  The reference box of a
    // rank's owned solids is static, and under small strain it is close to the
    // current box, so testing against it adds only a few particles.
    std::vector<double> rbox((size_t)6 * K);
    for (int r = 0; r < K; ++r)
    {
        rbox[6 * r + 0] = rbox[6 * r + 2] = rbox[6 * r + 4] = 1e300;
        rbox[6 * r + 1] = rbox[6 * r + 3] = rbox[6 * r + 5] = -1e300;
    }
    for (int g = 0; g < num; ++g)
    {
        const double* rw = &rows[(size_t)g * nf];
        if (rw[9] >= -10.0) continue;                  // not a solid
        const int r = ownerRank[g];
        for (int c = 0; c < 3; ++c)
        {
            const double v = rw[10 + c];
            if (v < rbox[6 * r + 2 * c])     rbox[6 * r + 2 * c] = v;
            if (v > rbox[6 * r + 2 * c + 1]) rbox[6 * r + 2 * c + 1] = v;
        }
    }

    // P11: the ghost solids come in two layers.  Layer 1 is what the current (and
    // reference) box selects; layer 2 is the A3 partners of layer 1 -- the solids
    // within 3h of a layer-1 solid in the *reference* configuration -- which the
    // layer-1 ghosts need to run their own constitution.  Layer 2 is only read.
    std::vector<std::vector<int>> ghostF(K), ghostS1(K), ghostS2(K), ghostW(K);
    for (int r = 0; r < K; ++r)
    {
        const double x0 = box[6 * r + 0] - band, x1 = box[6 * r + 1] + band;
        const double y0 = box[6 * r + 2] - band, y1 = box[6 * r + 3] + band;
        const double z0 = box[6 * r + 4] - band, z1 = box[6 * r + 5] + band;
        const bool has_solid = rbox[6 * r + 1] >= rbox[6 * r + 0];
        const double rx0 = rbox[6 * r + 0] - band, rx1 = rbox[6 * r + 1] + band;
        const double ry0 = rbox[6 * r + 2] - band, ry1 = rbox[6 * r + 3] + band;
        const double rz0 = rbox[6 * r + 4] - band, rz1 = rbox[6 * r + 5] + band;
        auto layer1 = [&](int g, const double* rw) -> bool
        {
            // C10/P16: the box test uses the live positions when the caller has them
            const double* px = pos ? (pos + 3 * (size_t)g) : rw;
            if (px[0] >= x0 && px[0] <= x1 && px[1] >= y0 && px[1] <= y1 &&
                px[2] >= z0 && px[2] <= z1) return true;
            if (!has_solid || rw[9] >= -10.0) return false;
            return rw[10] >= rx0 && rw[10] <= rx1 && rw[11] >= ry0 &&
                   rw[11] <= ry1 && rw[12] >= rz0 && rw[12] <= rz1;
        };
        for (int g = 0; g < num; ++g)
        {
            if (ownerRank[g] == r) continue;
            const double* rw = &rows[(size_t)g * nf];
            if (!layer1(g, rw)) continue;
            const double t = rw[9];
            if (t > 0.0)        ghostF[r].push_back(g);
            else if (t < -10.0) ghostS1[r].push_back(g);
            else                ghostW[r].push_back(g);
        }

        // reference AABB of layer 1, grown by the A3 support, then layer 2
        if (!ghostS1[r].empty())
        {
            double rb[6] = { 1e300, -1e300, 1e300, -1e300, 1e300, -1e300 };
            for (int g : ghostS1[r])
                for (int c = 0; c < 3; ++c)
                {
                    const double v = rows[(size_t)g * nf + 10 + c];
                    if (v < rb[2 * c])     rb[2 * c] = v;
                    if (v > rb[2 * c + 1]) rb[2 * c + 1] = v;
                }
            const double grow = 3.0 * SPH_HSML;      // A3 pairs are |dX0| < 3h
            for (int c = 0; c < 3; ++c) { rb[2 * c] -= grow; rb[2 * c + 1] += grow; }
            for (int g = 0; g < num; ++g)
            {
                if (ownerRank[g] == r) continue;
                const double* rw = &rows[(size_t)g * nf];
                if (rw[9] >= -10.0) continue;                   // solids only
                if (layer1(g, rw)) continue;                    // already layer 1
                if (rw[10] >= rb[0] && rw[10] <= rb[1] &&
                    rw[11] >= rb[2] && rw[11] <= rb[3] &&
                    rw[12] >= rb[4] && rw[12] <= rb[5]) ghostS2[r].push_back(g);
            }
        }
    }

    // per-rank decks: own (deck order) then ghosts, ordered fluid | solid-layer1 |
    // solid-layer2 | wall so every `p < s.d.nall`-style bound keeps its meaning
    J.ranks.resize(K);
    for (int r = 0; r < K; ++r)
    {
        Rank& R = J.ranks[r];
        R.id = r;
        R.links.clear();               // a repartition rebuilds them from scratch
        R.ownNum = (int)ownGids[r].size();
        R.ngs1   = (int)ghostS1[r].size();
        // one ghost list in *slot* order: the rows, the links and the exchange lists
        // all have to agree on it
        ghostGids[r].clear();
        ghostGids[r].insert(ghostGids[r].end(), ghostF[r].begin(), ghostF[r].end());
        ghostGids[r].insert(ghostGids[r].end(), ghostS1[r].begin(), ghostS1[r].end());
        ghostGids[r].insert(ghostGids[r].end(), ghostS2[r].begin(), ghostS2[r].end());
        ghostGids[r].insert(ghostGids[r].end(), ghostW[r].begin(), ghostW[r].end());
        R.numP   = R.ownNum + (int)ghostGids[r].size();
        R.rows.resize((size_t)R.numP * nf);
        R.gid.resize(R.numP);
        for (int i = 0; i < R.ownNum; ++i)
        {
            const int g = ownGids[r][i];
            memcpy(&R.rows[(size_t)i * nf], &rows[(size_t)g * nf], nf * sizeof(double));
            R.gid[i] = g;
        }
        for (int i = 0; i < (int)ghostGids[r].size(); ++i)
        {
            const int g = ghostGids[r][i];
            memcpy(&R.rows[(size_t)(R.ownNum + i) * nf], &rows[(size_t)g * nf], nf * sizeof(double));
            R.gid[R.ownNum + i] = g;
        }
        R.label = J.deck + " [rank " + std::to_string(r) + "]";
        R.ngs2 = (int)(ghostGids[r].size() - ghostF[r].size() - ghostW[r].size() - R.ngs1);
    }

    // links.  One scan of each rank's ghost block, in slot order, classifies
    // each ghost; the sender's list is filled in exactly the same order.  With
    // blocks a rank can have up to 26 face/edge/corner peers.
    auto link_index = [&](int a, int b) -> int
    {
        std::vector<DLink>& v = J.ranks[a].links;
        for (int i = 0; i < (int)v.size(); ++i) if (v[i].peer == b) return i;
        v.push_back(DLink());
        v.back().peer = b;
        return (int)v.size() - 1;
    };
    for (int r = 0; r < K; ++r)
    {
        Rank& R = J.ranks[r];
        for (int si = 0; si < (int)ghostGids[r].size(); ++si)
        {
            const int g = ghostGids[r][si];
            const int s = ownerRank[g];
            const int gslot = R.ownNum + si;
            const int li_s = link_index(s, r);
            const int li_r = link_index(r, s);
            DLink& Ls = J.ranks[s].links[li_s];
            DLink& Lr = J.ranks[r].links[li_r];
            const int loc = ownerLocal[g];
            const double t = rows[(size_t)g * nf + 9];
            if (t > 0.0)        { Ls.sendF.push_back(loc); Lr.recvF.push_back(gslot); }
            else if (t < -10.0) { Ls.sendS.push_back(loc); Lr.recvS.push_back(gslot); }
            else                { Ls.sendW.push_back(loc); Lr.recvW.push_back(gslot); }
        }
    }
    // Block layout.  Four exchange blocks per rank, each holding every peer's
    // records side by side in link order, so a whole exchange point is ONE
    // host<->device copy:
    //   E1  [fluid 8 | solid 7] per link (off[0] = link base, off[1] = solids)
    //   E2  [solid 13] per link, static [12] per link
    // E3 (fluid PIJW) is gone: it shipped zeros (P8), so o[3] stays 0 and no
    // buffer is reserved for it.
    for (int r = 0; r < K; ++r)
    {
        Rank& R = J.ranks[r];
        int o[5] = { 0, 0, 0, 0, 0 };
        for (DLink& L : R.links)
        {
            const int nf = (int)L.sendF.size(), ns = (int)L.sendS.size();
            const int nw = (int)L.sendW.size();
            L.off[0] = o[0];                                  // E1 block base
            L.off[1] = o[0] + SPH_XCH_REC_F * nf;             // its first solid
            L.off[2] = 0;                                     // E2 removed (P11)
            L.off[3] = 0;
            L.off[4] = o[4];
            o[0] += SPH_XCH_REC_F * nf + SPH_XCH_REC_S * ns;
            o[4] += SPH_XCH_REC_ST * (ns + nw);
        }
        R.nSendE1 = (size_t)o[0];         // P17: E1 block length (doubles)
        R.sendCap = 0;
        for (int k = 0; k < 5; ++k)
            if (k != 1) R.sendCap = std::max(R.sendCap, (size_t)o[k]);

        int q[5] = { 0, 0, 0, 0, 0 };
        for (DLink& L : R.links)
        {
            const int nf = (int)L.recvF.size(), ns = (int)L.recvS.size();
            const int nw = (int)L.recvW.size();
            L.recvOff[0] = q[0];
            L.recvOff[1] = q[0] + SPH_XCH_REC_F * nf;
            L.recvOff[2] = 0;
            L.recvOff[3] = 0;
            L.recvOff[4] = q[4];
            q[0] += SPH_XCH_REC_F * nf + SPH_XCH_REC_S * ns;
            q[4] += SPH_XCH_REC_ST * (ns + nw);
        }
        R.recvCap = 0;
        for (int k = 0; k < 5; ++k)
            if (k != 1) R.recvCap = std::max(R.recvCap, (size_t)q[k]);
    }
    for (int r = 0; r < K; ++r)
        for (DLink& L : J.ranks[r].links)
        {
            for (const DLink& M : J.ranks[L.peer].links)
                if (M.peer == r) { for (int k = 0; k < 5; ++k) L.peerOff[k] = M.off[k]; }
            L.peerE1 = J.ranks[L.peer].nSendE1;      // P17: where its tail value sits
        }

    // every ghost slot must be fed by exactly one link (a silent wrong answer
    // otherwise)
    for (int r = 0; r < K; ++r)
    {
        size_t got = 0;
        for (const DLink& L : J.ranks[r].links)
            got += L.recvF.size() + L.recvS.size() + L.recvW.size();
        const size_t want = (size_t)(J.ranks[r].numP - J.ranks[r].ownNum);
        if (got != want)
        {
            fprintf(stderr, "[D1] rank %d: %zu ghost slots but the links carry %zu\n",
                    r, want, got);
            return false;
        }
    }
    J.ghost_total = 0;
    for (int r = 0; r < K; ++r) J.ghost_total += J.ranks[r].numP - J.ranks[r].ownNum;
    return true;
}

// ---------------------------------------------------------------------------
// exchange helpers (called from one rank thread)
// ---------------------------------------------------------------------------
// One exchange point = one gather kernel per record type -> ONE device-to-host
// copy of the whole block -> barrier -> a plain CPU relay that copies every
// peer's block into this rank's receive block -> ONE host-to-device copy ->
// one scatter kernel per record type.
//
// P10: the send block is double buffered and consecutive exchange points use
// different halves, so the trailing barrier is gone -- it existed only to stop a
// rank from overwriting its send block while a peer was still relaying from it.
// A rank can write a half again only at the exchange point *two* steps later, and
// to get there it must pass the next exchange's barrier, which no rank reaches
// before finishing its own relay.  Six barriers per step became two (one per
// exchange point), and one once E2 is gone.
//
// The relay is the point: with 3-D blocks a rank has up to 26 peers, and one
// cudaMemcpyAsync per (peer, direction) on Windows/WDDM costs ~150 us, which
// dominated the step (measured 3.9 ms of E1 for 2.2 MB of useful data).  Four
// copies and four launches per step fix that; the CPU memcpy moves the same
// bytes at host-memory speed.
static double* send_buf(Rank& R) { return R.hSendB[R.sendSlot]; }

static void xch_pack(Job& J, Rank& R, int mode)
{
    if (J.K == 1) return;              // no peers: nothing to exchange, no barrier
    const double t0 = now_s();
    Solver<double>& S = R.S;
    (void)mode;                        // one exchange point left (P11)
    size_t bytes = ((size_t)SPH_XCH_REC_F * R.nSend[0] +
                    (size_t)SPH_XCH_REC_S * R.nSend[1]) * sizeof(double);
    R.nSendE1 = bytes / sizeof(double);
    launch_xch_gather(S.g, 0, R.dSendIdx[0], R.dSendOff[0], R.nSend[0], R.dStage, 0);
    launch_xch_gather(S.g, 1, R.dSendIdx[1], R.dSendOff[1], R.nSend[1], R.dStage, 0);
    // P17: the halo trigger's input rides in the tail of this block, so the copy
    // below carries it for free and the relay hands the peers' values over.
    if (J.skin > 0.0 && R.dXBuild && R.dMaxDisp)
    {
        CUDA_CHECK(cudaMemsetAsync(R.dMaxDisp, 0, sizeof(unsigned long long), 0));
        launch_disp_max(S.g, R.dXBuild, R.dMaxDisp, 0);
        launch_disp_store(R.dMaxDisp, R.dStage + R.nSendE1, 0);
        bytes += sizeof(double);
    }
    const double tg = now_s();
    R.x_g += tg - t0;
    if (bytes)
        CUDA_CHECK(cudaMemcpyAsync(send_buf(R), R.dStage, bytes, cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    const double t_sync = now_s();
    R.x_c += t_sync - tg;
    J.bar.wait();
    const double t_bar = now_s();
    R.x_bar   += t_bar - t_sync;
    R.xch_s   += t_bar - t0;
}

static void xch_unpack(Job& J, Rank& R, int mode)
{
    if (J.K == 1) return;
    const double t0 = now_s();
    Solver<double>& S = R.S;

    // 1) relay: every peer's block for me -> my receive block (CPU memcpy)
    (void)mode;                        // one exchange point left (P11)
    for (const DLink& L : R.links)
    {
        const size_t n = (size_t)SPH_XCH_REC_F * L.recvF.size() +
                         (size_t)SPH_XCH_REC_S * L.recvS.size();
        if (!n) continue;
        memcpy(R.hRecv + L.recvOff[0],
               J.ranks[L.peer].hSendB[R.sendSlot] + L.peerOff[0], n * sizeof(double));
    }

    const double tr = now_s();
    R.x_r += tr - t0;
    // 2) one host-to-device copy of the whole block, then one scatter kernel
    //    per record type
    const size_t bytes = ((size_t)SPH_XCH_REC_F * R.nRecv[0] +
                          (size_t)SPH_XCH_REC_S * R.nRecv[1]) * sizeof(double);
    if (bytes)
        CUDA_CHECK(cudaMemcpyAsync(R.dStage, R.hRecv, bytes, cudaMemcpyHostToDevice, 0));
    launch_xch_scatter(S.g, 0, R.dRecvIdx[0], R.dRecvOff[0], R.nRecv[0], R.dStage, 0);
    launch_xch_scatter(S.g, 1, R.dRecvIdx[1], R.dRecvOff[1], R.nRecv[1], R.dStage, 0);
    // TASK 2 / P9: p and irho travel for free -- rebuild them for the ghost
    // block with the same expressions k_prepare uses on the owner (rho arrived
    // exact, so the bits match).
    launch_xch_fixup(S.g, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));
    const double t_sync = now_s();
    R.x_h += t_sync - tr;
    R.xch_s   += t_sync - t0;
}

// P10: advance the buffer parity once per exchange point, on every rank (they
// all follow the same sequence of points, so the halves line up).
static void xch(Job& J, Rank& R, int mode)
{
    R.sendSlot = R.xchSeq & 1;
    ++R.xchSeq;
    xch_pack(J, R, mode);
    xch_unpack(J, R, mode);
}

// C9 helpers (defined with the repartition code below)
static void monitor_init(Rank& R);
static void disp_collect(Job& J, Rank& R);

// one-off: the static fields of the ghost block.  A and vecn only a rank's
// owner can produce (they depend on that rank's own neighbour set), and
// k_prepare never runs over ghosts.
static void xch_static(Job& J, Rank& R)
{
    if (J.K == 1) return;
    const double t0 = now_s();
    Solver<double>& S = R.S;
    const size_t sbytes = (size_t)SPH_XCH_REC_ST * R.nSend[4] * sizeof(double);
    const size_t rbytes = (size_t)SPH_XCH_REC_ST * R.nRecv[4] * sizeof(double);
    launch_xch_gather(S.g, 4, R.dSendIdx[4], R.dSendOff[4], R.nSend[4], R.dStage, 0);
    if (sbytes)
        CUDA_CHECK(cudaMemcpyAsync(R.hSendB[0], R.dStage, sbytes, cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    R.xch_s += now_s() - t0;
    J.bar.wait();

    const double t1 = now_s();
    for (const DLink& L : R.links)
    {
        const size_t n = (size_t)SPH_XCH_REC_ST * (L.recvS.size() + L.recvW.size());
        if (!n) continue;
        memcpy(R.hRecv + L.recvOff[4],
               J.ranks[L.peer].hSendB[0] + L.peerOff[4], n * sizeof(double));
    }
    if (rbytes)
        CUDA_CHECK(cudaMemcpyAsync(R.dStage, R.hRecv, rbytes, cudaMemcpyHostToDevice, 0));
    launch_xch_scatter(S.g, 4, R.dRecvIdx[4], R.dRecvOff[4], R.nRecv[4], R.dStage, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));
    R.xch_s += now_s() - t1;
    J.bar.wait();
}
// ---------------------------------------------------------------------------
// one step, in the order of the single-GPU loop but with the exchanges in it
// ---------------------------------------------------------------------------
static void step_d1(Job& J, Rank& R, int itime)
{
    Solver<double>& S = R.S;
    const bool prof = J.prof;
    auto lap = [&](int k)
    {
        if (!prof) return;
        CUDA_CHECK(cudaStreamSynchronize(0));
        const double t = now_s();
        R.ph[k] += t - R.ph_t;
        R.ph_t = t;
    };
    if (prof) R.ph_t = now_s();

    lap(0);

    launch_prepare(S.g, 0);
    lap(1);

    xch(J, R, 0);                    // E1: x vx rho (+SurFD for fluids)
    lap(2);
    if (J.skin > 0.0)
    {
        // P17: exact every step, from the E1 block itself.  The trigger sits one
        // MaxV*dt below s/2 because the step that detects the crossing runs on the
        // halo it is about to replace (C18).
        disp_collect(J, R);
        const double thresh = 0.5 * J.skin - 2.0 * SPH_MAXV * SPH_DT;
        R.rebuild_due = (R.disp > thresh);
        R.last_disp = R.disp;
    }
    launch_copy_xf(S.g, 0);          // ghosts moved: refresh the float copy

    CUDA_CHECK(cudaMemsetAsync(S.g.cellCur, 0, (size_t)(S.g.d.ncell + 1) * sizeof(int), 0));
    CUDA_CHECK(cudaMemsetAsync(S.g.cellSolid, 0, S.g.d.ncell * sizeof(int), 0));
    CUDA_CHECK(cudaMemsetAsync(S.g.cellFluid, 0, S.g.d.ncell * sizeof(int), 0));
    lap(3);
    launch_cell_hist(S.g, S.g.cellCur, S.g.cellSolid, S.g.cellFluid, 0);
    lap(4);
    launch_cell_scan(S.g.cellStart, S.g.cellCur, S.g.d.ncell, S.d_scan_tmp, S.d_scan_bytes, 0);
    lap(5);
    launch_cell_scatter(S.g, S.g.cellCur, 0);
    lap(6);
    launch_cell_sort(S.g.cellStart, S.g.sorted, S.g.d.ncell, S.g.gid, 0);
    lap(7);

    launch_pass1_fluid(S.g, 0);
    lap(8);
    // P11: pass1_solid and the constitution run over the owned solids *and* the
    // layer-1 ghost solids, so every ghost solid holds its own Pia by the time
    // pass2_solid reads it -- that is what removed E2.
    launch_pass1_solid(S.g, 0);
    lap(9);
    launch_constitution(S.g, 0);
    lap(10);

    launch_pass2_fluid(S.g, 0);
    lap(11);

    launch_pass2_solid(S.g, 0);
    lap(12);

    launch_update_fluid(S.g, 0);
    launch_update_solid(S.g, 0);
    launch_update_pos(S.g, 0);
    lap(13);

    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// results: every rank pushes its owned particles into global gid-indexed
// arrays, rank 0 writes the file in the exact format of write_result()
// ---------------------------------------------------------------------------
static void fetch_own(Job& J, Rank& R, Job::OutBuf& O)
{
    const SphDims& d = R.S.g.d;
    const int n = d.num, ns = d.solnum;
    const int N = d.numP;
    std::vector<double> hx((size_t)3 * n), hp(n);
    CUDA_CHECK(cudaMemcpy(hx.data(),         R.S.g.x, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hx.data() + n,     R.S.g.x + N, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hx.data() + 2 * n, R.S.g.x + 2 * N, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hp.data(), R.S.g.p, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> pia;
    if (ns > 0)
    {
        pia.resize((size_t)SPH_PIA_STRIDE * ns);
        CUDA_CHECK(cudaMemcpy(pia.data(), R.S.g.Pia, pia.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < n; ++i)
    {
        const int g = R.gid[i];
        O.gx[g] = hx[i];
        O.gy[g] = hx[n + i];
        O.gz[g] = hx[2 * n + i];
        if (i < d.ntotal)      O.gp[g] = hp[i];
        else if (i < d.nall)
        {
            const size_t k = (size_t)(i - d.ntotal) * SPH_PIA_STRIDE;
            O.gp[g] = pia[k] + pia[k + 4] + pia[k + 8];
        }
        else                   O.gp[g] = 0.0;
    }
}

static void write_global_result(Job& J, const Job::OutBuf& O, int idx)
{
    const bool f32 = (getenv("SPH_OUT_F32") != nullptr);
    char name[128];
    snprintf(name, sizeof(name), "result%d.sphbin", idx);
    FILE* f = fopen(name, "wb");
    if (!f) { fprintf(stderr, "[D1] cannot write %s\n", name); return; }

    unsigned char h[128] = {0};
    const char magic[8] = { 'S','P','H','B','I','N','\0','\0' };
    memcpy(h, magic, 8);
    const uint32_t ver = 1, hs = 128, num = (uint32_t)J.num,
                   ntotal = (uint32_t)J.ntotal, solnum = (uint32_t)J.solnum,
                   nfields = 4, step = (uint32_t)(idx * 500),
                   flags = 2u | (f32 ? 4u : 0u);
    memcpy(h + 8, &ver, 4);      memcpy(h + 12, &hs, 4);
    memcpy(h + 16, &num, 4);     memcpy(h + 20, &ntotal, 4);
    memcpy(h + 24, &solnum, 4);  memcpy(h + 28, &nfields, 4);
    memcpy(h + 32, &step, 4);    memcpy(h + 36, &flags, 4);
    const double hsml = SPH_HSML, dt = SPH_DT;
    memcpy(h + 40, &hsml, 8); memcpy(h + 48, &dt, 8);
    memcpy(h + 56, J.grid, 48);
    fwrite(h, 1, 128, f);

    const size_t rec = 4;
    if (f32)
    {
        std::vector<float> buf(rec * 65536);
        size_t used = 0;
        for (int i = 0; i < J.num; ++i)
        {
            if (used + rec > buf.size()) { fwrite(buf.data(), sizeof(float), used, f); used = 0; }
            buf[used++] = (float)O.gx[i]; buf[used++] = (float)O.gy[i];
            buf[used++] = (float)O.gz[i]; buf[used++] = (float)O.gp[i];
        }
        if (used) fwrite(buf.data(), sizeof(float), used, f);
    }
    else
    {
        std::vector<double> buf(rec * 65536);
        size_t used = 0;
        for (int i = 0; i < J.num; ++i)
        {
            if (used + rec > buf.size()) { fwrite(buf.data(), sizeof(double), used, f); used = 0; }
            buf[used++] = O.gx[i]; buf[used++] = O.gy[i];
            buf[used++] = O.gz[i]; buf[used++] = O.gp[i];
        }
        if (used) fwrite(buf.data(), sizeof(double), used, f);
    }
    fclose(f);
}

// ---------------------------------------------------------------------------
// Repartition (SPH_D1_REPART=N): recompute the blocks from the *current*
// positions, move the state the deck cannot reproduce, and rebuild everything
// that depends on ownership.  All rank threads run this together; the new
// layout itself is computed once, by rank 0, and published through Job.
//
// Migrated per particle (20 doubles): x[3], vx[3], rho, SurF, A[9], vecn[3].
// Everything else is either static or recomputed before its next use:
//   X0/mass/rho0/itype   from the deck rows,
//   A, vecn              carried here (they are the only *computed* statics --
//                        recomputing them would use the current rho and drift),
//   p, irho, SurFD       by k_prepare from rho and SurF,
//   F, DFdt, Pia         by pass1_solid / constitution,
//   SForce, PIJW         by pass2,
//   drhodt, viscos       by pass1.
// The own block is rebuilt from the deck (finish_deck), so every own slot gets
// its current dynamic state back from the gid-indexed migration buffer; ghost
// slots are refreshed by the next E1 and by the static exchange below.
// ---------------------------------------------------------------------------
enum { MIG_N = 20 };

static void build_send_lists(Job& J, Rank& R);   // defined below

static void xch_free(Rank& R)
{
    for (int s = 0; s < 2; ++s)
        if (R.hSendB[s]) { cudaFreeHost(R.hSendB[s]); R.hSendB[s] = nullptr; }
    if (R.hRecv)  { cudaFreeHost(R.hRecv); R.hRecv = nullptr; }
    if (R.dStage) { cudaFree(R.dStage);    R.dStage = nullptr; }
    // C11: build_send_lists() allocates 20 index/offset arrays per rank, and on
    // a repartition the old ones were simply overwritten -- about 0.65 MiB per
    // event on this deck, ~5 MB per event on the 1203k one.  Free them here and
    // let the next build_send_lists() allocate the new set.
    for (int g = 0; g < 5; ++g)
    {
        if (R.dSendIdx[g]) { cudaFree(R.dSendIdx[g]); R.dSendIdx[g] = nullptr; }
        if (R.dSendOff[g]) { cudaFree(R.dSendOff[g]); R.dSendOff[g] = nullptr; }
        if (R.dRecvIdx[g]) { cudaFree(R.dRecvIdx[g]); R.dRecvIdx[g] = nullptr; }
        if (R.dRecvOff[g]) { cudaFree(R.dRecvOff[g]); R.dRecvOff[g] = nullptr; }
        R.nSend[g] = 0;
        R.nRecv[g] = 0;
    }
    if (R.dXBuild)  { cudaFree(R.dXBuild);  R.dXBuild = nullptr; }
    if (R.dMaxDisp) { cudaFree(R.dMaxDisp); R.dMaxDisp = nullptr; }
}

// ---------------------------------------------------------------------------
// C9: snapshot of the positions the ghost set was chosen from, plus the device
// scalar the displacement reduce lands in.  Called after every (re)build.
// ---------------------------------------------------------------------------
static void monitor_init(Rank& R)
{
    if (R.dXBuild)  { cudaFree(R.dXBuild);  R.dXBuild = nullptr; }
    if (R.dMaxDisp) { cudaFree(R.dMaxDisp); R.dMaxDisp = nullptr; }
    const int n = R.S.g.d.num, N = R.S.g.d.numP;
    if (n <= 0) return;
    CUDA_CHECK(cudaMalloc((void**)&R.dXBuild, (size_t)3 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&R.dMaxDisp, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(R.dMaxDisp, 0, sizeof(unsigned long long)));
    for (int c = 0; c < 3; ++c)
        CUDA_CHECK(cudaMemcpy(R.dXBuild + (size_t)c * n, R.S.g.x + (size_t)c * N,
                              (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
}

// P17: the displacement is reduced into the tail of the E1 staging block (xch_pack
// does that), and after E1's barrier every rank reads the maximum over the peers'
// blocks -- the same relay it already performs, no readback of its own.
static void disp_collect(Job& J, Rank& R)
{
    if (!R.dXBuild || R.nSendE1 == 0) { R.disp = 0.0; return; }
    double mx = R.hSendB[R.sendSlot][R.nSendE1];
    for (const DLink& L : R.links)
    {
        const double v = J.ranks[L.peer].hSendB[R.sendSlot][L.peerE1];
        if (v > mx) mx = v;
    }
    R.disp = mx;
    if (getenv("SPH_D1_DBG") && R.id == 0)
        printf("[dbg] step: global max displacement %.6g mm (skin %.3g mm)\n",
               1000.0 * mx, 1000.0 * J.skin);
}

static void mig_snapshot(Job& J, Rank& R)
{
    const SphDims& d = R.S.g.d;
    const int n = d.num, N = d.numP;
    const double* src[MIG_N] = {
        R.S.g.x, R.S.g.x + N, R.S.g.x + 2 * N,
        R.S.g.vx, R.S.g.vx + N, R.S.g.vx + 2 * N,
        R.S.g.rho, R.S.g.SurF,
        R.S.g.A + 0 * N, R.S.g.A + 1 * N, R.S.g.A + 2 * N,
        R.S.g.A + 3 * N, R.S.g.A + 4 * N, R.S.g.A + 5 * N,
        R.S.g.A + 6 * N, R.S.g.A + 7 * N, R.S.g.A + 8 * N,
        R.S.g.vecn, R.S.g.vecn + N, R.S.g.vecn + 2 * N
    };
    std::vector<double> buf((size_t)n);
    for (int k = 0; k < MIG_N; ++k)
    {
        CUDA_CHECK(cudaMemcpy(buf.data(), src[k], (size_t)n * sizeof(double),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i)
            J.mig[(size_t)R.gid[i] * MIG_N + k] = buf[i];
    }
}

static void mig_restore(Job& J, Rank& R)
{
    const SphDims& d = R.S.g.d;
    const int n = d.num, N = d.numP;
    double* dst[MIG_N] = {
        R.S.g.x, R.S.g.x + N, R.S.g.x + 2 * N,
        R.S.g.vx, R.S.g.vx + N, R.S.g.vx + 2 * N,
        R.S.g.rho, R.S.g.SurF,
        R.S.g.A + 0 * N, R.S.g.A + 1 * N, R.S.g.A + 2 * N,
        R.S.g.A + 3 * N, R.S.g.A + 4 * N, R.S.g.A + 5 * N,
        R.S.g.A + 6 * N, R.S.g.A + 7 * N, R.S.g.A + 8 * N,
        R.S.g.vecn, R.S.g.vecn + N, R.S.g.vecn + 2 * N
    };
    std::vector<double> buf((size_t)n);
    for (int k = 0; k < MIG_N; ++k)
    {
        for (int i = 0; i < n; ++i)
            buf[i] = J.mig[(size_t)R.gid[i] * MIG_N + k];
        CUDA_CHECK(cudaMemcpy(dst[k], buf.data(), (size_t)n * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
}

static void repartition(Job& J, Rank& R, int itime, const char* why, bool recut)
{
    const double t0 = now_s();
    const int n = R.S.g.d.num, N = R.S.g.d.numP;
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: repartition(%s, recut=%d) start\n", R.id, why, (int)recut);

    // 1) publish my positions and my dynamic state, both indexed by gid
    {
        std::vector<double> buf((size_t)n);
        for (int c = 0; c < 3; ++c)
        {
            CUDA_CHECK(cudaMemcpy(buf.data(), R.S.g.x + (size_t)c * N,
                                  (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                J.pos[(size_t)R.gid[i] * 3 + c] = buf[i];
        }
    }
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: positions published\n", R.id);
    mig_snapshot(J, R);
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: mig snapshot ok\n", R.id);
    J.bar.wait();

    // 2) the new layout, from the current positions (identical on every rank
    //    because rank 0 publishes it and the others wait)
    // 2) the new layout.  P16: a *skin* refresh keeps the ownership -- the cut
    // planes are still balanced, the particles are only a few millimetres from
    // where the cut was made -- and derives each rank's box from its own current
    // particles.  That skips the three whole-deck sorts, the re-cut search and the
    // whole-deck copy, which were most of an event's cost.  A re-cut (an explicit
    // SPH_D1_REPART, or SPH_D1_REGRID) walks the full path.
    if (R.id == 0)
    {
        Part P;                                   // only filled by the re-cut path
        if (recut)
        {
            Deck D;
            D.num = J.num;
            D.x.resize(J.num); D.y.resize(J.num); D.z.resize(J.num); D.w.resize(J.num);
            D.x0.resize(J.num); D.y0.resize(J.num); D.z0.resize(J.num); D.solid.resize(J.num);
            for (int g = 0; g < J.num; ++g)
            {
                D.x[g] = J.pos[(size_t)g * 3 + 0];
                D.y[g] = J.pos[(size_t)g * 3 + 1];
                D.z[g] = J.pos[(size_t)g * 3 + 2];
                const double t = J.deckRows[(size_t)g * 14 + 9];
                D.w[g] = (t > 0.0) ? 1.0 : ((t < -10.0) ? g_w_solid : 0.0);
                D.solid[g] = (t < -10.0) ? 1 : 0;
                D.x0[g] = J.deckRows[(size_t)g * 14 + 10];
                D.y0[g] = J.deckRows[(size_t)g * 14 + 11];
                D.z0[g] = J.deckRows[(size_t)g * 14 + 12];
            }
            auto sort_axis = [&](const std::vector<double>& c)
            {
                std::vector<int> o(J.num);
                for (int i = 0; i < J.num; ++i) o[i] = i;
                std::sort(o.begin(), o.end(), [&](int a, int b)
                { return (c[a] < c[b]) || (c[a] == c[b] && a < b); });
                return o;
            };
            D.ordX = sort_axis(D.x);
            D.ordY = sort_axis(D.y);
            D.ordZ = sort_axis(D.z);

            std::vector<int> ixOf(J.num), iyOf(J.num), izOf(J.num), acc;
            P = make_part(D, J.gnx, J.gny, J.gnz, J.skin, ixOf, iyOf, izOf, acc);
            if (J.regrid)
            {
                // re-search the grid with the current positions
                Part best = P;
                for (int nx = 1; nx <= J.K; ++nx)
                    for (int ny = 1; ny <= J.K / nx; ++ny)
                    {
                        if (J.K % (nx * ny) != 0) continue;
                        const int nz = J.K / (nx * ny);
                        if (nx == J.gnx && ny == J.gny && nz == J.gnz) continue;
                        Part Q = make_part(D, nx, ny, nz, J.skin, ixOf, iyOf, izOf, acc);
                        if (Q.empty) continue;
                        if (Q.maxload < best.maxload ||
                            (Q.maxload == best.maxload && Q.ghosts < best.ghosts)) best = std::move(Q);
                    }
                P = std::move(best);
                J.gnx = P.nx; J.gny = P.ny; J.gnz = P.nz;
            }
            J.owner = P.owner;
        }
        else
        {
            // keep J.owner, refresh the boxes from the live positions
            std::vector<double> box((size_t)6 * J.K);
            for (int r = 0; r < J.K; ++r)
            {
                box[6 * r + 0] = box[6 * r + 2] = box[6 * r + 4] = 1e300;
                box[6 * r + 1] = box[6 * r + 3] = box[6 * r + 5] = -1e300;
            }
            for (int g = 0; g < J.num; ++g)
            {
                const int r = J.owner[(size_t)g];
                for (int c = 0; c < 3; ++c)
                {
                    const double v = J.pos[(size_t)g * 3 + c];
                    if (v < box[6 * r + 2 * c])     box[6 * r + 2 * c] = v;
                    if (v > box[6 * r + 2 * c + 1]) box[6 * r + 2 * c + 1] = v;
                }
            }
            J.box = std::move(box);
                }
        // C10: the ghost selection tests the *current* positions (J.pos), not the
        // t = 0 deck -- that is now a parameter, so the whole-deck copy the C10 fix
        // needed is gone as well.
        if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] before build_ranks (recut %d, pos %p)\n", (int)recut, (const void*)J.pos.data());
        build_ranks(J, J.deckRows, J.owner, J.box, J.pos.data());
        if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] after build_ranks\n");
        std::printf("[D1] %s at step %d (%s): grid %dx%dx%d, ghosts %lld (+%.1f%%)%s\n",
                    recut ? "repartition" : "halo refresh", itime, why,
                    J.gnx, J.gny, J.gnz, J.ghost_total,
                    100.0 * (double)J.ghost_total / (double)J.num,
                    recut ? "" : ", ownership kept");
    }
    J.bar.wait();

    // 3) rebuild this rank: deck statics + migrated dynamics + ghost statics
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: repartition stage 3 (rows %zu, numP %d)\n", R.id, R.rows.size(), R.numP);
    xch_free(R);
    R.S.free_state();
    if (!R.S.finish_deck(R.rows, R.label.c_str(), R.ownNum, &J.cellgrid, R.ngs1))
    {
        fprintf(stderr, "[D1] rank %d: cannot rebuild its deck\n", R.id);
        std::exit(1);
    }
    CUDA_CHECK(cudaMalloc((void**)&R.S.g.gid, R.gid.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(R.S.g.gid, R.gid.data(), R.gid.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: finish_deck ok (numP %d ngs1 %d)\n", R.id, R.S.g.d.numP, R.S.g.d.ngs1);
    mig_restore(J, R);
    monitor_init(R);                      // C9: the new reference for the trigger
    R.rebuild_due = false;
    R.disp = 0.0;
    build_send_lists(J, R);
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: send lists ok (nSendE1 %zu)\n", R.id, R.nSendE1);
    launch_copy_xf(R.S.g, 0);
    xch_static(J, R);
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: static exchange ok\n", R.id);                     // ghost A/vecn/p/irho/SurFD
    launch_build_solid_pack(R.S.g, 0);
    if (sph_use_static_solid) R.S.build_solid_lists();
    CUDA_CHECK(cudaDeviceSynchronize());

    R.reparts++;
    R.repart_s += now_s() - t0;
    if (getenv("SPH_D1_DBG"))
    {
        // C11: with the exchange arrays freed, this must not grow across events
        size_t freem = 0, totm = 0;
        cudaMemGetInfo(&freem, &totm);
        std::printf("[D1] rank %d: %.0f MB of %.0f MB in use after event %lld\n",
                    R.id, (totm - freem) / 1048576.0, totm / 1048576.0, R.reparts);
    }
    if (R.id == 0)
        std::printf("[D1] repartition done: %.1f ms (rank 0 wall), %lld event(s)\n",
                    1000.0 * (now_s() - t0), R.reparts);
}

// ---------------------------------------------------------------------------
// one rank thread
// ---------------------------------------------------------------------------
// Build the per-exchange gather/scatter lists and the three buffers, once.
// The gather side only needs local information (which owned particles go to
// which peer, and where their records live in my send block), so it is built
// before the peers' buffers exist; the relay offsets (peerOff) come from the
// same pass in build_job, so the scatter side can be built immediately too.
static void build_send_lists(Job& J, Rank& R)
{
    CUDA_CHECK(cudaSetDevice(R.gpu));
    const size_t sbytes = (std::max<size_t>(R.sendCap, 1) + 1) * sizeof(double);   // +1: P17 tail
    const size_t rbytes = std::max<size_t>(R.recvCap, 1) * sizeof(double);
    void* p = nullptr;
    for (int s = 0; s < 2; ++s)
    {
        p = nullptr;
        CUDA_CHECK(cudaHostAlloc(&p, sbytes, cudaHostAllocDefault));
        R.hSendB[s] = (double*)p;
    }
    p = nullptr;
    CUDA_CHECK(cudaHostAlloc(&p, rbytes, cudaHostAllocDefault));
    R.hRecv = (double*)p;
    CUDA_CHECK(cudaMalloc((void**)&R.dStage, std::max(sbytes, rbytes)));

    auto up_i = [&](const std::vector<int>& v, int*& d)
    {
        if (v.empty()) return;
        CUDA_CHECK(cudaMalloc((void**)&d, v.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice));
    };
    std::vector<int> idx, off;
    std::vector<int> idx2, off2;
    for (int g = 0; g < 5; ++g)
    {
        idx.clear(); off.clear();
        if (g == 0 || g == 1 || g == 4)     // groups 2/3 were E2/E3 (P11/P8)
            for (const DLink& L : R.links)
            {
                if (g == 0)
                    for (int i = 0; i < (int)L.sendF.size(); ++i)
                    { idx.push_back(L.sendF[i]); off.push_back(L.off[0] + SPH_XCH_REC_F * i); }
                else if (g == 1)
                    for (int i = 0; i < (int)L.sendS.size(); ++i)
                    { idx.push_back(L.sendS[i]); off.push_back(L.off[1] + SPH_XCH_REC_S * i); }
               else
                {
                    int i = 0;
                    for (int v : L.sendS) { idx.push_back(v); off.push_back(L.off[4] + SPH_XCH_REC_ST * i); ++i; }
                    for (int v : L.sendW) { idx.push_back(v); off.push_back(L.off[4] + SPH_XCH_REC_ST * i); ++i; }
                }
            }
        R.nSend[g] = (int)idx.size();
        up_i(idx, R.dSendIdx[g]);
        up_i(off, R.dSendOff[g]);
    }

    for (int g = 0; g < 5; ++g)
    {
        idx2.clear(); off2.clear();
        if (g == 0 || g == 1 || g == 4)
            for (const DLink& L : R.links)
            {
                if (g == 0)
                    for (int i = 0; i < (int)L.recvF.size(); ++i)
                    { idx2.push_back(L.recvF[i]); off2.push_back(L.recvOff[0] + SPH_XCH_REC_F * i); }
                else if (g == 1)
                    for (int i = 0; i < (int)L.recvS.size(); ++i)
                    { idx2.push_back(L.recvS[i]); off2.push_back(L.recvOff[1] + SPH_XCH_REC_S * i); }
               else
                {
                    int i = 0;
                    for (int v : L.recvS) { idx2.push_back(v); off2.push_back(L.recvOff[4] + SPH_XCH_REC_ST * i); ++i; }
                    for (int v : L.recvW) { idx2.push_back(v); off2.push_back(L.recvOff[4] + SPH_XCH_REC_ST * i); ++i; }
                }
            }
        R.nRecv[g] = (int)idx2.size();
        up_i(idx2, R.dRecvIdx[g]);
        up_i(off2, R.dRecvOff[g]);
    }
}
static void rank_main(Job* Jp, int r)
{
    Job& J = *Jp;
    Rank& R = J.ranks[r];
    tl_job = &J;
    tl_rank = r;
    CUDA_CHECK(cudaSetDevice(R.gpu));

    // exchange buffers and both halves of the lists, once
    build_send_lists(J, R);

    // own rows are the deck's; the ghosts follow, so numP = own + ghosts
    if (!R.S.finish_deck(R.rows, R.label.c_str(), R.ownNum, &J.cellgrid, R.ngs1))
    {
        fprintf(stderr, "[D1] rank %d: cannot load its deck\n", r);
        std::exit(1);
    }
    CUDA_CHECK(cudaMalloc((void**)&R.S.g.gid, R.gid.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(R.S.g.gid, R.gid.data(), R.gid.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    launch_copy_xf(R.S.g, 0);
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: deck loaded, numP=%d\n", r, R.S.g.d.numP);
    monitor_init(R);                  // C9: the reference for the displacement trigger
    R.S.build_cells();
    R.S.init_reference_state();
    CUDA_CHECK(cudaDeviceSynchronize());
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: reference state ok\n", r);
    J.bar.wait();

    // the ghost block's static fields, then the solid pack/list that needs them
    xch_static(J, R);
    launch_build_solid_pack(R.S.g, 0);
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: static exchange + pack ok\n", r);
    if (sph_use_static_solid) R.S.build_solid_lists();
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: solid lists ok (npair %d, nsolC %d)\n", r, R.S.g.npair, R.S.g.d.nsolC);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        const int nfb = gra_fallback_count();
        if (nfb != 0)
        {
            std::lock_guard<std::mutex> lk(J.print_m);
            fprintf(stderr, "[D1] rank %d: warning: %d particle(s) fell back to A = I\n", r, nfb);
        }
    }

    R.t_start = now_s();
    if (J.prof)
    {
        // where does the cell-list work go?
        const SphDims& d = R.S.g.d;
        std::vector<int> st((size_t)d.ncell + 1);
        CUDA_CHECK(cudaMemcpy(st.data(), R.S.g.cellStart, ((size_t)d.ncell + 1) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        int mx = 0, at = -1;
        long long tot = 0;
        for (int c = 0; c < d.ncell; ++c)
        {
            const int w = st[c + 1] - st[c];
            tot += w;
            if (w > mx) { mx = w; at = c; }
        }
        std::printf("[D1] rank %d: grid %dx%dx%d cells=%d particles=%lld "
                    "max cell=%d at %d (%d,%d,%d) start[0]=%d start[N]=%d\n",
                    r, d.nx, d.ny, d.nz, d.ncell, tot, mx, at,
                    at % d.nx, (at / d.nx) % d.ny, at / (d.nx * d.ny), st[0], st[d.ncell]);
    }
    for (int itime = 0; itime < J.maxStep; ++itime)
    {
        const auto s0 = std::chrono::steady_clock::now();
        step_d1(J, R, itime);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto s1 = std::chrono::steady_clock::now();
        R.step_s += std::chrono::duration<double>(s1 - s0).count();

        if (itime % 500 == 0)
        {
            // P14: hand the file to a writer thread and keep stepping.  Two stages
            // are enough: a stage is only reused two results later, and rank 0
            // waits for that writer before the barrier that releases it.
            // C20: derived, not stored -- every rank computes the same value
            const int s = (itime / 500) & 1;
            if (r == 0 && J.outBusy[s])
            {
                J.outFut[s].wait();
                J.outBusy[s] = false;
            }
            J.bar.wait();                 // stage s is free on every rank
            fetch_own(J, R, J.out[s]);
            J.bar.wait();                 // it now holds every rank's particles
            if (r == 0)
            {
                const int idx = itime / 500;
                Job::OutBuf& O = J.out[s];
                J.outFut[s] = std::async(std::launch::async,
                                         [&J, &O, idx] { write_global_result(J, O, idx); });
                J.outBusy[s] = true;
            }
        }

        // repartition at a step boundary, after the result write, so the file
        // still carries the decomposition it was fetched with
        bool due = false;
        if (J.repart > 0 && itime > 0 && (itime % J.repart) == 0) due = true;
        // C9: the skin has been spent -- rebuild before a pair can be lost.  The
        // decision is identical on every rank (same maximum, same threshold).
        const bool skin_due = R.rebuild_due;
        if ((due || skin_due) && itime < J.maxStep - 1)
        {
            char why[128];
            if (skin_due)
                snprintf(why, sizeof(why), "skin spent: %.3f mm moved, skin %.2f mm",
                         1000.0 * R.last_disp, 1000.0 * J.skin);
            else
                snprintf(why, sizeof(why), "SPH_D1_REPART");
            repartition(J, R, itime, why, due);
        }
    }
    if (r == 0)                       // P14: let the last writers finish
        for (int s = 0; s < 2; ++s)
            if (J.outBusy[s]) J.outFut[s].wait();
    R.t_end = now_s();
}

// ---------------------------------------------------------------------------
// L1: SPH_RANKS=auto -- rank the devices, then choose K from the partition.
//
// The review's point is that "how many cards" was never a measured decision:
// the default list was 0..K-1, which on this machine always includes the display
// GPU, and every exchange point ends in a barrier, so the slowest card sets the
// pace.  Two halves:
//
//   1. rank the devices by a bandwidth probe (a 64 MB device-to-device copy) and
//      by free memory, demoting the ones that carry a display watchdog;
//   2. for K = 1..ndev evaluate the real partition (every factorisation, exactly
//      as build_job does) and predict
//          T(K) = max_r(own_w) + w_ghost * ghosts/K + XCH_UNITS
//      i.e. the slowest rank's compute weight plus its share of the ghosts plus
//      one exchange-point floor.  The smallest K within SPH_D1_AUTO_SLACK of the
//      best prediction wins, so a batch of cases keeps more cards free.
//
// Both model constants are knobs (SPH_D1_XCH_UNITS, SPH_D1_AUTO_SLACK) and the
// table is printed, so the choice can be checked rather than trusted.
// ---------------------------------------------------------------------------
struct DevScore { int id = 0; double bw = 0.0; double freemb = 0.0; int display = 0; };

static std::vector<DevScore> rank_devices()
{
    std::vector<DevScore> v;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) return v;
    const size_t bytes = (size_t)64 << 20;
    for (int d = 0; d < ndev; ++d)
    {
        if (cudaSetDevice(d) != cudaSuccess) continue;
        DevScore s;
        s.id = d;
        size_t freem = 0, totalm = 0;
        cudaMemGetInfo(&freem, &totalm);
        s.freemb = (double)freem / 1048576.0;
        int wdt = 0;
        cudaDeviceGetAttribute(&wdt, cudaDevAttrKernelExecTimeout, d);
        s.display = wdt != 0;
        void* a = nullptr; void* b = nullptr;
        if (cudaMalloc(&a, bytes) == cudaSuccess && cudaMalloc(&b, bytes) == cudaSuccess)
        {
            cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice);   // warm up
            cudaDeviceSynchronize();
            const double t0 = now_s();
            const int R = 3;
            for (int i = 0; i < R; ++i) cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            const double dt = now_s() - t0;
            if (dt > 0) s.bw = (double)(R * bytes) / dt / 1e9;   // GB/s
        }
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        v.push_back(s);
    }
    std::sort(v.begin(), v.end(), [](const DevScore& a, const DevScore& b)
    {
        if (a.display != b.display) return a.display < b.display;   // real GPUs first
        if (a.bw != b.bw) return a.bw > b.bw;
        return a.freemb > b.freemb;
    });
    return v;
}

// best partition of the deck into K blocks: the same search build_job runs,
// without touching a device
static Part best_partition(const Deck& D, int K, double skin)
{
    std::vector<int> ixOf(D.num), iyOf(D.num), izOf(D.num), acc;
    Part best;
    bool have = false;
    for (int nx = 1; nx <= K; ++nx)
        for (int ny = 1; ny <= K / nx; ++ny)
        {
            if (K % (nx * ny) != 0) continue;
            const int nz = K / (nx * ny);
            Part P = make_part(D, nx, ny, nz, skin, ixOf, iyOf, izOf, acc);
            if (P.empty) continue;
            if (!have || P.maxload < best.maxload ||
                (P.maxload == best.maxload && P.ghosts < best.ghosts))
            {
                best = std::move(P);
                have = true;
            }
        }
    if (!have) best.empty = true;
    return best;
}

static int choose_auto_k(const Deck& D, int ndev, double skin, std::vector<double>& pred)
{
    double xch_units = 4000.0, slack = 0.10;
    if (const char* e = getenv("SPH_D1_XCH_UNITS")) xch_units = atof(e);
    if (const char* e = getenv("SPH_D1_AUTO_SLACK")) slack = atof(e);
    pred.assign((size_t)ndev + 1, 0.0);
    std::printf("[D1] K prediction (load units): ghost weight %.2f, solid %.2f, "
                "exchange floor %.0f\n", g_w_ghost, g_w_solid, xch_units);
    int bestK = 1;
    double bestT = 0.0;
    for (int K = 1; K <= ndev; ++K)
    {
        const Part P = best_partition(D, K, skin);
        if (P.empty) { pred[K] = -1.0; std::printf("[D1]   K=%d: no non-empty split\n", K); continue; }
        const double t = P.maxload + g_w_ghost * (double)P.ghosts / (double)K + xch_units;
        pred[K] = t;
        std::printf("[D1]   K=%d: %dx%dx%d ghost %lld, load %.0f -> T = %.0f\n",
                    K, P.nx, P.ny, P.nz, P.ghosts, P.maxload, t);
        if (bestT == 0.0 || t < bestT) { bestT = t; bestK = K; }
    }
    // prefer the smallest K that is within `slack` of the best prediction
    int pick = bestK;
    for (int K = 1; K < bestK; ++K)
        if (pred[K] > 0.0 && pred[K] <= bestT * (1.0 + slack)) { pick = K; break; }
    std::printf("[D1] SPH_RANKS=auto: %d device(s), best K=%d, chosen K=%d (slack %.0f%%)\n",
                ndev, bestK, pick, 100.0 * slack);
    return pick;
}

static int run(const char* deck, int maxStep, int K, const std::vector<int>& gpus);

// L1: rank the devices, choose K, then run the ordinary driver on that set.
static int run_auto(const char* deck, int maxStep)
{
    std::vector<double> rows;
    int num = 0, ntotal = 0, solnum = 0;
    if (!read_deck(deck, rows, num, ntotal, solnum)) return 1;

    const std::vector<DevScore> devs = rank_devices();
    if (devs.empty()) { fprintf(stderr, "[D1] SPH_RANKS=auto: no CUDA device\n"); return 2; }
    std::printf("[D1] devices (ranked):");
    for (const DevScore& d : devs)
        std::printf(" %d(%.0f GB/s, %.0f MB free%s)", d.id, d.bw, d.freemb,
                    d.display ? ", display" : "");
    std::printf("\n");

    Deck D;
    D.num = num;
    D.x.resize(num); D.y.resize(num); D.z.resize(num); D.w.resize(num);
    D.x0.resize(num); D.y0.resize(num); D.z0.resize(num); D.solid.resize(num);
    for (int i = 0; i < num; ++i)
    {
        const double* r = &rows[(size_t)i * 14];
        D.x[i] = r[0]; D.y[i] = r[1]; D.z[i] = r[2];
        const double t = r[9];
        D.w[i] = (t > 0.0) ? 1.0 : ((t < -10.0) ? g_w_solid : 0.0);
        D.solid[i] = (t < -10.0) ? 1 : 0;
        D.x0[i] = r[10]; D.y0[i] = r[11]; D.z0[i] = r[12];
    }
    auto sort_axis = [&](const std::vector<double>& c)
    {
        std::vector<int> o(num);
        for (int i = 0; i < num; ++i) o[i] = i;
        std::sort(o.begin(), o.end(), [&](int a, int b)
        { return (c[a] < c[b]) || (c[a] == c[b] && a < b); });
        return o;
    };
    D.ordX = sort_axis(D.x);
    D.ordY = sort_axis(D.y);
    D.ordZ = sort_axis(D.z);

    double skin = SPH_D1_SKIN_DEFAULT;
    if (const char* e = getenv("SPH_D1_SKIN")) skin = atof(e);
    if (skin < 0.0) skin = 0.0;

    std::vector<double> pred;
    const int K = choose_auto_k(D, (int)devs.size(), skin, pred);
    std::vector<int> gpus;
    for (int i = 0; i < K; ++i) gpus.push_back(devs[(size_t)i].id);
    return run(deck, maxStep, K, gpus);
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------
static int run(const char* deck, int maxStep, int K, const std::vector<int>& gpus)
{
    Job J;
    J.K = K;
    J.bar.K = K;
    J.maxStep = maxStep;
    J.deck = deck;
    J.prof = (getenv("SPH_D1PROF") != nullptr);
    // C12: from here on build_solid_lists() decides the A6 layout collectively
    sph_a3_fat_hook = &a3_fat_agreed;
    if (const char* b = getenv("SPH_BARRIER"))
        g_barrier_mode = (!strcmp(b, "spin")) ? 1 : (!strcmp(b, "block")) ? 2 : 0;
    if (const char* e = getenv("SPH_D1_REPART")) J.repart = atoi(e);
    {
        const char* e = getenv("SPH_D1_REGRID");
        J.regrid = (e != nullptr) && atoi(e) != 0;
    }
    // C9: the ghost shell is widened by a skin and re-chosen whenever a particle
    // has spent half of it.  SPH_D1_SKIN=0 restores the old behaviour (ghosts
    // picked once, never revalidated) and is only useful for A/B testing.
    J.skin = SPH_D1_SKIN_DEFAULT;
    if (const char* e = getenv("SPH_D1_SKIN")) J.skin = atof(e);
    if (J.skin < 0.0) J.skin = 0.0;

    std::vector<double> rows;
    if (!read_deck(deck, rows, J.num, J.ntotal, J.solnum)) return 1;

    // C13: E1 carries only the fluid and solid groups, so a wall ghost would
    // never see its owner's velocity.  Every wall of this deck is static (and a
    // static wall cannot start moving: no kernel writes SForce or PIJW for it),
    // so refuse the decomposition instead of running wrong.
    for (int i = 0; i < J.num; ++i)
    {
        const double* rw = &rows[(size_t)i * 14];
        const double t = rw[9];
        if (t > 0.0 || t < -10.0) continue;            // fluid or solid
        if (rw[4] != 0.0 || rw[5] != 0.0 || rw[6] != 0.0)
        {
            fprintf(stderr, "[D1] particle %d is a wall with velocity (%g, %g, %g); "
                            "the decomposition does not exchange wall ghosts, so a "
                            "moving wall needs a single rank (SPH_RANKS unset)\n",
                    i, rw[4], rw[5], rw[6]);
            return 2;
        }
    }
    std::printf("[D1] %s: num=%d fluid=%d solid=%d wall=%d, %d rank(s) on gpu",
                deck, J.num, J.ntotal, J.solnum, J.num - J.ntotal - J.solnum, K);
    if (const char* s = getenv("SPH_D1_SPLIT")) std::printf(" (forced split %s)", s);
    for (int i = 0; i < K; ++i) std::printf(" %d", gpus[i]);
    std::printf("\n");

    // global box for the result header, from the same rule finish_deck uses
    {
        double mn[3] = { 1e300, 1e300, 1e300 }, mx[3] = { -1e300, -1e300, -1e300 };
        for (int i = 0; i < J.num; ++i)
            for (int c = 0; c < 3; ++c)
            {
                const double v = rows[(size_t)i * 14 + c];
                if (v < mn[c]) mn[c] = v;
                if (v > mx[c]) mx[c] = v;
            }
        const double cs = SPH_CS, margin = cs + 0.02;
        const double gx0 = mn[0] - margin, gy0 = mn[1] - margin, gz0 = mn[2] - margin;
        const double gx1 = mx[0] + margin, gy1 = mx[1] + margin, gz1 = mx[2] + margin;
        const int nx = (int)floor((gx1 - gx0) / cs) + 1;
        const int ny = (int)floor((gy1 - gy0) / cs) + 1;
        const int nz = (int)floor((gz1 - gz0) / cs) + 1;
        J.grid[0] = gx0; J.grid[1] = gy0; J.grid[2] = gz0;
        J.grid[3] = gx0 + nx * SPH_CS;
        J.grid[4] = gy0 + ny * SPH_CS;
        J.grid[5] = gz0 + nz * SPH_CS;
        // V2: this is the grid every rank bins into, so the cell visiting order
        // matches the single-GPU run bit for bit
        J.cellgrid.org[0] = gx0; J.cellgrid.org[1] = gy0; J.cellgrid.org[2] = gz0;
        J.cellgrid.nx = nx; J.cellgrid.ny = ny; J.cellgrid.nz = nz;
    }

    if (!build_job(J, rows)) return 1;
    // the full deck is kept: a repartition rebuilds the per-rank rows from it
    J.deckRows = std::move(rows);
    // C12 needs one slot per rank whatever the trigger configuration is
    J.a3_fat.assign((size_t)J.K, 0.0);
    if (J.repart > 0 || J.skin > 0.0)
    {
        J.pos.assign((size_t)3 * J.num, 0.0);
        J.mig.assign((size_t)MIG_N * J.num, 0.0);
    }
    if (J.repart > 0)
        std::printf("[D1] repartition every %d steps%s\n", J.repart,
                    J.regrid ? " (re-searching the block grid)" : "");
    if (J.skin > 0.0)
        std::printf("[D1] ghost skin %.2f mm (%.2f h): rebuild when a particle has "
                    "moved %.2f mm\n", 1000.0 * J.skin, J.skin / SPH_HSML,
                    500.0 * J.skin);
    else
        fprintf(stderr, "[D1] WARNING: SPH_D1_SKIN=0 -- the ghost set is chosen once "
                        "and never revalidated, so long runs silently drop "
                        "interactions (C9, see review/Claude)\n");
    for (int r = 0; r < K; ++r) J.ranks[r].gpu = gpus[r];

    for (int s = 0; s < 2; ++s)      // P14: two result staging buffers
    {
        J.out[s].gx.assign(J.num, 0.0); J.out[s].gy.assign(J.num, 0.0);
        J.out[s].gz.assign(J.num, 0.0); J.out[s].gp.assign(J.num, 0.0);
    }

    {
        long long own = 0;
        for (int r = 0; r < K; ++r) own += J.ranks[r].ownNum;
        std::printf("[D1] block grid %dx%dx%d, %lld ghost slots over %lld owned "
                    "particles (%.1f%% duplicate)\n",
                    J.gnx, J.gny, J.gnz, J.ghost_total, own,
                    100.0 * (double)J.ghost_total / (double)(own > 0 ? own : 1));
    }
    for (int r = 0; r < K; ++r)
    {
        const Rank& R = J.ranks[r];
        int nf = 0, ns = 0, nw = 0, gf = 0, gs = 0, gw = 0;
        for (int i = 0; i < R.ownNum; ++i)
        {
            const double t = R.rows[(size_t)i * 14 + 9];
            if (t > 0.0) ++nf; else if (t < -10.0) ++ns; else ++nw;
        }
        for (int i = R.ownNum; i < R.numP; ++i)
        {
            const double t = R.rows[(size_t)i * 14 + 9];
            if (t > 0.0) ++gf; else if (t < -10.0) ++gs; else ++gw;
        }
        std::printf("[D1] rank %d gpu %d: own %d (fluid %d solid %d wall %d), "
                    "ghost %d (fluid %d solid %d wall %d), peers %zu\n",
                    r, R.gpu, R.ownNum, nf, ns, nw, R.numP - R.ownNum, gf, gs, gw,
                    R.links.size());
        if (J.prof)
            for (const DLink& L : R.links)
                std::printf("[D1] rank %d link -> rank %d: send f/s/w %zu/%zu/%zu, "
                            "recv f/s/w %zu/%zu/%zu\n",
                            r, L.peer, L.sendF.size(), L.sendS.size(), L.sendW.size(),
                            L.recvF.size(), L.recvS.size(), L.recvW.size());
    }

    std::vector<std::thread> th;
    th.reserve(K);
    for (int r = 0; r < K; ++r) th.emplace_back(rank_main, &J, r);
    for (auto& t : th) t.join();

    double wall = 0.0;
    for (int r = 0; r < K; ++r) wall = std::max(wall, J.ranks[r].t_end - J.ranks[r].t_start);
    const int steps = maxStep > 0 ? maxStep : 1;
    for (int r = 0; r < K; ++r)
    {
        const Rank& R = J.ranks[r];
        // P19: the wall clock, not just the sum of the step kernels -- the previous
        // figure excluded the halo rebuilds and the result writes, which are a large
        // part of a long run.
        const double wall_r = R.t_end - R.t_start;
        std::printf("[D1] rank %d: %d steps, %.3f s wall (%.3f ms/step), %.3f s in %lld "
                    "halo rebuild(s), %.3f s parked on the exchange barrier\n",
                    r, maxStep, wall_r, 1000.0 * wall_r / steps,
                    R.repart_s, R.reparts, R.x_bar);
    }
    std::printf("[D1] %d step(s), %.3f s wall, %.3f ms/step (slowest rank, wall clock)\n",
                maxStep, wall, 1000.0 * wall / steps);
    if (K > 0 && !J.ranks.empty())
        std::printf("[D1] halo rebuilds: %lld, %.3f s of rank 0's wall clock\n",
                    J.ranks[0].reparts, J.ranks[0].repart_s);
    if (J.prof)
    {
        static const char* nm[14] = { "disp", "prepare", "E1", "memset", "cell_hist", "cell_scan",
                                      "cell_scatter", "cell_sort", "pass1_fluid",
                                      "pass1_solid", "constitution",
                                      "pass2_fluid", "pass2_solid", "update" };
        for (int r = 0; r < K; ++r)
        {
            const Rank& R = J.ranks[r];
            std::printf("[D1] rank %d phases (ms/step):", r);
            for (int k = 0; k < 14; ++k) std::printf(" %s=%.3f", nm[k], 1000.0 * R.ph[k] / steps);
            std::printf("\n[D1] rank %d exchange split (ms/step): gather-launch=%.3f "
                        "copyD2H+sync=%.3f relay=%.3f copyH2D+scatter=%.3f barriers=%.3f\n",
                        r, 1000.0 * R.x_g / steps, 1000.0 * R.x_c / steps,
                        1000.0 * R.x_r / steps, 1000.0 * R.x_h / steps,
                        1000.0 * R.x_bar / steps);
        }
    }
    return 0;
}

}  // namespace d1
#endif  // SPH_HAVE_FP64

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
        CUDA_CHECK(cudaMemsetAsync(S.g.cellCur, 0, (size_t)(S.g.d.ncell + 1) * sizeof(int), 0));
        CUDA_CHECK(cudaMemsetAsync(S.g.cellSolid, 0, S.g.d.ncell * sizeof(int), 0));
        CUDA_CHECK(cudaMemsetAsync(S.g.cellFluid, 0, S.g.d.ncell * sizeof(int), 0));
            launch_cell_hist(S.g, S.g.cellCur, S.g.cellSolid, S.g.cellFluid, 0);
            launch_cell_scan(S.g.cellStart, S.g.cellCur, S.g.d.ncell, S.d_scan_tmp, S.d_scan_bytes, 0);
            launch_cell_scatter(S.g, S.g.cellCur, 0);
        launch_cell_sort(S.g.cellStart, S.g.sorted, S.g.d.ncell, S.g.gid, 0);
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
    // A6: which solid-list layout to use.  -1 = auto (lean when the fat list
    // would dominate VRAM), 0 = fat (sG/sB/sD stored, fastest), 1 = lean.
    // SPH_A3_FULL=0/1 forces lean/fat; SPH_A3_LIMIT_MB sets the auto threshold.
    sph_a3_lean = -1;
    if (const char* e = getenv("SPH_A3_FULL")) sph_a3_lean = (atoi(e) == 0) ? 1 : 0;
    // M8: the z = 1 reflection.  'flip' is 1.cpp's unconditional form.
    if (const char* e = getenv("SPH_REFLECT")) sph_reflect_guard = strcmp(e, "flip") != 0;

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

    // ------------------------------------------------------------------
    // D1: SPH_RANKS=K >= 1 switches to the domain-decomposed driver.
    // SPH_RANKS_GPUS=0,1,... picks the devices (default 0..K-1).
    // ------------------------------------------------------------------
    if (const char* e = getenv("SPH_RANKS"))
    {
        const int K = atoi(e);
        if (!strcmp(e, "auto"))             // L1: rank the devices, choose K
        {
            if (want_f32)
            {
                fprintf(stderr, "SPH_RANKS=auto: the multi-GPU path is fp64 only; "
                                "drop --fp32 / SPH_PRECISION=fp32\n");
                return 2;
            }
#if SPH_HAVE_FP64
            return d1::run_auto(input, iMaxStep);
#else
            fprintf(stderr, "this build has no fp64 path (SPH_ENABLE_FP64=OFF)\n");
            return 2;
#endif
        }
        if (K < 1) { fprintf(stderr, "SPH_RANKS must be >= 1 or 'auto' (got '%s')\n", e); return 2; }
        if (want_f32)
        {
            fprintf(stderr, "SPH_RANKS=%d: the multi-GPU path is fp64 only; "
                            "drop --fp32 / SPH_PRECISION=fp32\n", K);
            return 2;
        }
        std::vector<int> gpus;
        if (const char* g = getenv("SPH_RANKS_GPUS"))
        {
            const char* q = g;
            while (*q)
            {
                while (*q == ' ' || *q == ',' || *q == ';') ++q;
                if (!*q) break;
                char* end = nullptr;
                const long v = strtol(q, &end, 10);
                if (end == q) { fprintf(stderr, "SPH_RANKS_GPUS: cannot parse '%s'\n", g); return 2; }
                gpus.push_back((int)v);
                q = end;
            }
        }
        else
        {
            for (int i = 0; i < K; ++i) gpus.push_back(i);
        }
        if ((int)gpus.size() < K)
        {
            fprintf(stderr, "SPH_RANKS_GPUS lists %zu device(s), %d needed\n", gpus.size(), K);
            return 2;
        }
        gpus.resize(K);
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0)
        {
            fprintf(stderr, "SPH_RANKS: no CUDA device available\n");
            return 2;
        }
        for (int i = 0; i < K; ++i)
            if (gpus[i] < 0 || gpus[i] >= ndev)
            {
                fprintf(stderr, "SPH_RANKS_GPUS: device %d out of range (0..%d)\n",
                        gpus[i], ndev - 1);
                return 2;
            }
#if SPH_HAVE_FP64
        if (profile)
            fprintf(stderr, "[D1] note: SPH_PROFILE is not implemented on the decomposed path\n");
        return d1::run(input, iMaxStep, K, gpus);
#else
        (void)gpus;
        fprintf(stderr, "this build has no fp64 path (SPH_ENABLE_FP64=OFF)\n");
        return 2;
#endif
    }

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
























