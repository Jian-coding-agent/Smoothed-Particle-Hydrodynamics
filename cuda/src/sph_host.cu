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
#include <filesystem>

// ---------------------------------------------------------------------------
// Result cadence (SPH_OUTCADENCE).  The historical value is 500; a smaller
// cadence is what a video needs, and 0 disables result frames entirely.
// ---------------------------------------------------------------------------
static int g_out_cadence = 500;

// ---------------------------------------------------------------------------
// Checkpointing (single-GPU path): SPH_CKPT writes, SPH_CKPT_IN resumes,
// SPH_CKPT_EVERY is the cadence (0 = only at the end), SPH_STEP0 is the absolute
// step number of a fresh run.  cuda/docs/CHECKPOINT.md has the format and the
// reasoning; the array list is in Solver::ckpt_arrays().
// ---------------------------------------------------------------------------
static const char SPHCHK_MAGIC[8] = { 'S', 'P', 'H', 'C', 'H', 'K', '\0', '\0' };
static const uint32_t SPHCHK_VERSION = 2;
static const char* g_ckpt_out = nullptr;
static const char* g_ckpt_in = nullptr;
static int  g_ckpt_every  = 0;
static bool g_ckpt_wanted = false;   // any checkpoint in play -> hash the deck
// The solid wall clamp as the user asked for it (main() parses it).  The device
// only ever sees a copy; a resume is compared against these host values.
static double g_soldbox_z = 0.01, g_soldbox_y0 = 0.0, g_soldbox_y1 = 3.31;

// FNV-1a over raw bytes: catches "resumed against the wrong deck".
static uint64_t sph_fnv1a(uint64_t h, const void* p, size_t n)
{
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) { h ^= (uint64_t)b[i]; h *= 1099511628211ULL; }
    return h;
}
static const uint64_t SPH_FNV_OFFSET = 14695981039346656037ULL;

// Fingerprint of the deck's constant columns (mass, itype, X0, rho0).  Both
// checkpoint containers use it, so a resume with a different deck is refused
// before any state is touched.
static uint64_t sph_deck_fingerprint(const std::vector<double>& rows, int num)
{
    const int nf = 14;
    uint64_t h = SPH_FNV_OFFSET;
    for (int i = 0; i < num; ++i)
    {
        const size_t b = (size_t)i * nf;
        h = sph_fnv1a(h, &rows[b + 8], 8);       // mass
        h = sph_fnv1a(h, &rows[b + 9], 8);       // itype
        h = sph_fnv1a(h, &rows[b + 10], 3 * 8);  // X0
        h = sph_fnv1a(h, &rows[b + 13], 8);      // rho0
    }
    return h;
}

// 64-bit stdio seek/tell.  `long` is 32 bits on Windows, and a >2 GB deck (or a
// failed seek) used to give sz = -1 and an out-of-bounds write at buf[rd].
#ifdef _WIN32
#define sph_fseek64 _fseeki64
#define sph_ftell64 _ftelli64
#else
#define sph_fseek64 fseeko
#define sph_ftell64 ftello
#endif

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
    uint64_t deck_fp = 0;   // fingerprint of the deck's constant columns (checkpoint)
    int      nfb     = 0;   // GraCorrec fallbacks (C8), carried through a checkpoint
    long long ckpt_done = -1;   // step of the checkpoint already on disk (avoids a duplicate write)

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
            // 64-bit seek/tell: on Windows `long` is 32 bits, so a >2 GB deck (or
            // a failed ftell) used to give sz = -1 and an out-of-bounds buf[rd].
            sph_fseek64(f, 0, SEEK_END);
            const long long actual = sph_ftell64(f);
            const long long expect = 128 + (long long)num * 14 * 8;
            if (actual != expect)
                fprintf(stderr, "[input] %s: warning: %lld bytes, header implies %lld "
                                "(trailing data ignored)\n", path, actual, expect);
            sph_fseek64(f, 128, SEEK_SET);       // skip the header we already read
            rows.resize((size_t)num * 14);
            if (fread(rows.data(), sizeof(double), rows.size(), f) != rows.size())
            {
                fprintf(stderr, "[input] %s: short read\n", path); fclose(f); return false;
            }
        }
        else
        {
            // 64-bit seek/tell: see above.  A 2 GB text deck is real at 14 columns
            // of ascii (~100 B/row, i.e. 20 M particles).
            sph_fseek64(f, 0, SEEK_END);
            const long long szi = sph_ftell64(f);
            sph_fseek64(f, 0, SEEK_SET);
            if (szi < 0)
            {
                fprintf(stderr, "[input] %s: cannot determine the file size\n", path);
                fclose(f); return false;
            }
            std::vector<char> buf((size_t)szi + 1);
            const size_t rd = fread(buf.data(), 1, (size_t)szi, f);
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

        // Constants the checkpoint does not store (they belong to the deck, not
        // to the run): hash them so a resume with the wrong deck is refused.
        // The decomposed driver hashes the whole deck itself (the rank rows are
        // only a slice).
        if (g_ckpt_wanted && numAll == num) deck_fp = sph_deck_fingerprint(rows, num);

        // The solid clamp defaults follow the 3.7M case (y1 = 3.31).  A deck
        // whose structure reaches beyond that would have every particle above
        // 3.31 frozen at each step, which is a silent physics change -- say so
        // instead of letting the run look plausible.
        if (sph_use_soldbox)
        {
            double maxy = -1e300;
            for (int i = 0; i < numAll; ++i)
                if (rows[(size_t)i * nf + 9] < -10.0)
                {
                    const double y = rows[(size_t)i * nf + 1];
                    if (y > maxy) maxy = y;
                }
            if (maxy > g_soldbox_y1 + 1e-6)
                fprintf(stderr, "[soldbox] warning: solids reach y=%.4f but the clamp is "
                                "y<=%.4f; everything above %.4f is frozen every step.  "
                                "Use SPH_SOLDBOX=%.3f,%.3f,%.3f for this deck (or =off).\n",
                        maxy, g_soldbox_y1, g_soldbox_y1,
                        g_soldbox_z, g_soldbox_y0, maxy + 0.01);
        }
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
        // V2: the same check works on a decomposed rank -- there are numP slots
        // (owned + ghosts) and every one of them must appear in exactly the cell
        // its live position maps to.  This is the regression check for the
        // histogram/scatter pair (the old bug binned x into an X0 histogram).
        const int nv = g.d.numP;
        std::vector<int> st(g.d.ncell + 1), so(nv);
        std::vector<Real> hx(3 * nv);
        CUDA_CHECK(cudaMemcpy(st.data(), g.cellStart, (g.d.ncell + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(so.data(), g.sorted, (size_t)nv * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hx.data(), g.x, (size_t)3 * nv * sizeof(Real), cudaMemcpyDeviceToHost));

        std::vector<int> seen(nv, 0);
        int dup = 0, outOfCell = 0, missing = 0;
        for (int c = 0; c < g.d.ncell; ++c)
        {
            if (st[c] < 0 || st[c] > st[c + 1] || st[c + 1] > nv) { printf("  cellStart bad at %d\n", c); break; }
            for (int a = st[c]; a < st[c + 1]; ++a)
            {
                const int p = so[a];
                if (p < 0 || p >= nv) continue;
                seen[p]++;
                const int cc = host_cell(hx[p], hx[nv + p], hx[2 * nv + p]);
                if (cc != c) outOfCell++;
            }
        }
        for (int p = 0; p < nv; ++p) { if (seen[p] == 0) missing++; if (seen[p] > 1) dup += seen[p] - 1; }
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
    bool write_result(long long itime)
    {
        // value test, not presence: SPH_OUT_F32=0 must mean float64.  Every other
        // switch is atoi-parsed, and the docs say "SPH_OUT_F32=1".
        const char* ef32 = getenv("SPH_OUT_F32");
        const bool f32 = (ef32 != nullptr) && (atoi(ef32) != 0);
        const int idx = (int)(itime / (g_out_cadence > 0 ? g_out_cadence : 1));
        char name[128];
        snprintf(name, sizeof(name), "result%d.sphbin", idx);
        FILE* f = fopen(name, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", name); return false; }

        unsigned char h[128] = {0};
        const char magic[8] = { 'S','P','H','B','I','N','\0','\0' };
        memcpy(h, magic, 8);
        const uint32_t ver = 1, hs = 128, num = (uint32_t)g.d.num,
                       ntotal = (uint32_t)g.d.ntotal, solnum = (uint32_t)g.d.solnum,
                       nfields = 4, step = (uint32_t)itime,
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

        // Every write is checked: a short result file (full disk, dropped share)
        // used to be silent, and the solver has no reader that would notice.
        bool ok = (fwrite(h, 1, 128, f) == 128);

        const size_t rec = 4;
        if (f32)
        {
            std::vector<float> buf(rec * 65536);
            size_t used = 0;
            for (int i = 0; i < g.d.num && ok; ++i)
            {
                if (used + rec > buf.size())
                {
                    ok = (fwrite(buf.data(), sizeof(float), used, f) == used);
                    used = 0;
                }
                buf[used++] = (float)ox[i]; buf[used++] = (float)oy[i];
                buf[used++] = (float)oz[i]; buf[used++] = (float)op[i];
            }
            if (used != 0 && ok) ok = (fwrite(buf.data(), sizeof(float), used, f) == used);
        }
        else
        {
            std::vector<double> buf(rec * 65536);
            size_t used = 0;
            for (int i = 0; i < g.d.num && ok; ++i)
            {
                if (used + rec > buf.size())
                {
                    ok = (fwrite(buf.data(), sizeof(double), used, f) == used);
                    used = 0;
                }
                buf[used++] = ox[i]; buf[used++] = oy[i];
                buf[used++] = oz[i]; buf[used++] = op[i];
            }
            if (used != 0 && ok) ok = (fwrite(buf.data(), sizeof(double), used, f) == used);
        }
        ok = (fflush(f) == 0) && ok;
        ok = (fclose(f) == 0) && ok;
        if (!ok)
        {
            fprintf(stderr, "short write on %s (disk full?) -- frame abandoned\n", name);
            remove(name);
        }
        return ok;
    }

    // ------------------------------------------------------------------
    // Checkpoint (.sphchk): the cross-step state, nothing else.
    //
    // A step reads only what the previous step left in these arrays:
    //      x, vx, rho   the integration variables;
    //      SurF         one step of history -- k_prepare copies it into SurFD
    //                   and the *next* step's pass1 selects eta from SurFD;
    //      A, vecn      built once at t = 0 by init_reference_state() and never
    //                   refreshed.  Both are searched in the *live* positions
    //                   (A also weights with X0, vecn with rho), so recomputing
    //                   them from a deformed state gives a different answer --
    //                   the same trap as the static solid list (C1).
    //      u, xc, rhoc  fp32 only: the displacement form x = X0 + u and the
    //                   Kahan compensations; they cannot be rebuilt from x.
    // Everything else (p, irho, drhodt, dvxdt, viscos, SForce, PIJW, F, DFdt,
    // Pia, SurFD, the cell list) is written before it is read inside one step.
    // X0, rho0, mass, itype are deck constants: not stored, fingerprinted in
    // the header instead.  A resumed run must therefore pass the same deck.
    // ------------------------------------------------------------------
    int ckpt_array_count() const { return (sizeof(Real) < sizeof(double)) ? 9 : 6; }

    struct CkptArr { Real* p; size_t len; };

    int ckpt_arrays(CkptArr* a) const
    {
        const size_t n = (size_t)g.d.num;
        a[0] = { g.x,    3 * n };
        a[1] = { g.vx,   3 * n };
        a[2] = { g.rho,  n };
        a[3] = { g.SurF, n };
        a[4] = { g.A,    9 * n };
        a[5] = { g.vecn, 3 * n };
        if (sizeof(Real) < sizeof(double))
        {
            a[6] = { g.u,    3 * n };
            a[7] = { g.xc,   3 * n };
            a[8] = { g.rhoc, n };
        }
        return ckpt_array_count();
    }

    uint32_t ckpt_flags() const
    {
        const bool f32 = (sizeof(Real) < sizeof(double));
        return (f32 ? 1u : 0u)                       // data is float32
             | (f32 ? 2u : 0u)                       // compensated arrays present
             | (sph_use_soldbox ? 4u : 0u)           // solid wall clamp
             | (sph_use_static_solid ? 8u : 0u)      // A3 static solid list
             | (sph_use_fluid_list ? 16u : 0u)       // P1 fluid neighbour list
             | (sph_fast_fluid_loop ? 32u : 0u)      // A5 factored pass1
             | (sph_reflect_guard ? 64u : 0u);       // M8 reflection at z = 1
    }

    void ckpt_header(unsigned char* h, long long done, uint64_t payload) const
    {
        memset(h, 0, 128);
        memcpy(h, SPHCHK_MAGIC, 8);
        const uint32_t ver = SPHCHK_VERSION, hs = 128,
                       num = (uint32_t)g.d.num, ntotal = (uint32_t)g.d.ntotal,
                       solnum = (uint32_t)g.d.solnum,
                       narray = (uint32_t)ckpt_array_count(),
                       flags = ckpt_flags(), nfbs = (uint32_t)nfb;
        memcpy(h + 8,  &ver, 4);    memcpy(h + 12, &hs, 4);
        memcpy(h + 16, &num, 4);    memcpy(h + 20, &ntotal, 4);
        memcpy(h + 24, &solnum, 4); memcpy(h + 28, &narray, 4);
        memcpy(h + 32, &flags, 4);  memcpy(h + 36, &nfbs, 4);
        const uint64_t step = (uint64_t)done;
        const double hsml = SPH_HSML, dt = SPH_DT;
        memcpy(h + 40, &step, 8);
        memcpy(h + 48, &hsml, 8);   memcpy(h + 56, &dt, 8);
        memcpy(h + 64, &g_soldbox_z, 8);
        memcpy(h + 72, &g_soldbox_y0, 8);
        memcpy(h + 80, &g_soldbox_y1, 8);
        memcpy(h + 88, &deck_fp, 8);
        memcpy(h + 96, &payload, 8);
        // The remaining compile-time physics that is *not* derived from the deck.
        // Something has to catch a rebuilt solver whose constants moved: the
        // header contract is "refuse anything that would resume a different
        // problem", and c0/lamd/muyou change every stress in the run.
        const double c0 = SPH_C0, lamd = SPH_LAMD, muyou = SPH_MUYOU;
        memcpy(h + 104, &c0, 8); memcpy(h + 112, &lamd, 8); memcpy(h + 120, &muyou, 8);
    }

    // Write to <path>.tmp and rename over <path>: an interrupted or failed
    // write must leave the previous checkpoint intact -- that file is exactly
    // what the next resume will be reading after a crash.
    bool save_checkpoint(const char* path, long long done)
    {
        if (g.d.numP != g.d.num)
        {
            fprintf(stderr, "[ckpt] a decomposed run has ghosts; checkpointing is only "
                            "implemented for the single-GPU path (leave SPH_RANKS unset "
                            "and use the ordinary single-GPU path)\n");
            return false;
        }
        if (done == ckpt_done) return true;         // that state is already in the file
        char tmp[520];
        snprintf(tmp, sizeof(tmp), "%s.tmp", path);
        FILE* f = fopen(tmp, "wb");
        if (!f) { fprintf(stderr, "[ckpt] cannot write %s\n", tmp); return false; }

        CkptArr arr[9];
        const int na = ckpt_arrays(arr);
        uint64_t payload = 0;
        for (int a = 0; a < na; ++a) payload += (uint64_t)arr[a].len * sizeof(Real);

        unsigned char h[128];
        ckpt_header(h, done, payload);
        bool ok = (fwrite(h, 1, 128, f) == 128);

        const size_t chunk = 1u << 20;              // 8 MB of doubles
        std::vector<Real> buf(chunk);
        for (int a = 0; a < na && ok; ++a)
        {
            size_t left = arr[a].len;
            const Real* src = arr[a].p;
            while (left != 0)
            {
                const size_t m = (left < chunk) ? left : chunk;
                const cudaError_t e = cudaMemcpy(buf.data(), src, m * sizeof(Real),
                                                 cudaMemcpyDeviceToHost);
                if (e != cudaSuccess)
                {
                    fprintf(stderr, "[ckpt] device->host copy failed: %s\n", cudaGetErrorString(e));
                    ok = false; break;
                }
                if (fwrite(buf.data(), sizeof(Real), m, f) != m) { ok = false; break; }
                src += m; left -= m;
            }
        }
        const bool flushed = (fflush(f) == 0);
        fclose(f);
        if (!ok || !flushed)
        {
            fprintf(stderr, "[ckpt] write failed: %s\n", tmp);
            remove(tmp);
            return false;
        }
        // std::filesystem::rename is MoveFileEx(..., MOVEFILE_REPLACE_EXISTING) on
        // Windows: it replaces the old checkpoint in one step, so a crash cannot
        // leave the run with neither file (remove() + rename() could).
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec)
        {
            fprintf(stderr, "[ckpt] cannot rename %s -> %s: %s\n", tmp, path, ec.message().c_str());
            return false;
        }
        ckpt_done = done;
        printf("[ckpt] %s: %lld step%s, %.1f MB\n", path, done, (done == 1) ? "" : "s",
               (double)(128 + payload) / 1048576.0);
        fflush(stdout);                 // a checkpoint message must precede the next long run
        return true;
    }

    // Refuses anything that would silently resume a different problem.
    bool load_checkpoint(const char* path, long long& done)
    {
        if (g.d.numP != g.d.num)
        {
            fprintf(stderr, "[ckpt] a decomposed run has ghosts; checkpointing is only "
                            "implemented for the single-GPU path\n");
            return false;
        }
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "[ckpt] cannot open %s\n", path); return false; }

        auto bad = [&](const char* why) -> bool
        {
            fprintf(stderr, "[ckpt] %s: %s\n", path, why);
            fclose(f);
            return false;
        };

        unsigned char h[128];
        if (fread(h, 1, 128, f) != 128 || memcmp(h, SPHCHK_MAGIC, 8) != 0)
            return bad("not a checkpoint (bad magic or short header) -- a result*.sphbin "
                       "is not a restart state");

        uint32_t ver, hs, num, ntotal, solnum, narray, flags, nfbs;
        uint64_t step, payload, fp;
        double hsml, dt, bz, by0, by1, c0, lamd, muyou;
        memcpy(&ver, h + 8, 4);    memcpy(&hs, h + 12, 4);
        memcpy(&num, h + 16, 4);   memcpy(&ntotal, h + 20, 4);
        memcpy(&solnum, h + 24, 4); memcpy(&narray, h + 28, 4);
        memcpy(&flags, h + 32, 4); memcpy(&nfbs, h + 36, 4);
        memcpy(&step, h + 40, 8);
        memcpy(&hsml, h + 48, 8);  memcpy(&dt, h + 56, 8);
        memcpy(&bz, h + 64, 8);    memcpy(&by0, h + 72, 8); memcpy(&by1, h + 80, 8);
        memcpy(&fp, h + 88, 8);    memcpy(&payload, h + 96, 8);
        memcpy(&c0, h + 104, 8);   memcpy(&lamd, h + 112, 8); memcpy(&muyou, h + 120, 8);

        if (hs != 128 || (ver != 1 && ver != SPHCHK_VERSION))
            return bad("unsupported container version");
        if (num != (uint32_t)g.d.num || ntotal != (uint32_t)g.d.ntotal ||
            solnum != (uint32_t)g.d.solnum)
            return bad("particle counts differ from the deck");
        const bool f32 = (sizeof(Real) < sizeof(double));
        if (((flags & 1u) != 0) != f32 || ((flags & 2u) != 0) != f32)
            return bad(f32 ? "checkpoint holds fp64 state, this run is fp32"
                           : "checkpoint holds fp32 state, this run is fp64");
        if (narray != (uint32_t)ckpt_array_count())
            return bad("array count does not match this precision");
        if (hsml != SPH_HSML || dt != SPH_DT) return bad("hsml/dt differ from this build");
        if (c0 != SPH_C0 || lamd != SPH_LAMD || muyou != SPH_MUYOU)
            return bad("c0/lamd/muyou differ from this build (the solver was rebuilt with "
                       "different constitutive constants)");
        if (((flags & 4u) != 0) != (sph_use_soldbox != 0))
            return bad("solid wall clamp (SPH_SOLDBOX) differs from the run that wrote it");
        if (sph_use_soldbox && (bz != g_soldbox_z || by0 != g_soldbox_y0 || by1 != g_soldbox_y1))
            return bad("SPH_SOLDBOX bounds differ from the run that wrote it");
        if (((flags & 8u) != 0) != (sph_use_static_solid != 0) ||
            ((flags & 16u) != 0) != (sph_use_fluid_list != 0) ||
            ((flags & 32u) != 0) != (sph_fast_fluid_loop != 0))
            return bad("solver switches (SPH_SOLDDYN / SPH_FLUIDLIST / SPH_LITERALFLUID) "
                       "differ from the run that wrote it");
        if (((flags & 64u) != 0) != (sph_reflect_guard != 0))
        {
            // Version 1 was written by the pre-merge single-GPU build, which had
            // no M8 switch and always used 1.cpp's unconditional flip.  Treat a
            // v1 checkpoint as "flip" instead of guessing.
            if (ver == 1)
            {
                if (sph_reflect_guard != 0)
                    return bad("this is a version-1 checkpoint from the pre-merge build, "
                               "which reflects v_z unconditionally at z=1; resume it with "
                               "SPH_REFLECT=flip, or start this chain with the new build");
            }
            else
                return bad("the z = 1 reflection (SPH_REFLECT) differs from the run that wrote it");
        }
        if (fp != deck_fp)
        {
            // Print both values: a mismatch between an old checkpoint and a
            // rebuilt solver looks exactly like a wrong deck, and the two
            // numbers are the only way to tell them apart afterwards.
            fprintf(stderr, "[ckpt] %s: deck fingerprint mismatch -- the checkpoint says "
                            "0x%016llx, this deck hashes to 0x%016llx\n"
                            "       (a checkpoint belongs to the run that wrote it: same deck, "
                            "same precision, same switches, same build.  Re-run the first segment "
                            "with this build, or resume with the build that wrote the file.)\n",
                    path, (unsigned long long)fp, (unsigned long long)deck_fp);
            fclose(f);
            return false;
        }

        uint64_t expect = 0;
        CkptArr arr[9];
        const int na = ckpt_arrays(arr);
        for (int a = 0; a < na; ++a) expect += (uint64_t)arr[a].len * sizeof(Real);
        if (payload != expect) return bad("payload size does not match the layout");
        sph_fseek64(f, 0, SEEK_END);
        const long long sz = sph_ftell64(f);
        if ((uint64_t)sz != expect + 128)
            return bad("file size does not match the header (truncated write?)");
        sph_fseek64(f, 128, SEEK_SET);

        const size_t chunk = 1u << 20;
        std::vector<Real> buf(chunk);
        for (int a = 0; a < na; ++a)
        {
            size_t left = arr[a].len;
            Real* dst = arr[a].p;
            while (left != 0)
            {
                const size_t m = (left < chunk) ? left : chunk;
                if (fread(buf.data(), sizeof(Real), m, f) != m)
                    return bad("short read (truncated write?)");
                const cudaError_t e = cudaMemcpy(dst, buf.data(), m * sizeof(Real),
                                                 cudaMemcpyHostToDevice);
                if (e != cudaSuccess)
                {
                    fprintf(stderr, "[ckpt] host->device copy failed: %s\n", cudaGetErrorString(e));
                    fclose(f);
                    return false;
                }
                dst += m; left -= m;
            }
        }
        fclose(f);

        done = (long long)step;
        nfb  = (int)nfbs;
        printf("[ckpt] %s: step %lld, %s, %.1f MB\n", path, done,
               f32 ? "fp32" : "fp64", (double)sz / 1048576.0);
        fflush(stdout);
        if (nfb != 0)
            fprintf(stderr, "[ckpt] warning: %d particle(s) had a singular gradient-correction "
                            "matrix in the run that wrote this checkpoint\n", nfb);
        if (nfb != 0 && f32)
        {
            fprintf(stderr, "[ckpt] fp32: the F - I displacement form is invalid for this deck; "
                            "re-run in fp64\n");
            return false;
        }
        return true;
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
//  Per step a rank exchanges one E1 record block (fluid 8, solid 10 doubles) with
//  each of its up to 26 peers; see sph_kernels.cu for the layouts.  Packing is
//  a gather kernel into a device staging block, ONE cudaMemcpyAsync of the whole
//  block into a pinned host block, a barrier over all K threads, then a pinned
//  -> device copy and a scatter kernel into the ghost slots on the receiving
//  side.  The arena is double for both precisions, so a float value is carried
//  exactly and V2's byte identity holds in fp32 too.
//
//  The single-GPU path above (run_case) is the same Solver template but never
//  allocates a gid table, never sets numP != num and never enters this
//  namespace.
// ===========================================================================
#if (SPH_HAVE_FP64 || SPH_HAVE_FP32)
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

template <class Real>
struct Rank
{
    int id = 0, gpu = 0, ownNum = 0, numP = 0, ngs1 = 0, ngs2 = 0;
    std::vector<int>    gid;       // [numP] deck index of every local slot
    std::vector<double> rows;      // 14 * numP
    std::vector<DLink>  links;
    std::string         label;
    Solver<Real>        S;

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

template <class Real>
struct Job
{
    int K = 1, maxStep = 0;
    bool prof = false;
    int gnx = 1, gny = 1, gnz = 1;      // chosen block grid
    long long ghost_total = 0;          // sum of ghost slots over the ranks
    std::vector<Rank<Real>> ranks;
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
    std::vector<double> mig;            // [20/27*num] dynamic state, by gid
    int  repart = 0;                    // 0 = never
    bool regrid = false;                // re-search the candidate grids too
    long long repart_count = 0;
    double repart_s = 0.0;              // wall spent in repartition (rank 0)

    // ---- multi-GPU checkpoint (SPHD1CK) -------------------------------
    // ckState is the assembled global state by gid; every rank fills its owned
    // entries, rank 0 writes the file.  step0 is the absolute step of a resumed
    // run (0 for a fresh one), resume says to rebuild from the saved positions.
    std::vector<Real> ckState;
    uint64_t deck_fp = 0;
    long long step0 = 0;
    bool resume = false;

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
template <class Real>
static thread_local Job<Real>* tl_job = nullptr;
static thread_local int  tl_rank = -1;

template <class Real>
static double a3_fat_agreed(double mb)
{
    Job<Real>* J = tl_job<Real>;
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
        sph_fseek64(f, 0, SEEK_END);
        const long long szi = sph_ftell64(f);
        sph_fseek64(f, 0, SEEK_SET);
        if (szi < 0)
        {
            fprintf(stderr, "[D1] %s: cannot determine the file size\n", path);
            fclose(f); return false;
        }
        std::vector<char> buf((size_t)szi + 1);
        const size_t rd = fread(buf.data(), 1, (size_t)szi, f);
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

template <class Real>
static bool build_ranks(Job<Real>& J, const std::vector<double>& rows,
                        const std::vector<int>& ownerRank,
                        const std::vector<double>& box,
                        const double* pos = nullptr);

template <class Real>
static bool build_job(Job<Real>& J, const std::vector<double>& rows)
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
template <class Real>
static bool build_ranks(Job<Real>& J, const std::vector<double>& rows,
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
        Rank<Real>& R = J.ranks[r];
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
        Rank<Real>& R = J.ranks[r];
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
        Rank<Real>& R = J.ranks[r];
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
template <class Real>
static double* send_buf(Rank<Real>& R) { return R.hSendB[R.sendSlot]; }

template <class Real>
static void xch_pack(Job<Real>& J, Rank<Real>& R, int mode)
{
    if (J.K == 1) return;              // no peers: nothing to exchange, no barrier
    const double t0 = now_s();
    Solver<Real>& S = R.S;
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

template <class Real>
static void xch_unpack(Job<Real>& J, Rank<Real>& R, int mode)
{
    if (J.K == 1) return;
    const double t0 = now_s();
    Solver<Real>& S = R.S;

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
template <class Real>
static void xch(Job<Real>& J, Rank<Real>& R, int mode)
{
    R.sendSlot = R.xchSeq & 1;
    ++R.xchSeq;
    xch_pack(J, R, mode);
    xch_unpack(J, R, mode);
}

// C9 helpers (defined with the repartition code below)
template <class Real>
static void monitor_init(Rank<Real>& R);
template <class Real>
static void disp_collect(Job<Real>& J, Rank<Real>& R);

// one-off: the static fields of the ghost block.  A and vecn only a rank's
// owner can produce (they depend on that rank's own neighbour set), and
// k_prepare never runs over ghosts.
template <class Real>
static void xch_static(Job<Real>& J, Rank<Real>& R)
{
    if (J.K == 1) return;
    const double t0 = now_s();
    Solver<Real>& S = R.S;
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
template <class Real>
static void step_d1(Job<Real>& J, Rank<Real>& R, int itime)
{
    Solver<Real>& S = R.S;
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
template <class Real>
static void fetch_own(Job<Real>& J, Rank<Real>& R, typename Job<Real>::OutBuf& O)
{
    const SphDims& d = R.S.g.d;
    const int n = d.num, ns = d.solnum;
    const int N = d.numP;
    std::vector<Real> hx((size_t)3 * n), hp(n);
    CUDA_CHECK(cudaMemcpy(hx.data(),         R.S.g.x, (size_t)n * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hx.data() + n,     R.S.g.x + N, (size_t)n * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hx.data() + 2 * n, R.S.g.x + 2 * N, (size_t)n * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hp.data(), R.S.g.p, (size_t)n * sizeof(Real), cudaMemcpyDeviceToHost));
    std::vector<Real> pia;
    if (ns > 0)
    {
        pia.resize((size_t)SPH_PIA_STRIDE * ns);
        CUDA_CHECK(cudaMemcpy(pia.data(), R.S.g.Pia, pia.size() * sizeof(Real),
                              cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < n; ++i)
    {
        const int g = R.gid[i];
        O.gx[g] = (double)hx[i];
        O.gy[g] = (double)hx[n + i];
        O.gz[g] = (double)hx[2 * n + i];
        if (i < d.ntotal)      O.gp[g] = (double)hp[i];
        else if (i < d.nall)
        {
            const size_t k = (size_t)(i - d.ntotal) * SPH_PIA_STRIDE;
            O.gp[g] = (double)(pia[k] + pia[k + 4] + pia[k + 8]);
        }
        else                   O.gp[g] = 0.0;
    }
}

template <class Real>
static void write_global_result(Job<Real>& J, const typename Job<Real>::OutBuf& O, int idx)
{
    const char* ef32 = getenv("SPH_OUT_F32");
    const bool f32 = (ef32 != nullptr) && (atoi(ef32) != 0);
    char name[128];
    snprintf(name, sizeof(name), "result%d.sphbin", idx);
    FILE* f = fopen(name, "wb");
    if (!f) { fprintf(stderr, "[D1] cannot write %s\n", name); return; }

    unsigned char h[128] = {0};
    const char magic[8] = { 'S','P','H','B','I','N','\0','\0' };
    memcpy(h, magic, 8);
    const uint32_t ver = 1, hs = 128, num = (uint32_t)J.num,
                   ntotal = (uint32_t)J.ntotal, solnum = (uint32_t)J.solnum,
                   nfields = 4,
                   step = (uint32_t)((long long)idx * (g_out_cadence > 0 ? g_out_cadence : 1)),
                   flags = 2u | (f32 ? 4u : 0u);
    memcpy(h + 8, &ver, 4);      memcpy(h + 12, &hs, 4);
    memcpy(h + 16, &num, 4);     memcpy(h + 20, &ntotal, 4);
    memcpy(h + 24, &solnum, 4);  memcpy(h + 28, &nfields, 4);
    memcpy(h + 32, &step, 4);    memcpy(h + 36, &flags, 4);
    const double hsml = SPH_HSML, dt = SPH_DT;
    memcpy(h + 40, &hsml, 8); memcpy(h + 48, &dt, 8);
    memcpy(h + 56, J.grid, 48);

    bool ok = (fwrite(h, 1, 128, f) == 128);
    const size_t rec = 4;
    if (f32)
    {
        std::vector<float> buf(rec * 65536);
        size_t used = 0;
        for (int i = 0; i < J.num && ok; ++i)
        {
            if (used + rec > buf.size())
            {
                ok = (fwrite(buf.data(), sizeof(float), used, f) == used);
                used = 0;
            }
            buf[used++] = (float)O.gx[i]; buf[used++] = (float)O.gy[i];
            buf[used++] = (float)O.gz[i]; buf[used++] = (float)O.gp[i];
        }
        if (used != 0 && ok) ok = (fwrite(buf.data(), sizeof(float), used, f) == used);
    }
    else
    {
        std::vector<double> buf(rec * 65536);
        size_t used = 0;
        for (int i = 0; i < J.num && ok; ++i)
        {
            if (used + rec > buf.size())
            {
                ok = (fwrite(buf.data(), sizeof(double), used, f) == used);
                used = 0;
            }
            buf[used++] = O.gx[i]; buf[used++] = O.gy[i];
            buf[used++] = O.gz[i]; buf[used++] = O.gp[i];
        }
        if (used != 0 && ok) ok = (fwrite(buf.data(), sizeof(double), used, f) == used);
    }
    ok = (fflush(f) == 0) && ok;
    ok = (fclose(f) == 0) && ok;
    if (!ok)
    {
        fprintf(stderr, "[D1] short write on %s (disk full?) -- frame abandoned\n", name);
        remove(name);
    }
}

// ---------------------------------------------------------------------------
// Multi-GPU checkpoint (SPHD1CK).
//
// A decomposed run does not save its decomposition: the file holds the global
// state by deck id, and a resume rebuilds the ranks with build_job() from the
// deck and then runs one halo refresh from the restored *current* positions.
// That re-selects the ghost set for where the particles are now (a stale set
// would silently drop neighbours), while the physics and the gid summation order
// remain the ones a straight run uses.  The state is therefore:
//
//   x[3], vx[3], rho, SurF, A[9], vecn[3]          (20 values)
//   u[3], xc[3], rhoc                              (fp32 only, 7 more)
//
// exactly the cross-step state of the single-GPU checkpoint.  F/DFdt/Pia,
// SForce/PIJW, p/irho/SurFD and the cell list are step-local and are rebuilt.
// ---------------------------------------------------------------------------
template <class Real>
constexpr int d1_state_nvals() { return 20 + (sizeof(Real) < sizeof(double) ? 7 : 0); }

// The rank count is not stored in the decomposition (there is none), but it is
// in the header and a resume must use it: SPH_RANKS=auto reads it before it
// chooses devices.
static int read_d1_checkpoint_K(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ckpt] cannot open %s\n", path); return -1; }
    unsigned char h[160];
    if (fread(h, 1, sizeof(h), f) != sizeof(h) || memcmp(h, "SPHD1CK\0", 8) != 0)
    {
        fprintf(stderr, "[ckpt] %s: not a multi-GPU checkpoint\n", path);
        fclose(f); return -1;
    }
    fclose(f);
    uint32_t k = 0;
    memcpy(&k, h + 28, 4);
    return (int)k;
}

template <class Real>
static uint32_t d1_ckpt_flags()
{
    const bool f32 = (sizeof(Real) < sizeof(double));
    return (f32 ? 1u : 0u)                       // data is float32
         | (f32 ? 2u : 0u)                       // compensated arrays present
         | (sph_use_soldbox ? 4u : 0u)           // solid wall clamp
         | (sph_use_static_solid ? 8u : 0u)      // A3 static solid list
         | (sph_use_fluid_list ? 16u : 0u)       // P1 fluid neighbour list
         | (sph_fast_fluid_loop ? 32u : 0u)      // A5 factored pass1
         | (sph_reflect_guard ? 64u : 0u);       // M8 reflection at z = 1
}

// Read the header and the global state.  Returns false with a specific message
// instead of resuming a different problem.
template <class Real>
static bool load_d1_checkpoint(const char* path, int num, int ntotal, int solnum,
                               int K, uint64_t deck_fp,
                               std::vector<Real>& state, long long& step)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ckpt] cannot open %s\n", path); return false; }
    auto bad = [&](const char* why) -> bool
    {
        fprintf(stderr, "[ckpt] %s: %s\n", path, why);
        fclose(f);
        return false;
    };

    unsigned char h[160];
    if (fread(h, 1, sizeof(h), f) != sizeof(h) || memcmp(h, "SPHD1CK\0", 8) != 0)
        return bad("not a multi-GPU checkpoint (bad magic or short header)");
    uint32_t ver, hs, hn, hnt, hsold, hk, hnval, flags, gnx, gny, gnz;
    uint64_t hstep, hfp, payload;
    double hsml, dt, bz, by0, by1, c0, lamd, muyou;
    memcpy(&ver, h + 8, 4);   memcpy(&hs, h + 12, 4);
    memcpy(&hn, h + 16, 4);   memcpy(&hnt, h + 20, 4);
    memcpy(&hsold, h + 24, 4); memcpy(&hk, h + 28, 4);
    memcpy(&hnval, h + 32, 4); memcpy(&flags, h + 36, 4);
    memcpy(&hstep, h + 40, 8); memcpy(&hfp, h + 48, 8);
    memcpy(&hsml, h + 56, 8);  memcpy(&dt, h + 64, 8);
    memcpy(&bz, h + 72, 8);    memcpy(&by0, h + 80, 8); memcpy(&by1, h + 88, 8);
    memcpy(&c0, h + 96, 8);    memcpy(&lamd, h + 104, 8); memcpy(&muyou, h + 112, 8);
    memcpy(&payload, h + 120, 8);
    memcpy(&gnx, h + 128, 4);  memcpy(&gny, h + 132, 4); memcpy(&gnz, h + 136, 4);

    if (ver != 1 || hs != 160) return bad("unsupported container version");
    if (hn != (uint32_t)num || hnt != (uint32_t)ntotal || hsold != (uint32_t)solnum)
        return bad("particle counts differ from the deck");
    if (hk != (uint32_t)K)
    {
        fprintf(stderr, "[ckpt] %s: checkpoint was run with SPH_RANKS=%u, this run has %d; "
                        "the decomposition is not stored, so the rank count must match\n",
                path, hk, K);
        fclose(f);
        return false;
    }
    if (hnval != (uint32_t)d1_state_nvals<Real>())
        return bad("state layout does not match this precision");
    const bool f32 = (sizeof(Real) < sizeof(double));
    if (((flags & 1u) != 0) != f32 || ((flags & 2u) != 0) != f32)
        return bad(f32 ? "checkpoint holds fp64 state, this run is fp32"
                       : "checkpoint holds fp32 state, this run is fp64");
    if (((flags & 4u) != 0) != (sph_use_soldbox != 0))
        return bad("solid wall clamp (SPH_SOLDBOX) differs from the run that wrote it");
    if (sph_use_soldbox && (bz != g_soldbox_z || by0 != g_soldbox_y0 || by1 != g_soldbox_y1))
        return bad("SPH_SOLDBOX bounds differ from the run that wrote it");
    if (((flags & 8u) != 0) != (sph_use_static_solid != 0) ||
        ((flags & 16u) != 0) != (sph_use_fluid_list != 0) ||
        ((flags & 32u) != 0) != (sph_fast_fluid_loop != 0))
        return bad("solver switches (SPH_SOLDDYN / SPH_FLUIDLIST / SPH_LITERALFLUID) "
                   "differ from the run that wrote it");
    if (((flags & 64u) != 0) != (sph_reflect_guard != 0))
        return bad("the z = 1 reflection (SPH_REFLECT) differs from the run that wrote it");
    if (hsml != SPH_HSML || dt != SPH_DT) return bad("hsml/dt differ from this build");
    if (c0 != SPH_C0 || lamd != SPH_LAMD || muyou != SPH_MUYOU)
        return bad("c0/lamd/muyou differ from this build");
    if (hfp != deck_fp)
    {
        fprintf(stderr, "[ckpt] %s: deck fingerprint mismatch -- checkpoint 0x%016llx, "
                        "this deck 0x%016llx\n", path,
                (unsigned long long)hfp, (unsigned long long)deck_fp);
        fclose(f);
        return false;
    }
    const uint64_t expect = (uint64_t)hnval * (uint64_t)num * sizeof(Real);
    if (payload != expect) return bad("payload size does not match the layout");
    sph_fseek64(f, 0, SEEK_END);
    const long long sz = sph_ftell64(f);
    if ((uint64_t)sz != 160 + expect) return bad("file size does not match the header");
    sph_fseek64(f, 160, SEEK_SET);

    state.resize((size_t)hnval * (size_t)num);
    if (fread(state.data(), sizeof(Real), state.size(), f) != state.size())
        return bad("short read (truncated write?)");
    fclose(f);

    step = (long long)hstep;
    printf("[ckpt] %s: step %lld, %s, K=%u, %u particles, grid %ux%ux%u, %.1f MB\n",
           path, step, f32 ? "fp32" : "fp64", hk, hn, gnx, gny, gnz,
           (double)sz / 1048576.0);
    return true;
}

// Copy the owned part of the global checkpoint state into a rank's arrays.
// Only the owned block is uploaded: a resume runs a halo refresh next, so the
// initial ghost slots are never read before they are rebuilt.
template <class Real>
static void upload_owned_state(Job<Real>& J, Rank<Real>& R)
{
    const int n = R.ownNum;
    if (n <= 0) return;
    if (J.ckState.size() != (size_t)d1_state_nvals<Real>() * (size_t)J.num)
    {
        fprintf(stderr, "[ckpt] internal: checkpoint state has the wrong size\n");
        std::exit(1);
    }
    const size_t N = (size_t)J.num;
    const int stride = R.S.g.d.numP;
    auto up = [&](Real* dst, int dstStride, int k0, int comps)
    {
        std::vector<Real> buf((size_t)n);
        for (int c = 0; c < comps; ++c)
        {
            CUDA_CHECK(cudaMemcpy(buf.data(), dst + (size_t)c * dstStride,
                                  (size_t)n * sizeof(Real), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                buf[i] = J.ckState[(size_t)(k0 + c) * N + (size_t)R.gid[i]];
            CUDA_CHECK(cudaMemcpy(dst + (size_t)c * dstStride, buf.data(),
                                  (size_t)n * sizeof(Real), cudaMemcpyHostToDevice));
        }
    };
    up(R.S.g.x,    stride, 0, 3);
    up(R.S.g.vx,   stride, 3, 3);
    up(R.S.g.rho,  1,      6, 1);
    up(R.S.g.SurF, 1,      7, 1);
    up(R.S.g.A,    stride, 8, 9);
    up(R.S.g.vecn, stride, 17, 3);
    if constexpr (sizeof(Real) < sizeof(double))
    {
        up(R.S.g.u,    stride, 20, 3);
        up(R.S.g.xc,   stride, 23, 3);
        up(R.S.g.rhoc, 1,      26, 1);
    }
}

template <class Real>
static void save_d1_checkpoint(Job<Real>& J, Rank<Real>& R, long long step)
{
    if (g_ckpt_out == nullptr) return;
    const int nvals = d1_state_nvals<Real>();
    if (J.ckState.size() != (size_t)nvals * (size_t)J.num)
    {
        fprintf(stderr, "[ckpt] internal: checkpoint buffer has the wrong size\n");
        std::exit(1);
    }
    const int n = R.S.g.d.num, N = R.S.g.d.numP;
    const size_t NGlobal = (size_t)J.num;

    J.bar.wait();
    auto down = [&](const Real* src, int srcStride, int k0, int comps)
    {
        std::vector<Real> buf((size_t)n);
        for (int c = 0; c < comps; ++c)
        {
            CUDA_CHECK(cudaMemcpy(buf.data(), src + (size_t)c * srcStride,
                                  (size_t)n * sizeof(Real), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                J.ckState[(size_t)(k0 + c) * NGlobal + (size_t)R.gid[i]] = buf[i];
        }
    };
    down(R.S.g.x,    N, 0, 3);
    down(R.S.g.vx,   N, 3, 3);
    down(R.S.g.rho,  1, 6, 1);
    down(R.S.g.SurF, 1, 7, 1);
    down(R.S.g.A,    N, 8, 9);
    down(R.S.g.vecn, N, 17, 3);
    if constexpr (sizeof(Real) < sizeof(double))
    {
        down(R.S.g.u,    N, 20, 3);
        down(R.S.g.xc,   N, 23, 3);
        down(R.S.g.rhoc, 1, 26, 1);
    }
    J.bar.wait();

    if (R.id == 0)
    {
        char tmp[520];
        snprintf(tmp, sizeof(tmp), "%s.tmp", g_ckpt_out);
        FILE* f = fopen(tmp, "wb");
        bool ok = (f != nullptr);
        if (!ok) fprintf(stderr, "[ckpt] cannot write %s\n", tmp);
        if (ok)
        {
            unsigned char h[160] = {0};
            const char magic[8] = { 'S','P','H','D','1','C','K','\0' };
            memcpy(h, magic, 8);
            const uint32_t ver = 1, hs = 160, hn = (uint32_t)J.num,
                           hnt = (uint32_t)J.ntotal, hsold = (uint32_t)J.solnum,
                           hk = (uint32_t)J.K, hnval = (uint32_t)nvals,
                           flags = d1_ckpt_flags<Real>(),
                           gnx = (uint32_t)J.gnx, gny = (uint32_t)J.gny, gnz = (uint32_t)J.gnz;
            const uint64_t payload = (uint64_t)nvals * (uint64_t)J.num * sizeof(Real);
            memcpy(h + 8, &ver, 4);   memcpy(h + 12, &hs, 4);
            memcpy(h + 16, &hn, 4);   memcpy(h + 20, &hnt, 4);
            memcpy(h + 24, &hsold, 4); memcpy(h + 28, &hk, 4);
            memcpy(h + 32, &hnval, 4); memcpy(h + 36, &flags, 4);
            const uint64_t step64 = (uint64_t)step;
            memcpy(h + 40, &step64, 8); memcpy(h + 48, &J.deck_fp, 8);
            const double hsml = SPH_HSML, dt = SPH_DT;
            const double c0 = SPH_C0, lamd = SPH_LAMD, muyou = SPH_MUYOU;
            memcpy(h + 56, &hsml, 8);  memcpy(h + 64, &dt, 8);
            memcpy(h + 72, &g_soldbox_z, 8);
            memcpy(h + 80, &g_soldbox_y0, 8);
            memcpy(h + 88, &g_soldbox_y1, 8);
            memcpy(h + 96, &c0, 8);    memcpy(h + 104, &lamd, 8);
            memcpy(h + 112, &muyou, 8); memcpy(h + 120, &payload, 8);
            memcpy(h + 128, &gnx, 4);  memcpy(h + 132, &gny, 4); memcpy(h + 136, &gnz, 4);
            ok = (fwrite(h, 1, sizeof(h), f) == sizeof(h));
            if (ok) ok = (fwrite(J.ckState.data(), sizeof(Real), J.ckState.size(), f) == J.ckState.size());
            ok = (fflush(f) == 0) && ok;
            ok = (fclose(f) == 0) && ok;
            if (!ok)
            {
                fprintf(stderr, "[ckpt] write failed: %s\n", tmp);
                remove(tmp);
            }
            else
            {
                std::error_code ec;
                std::filesystem::rename(tmp, g_ckpt_out, ec);
                if (ec)
                {
                    fprintf(stderr, "[ckpt] cannot rename %s -> %s: %s\n", tmp, g_ckpt_out,
                            ec.message().c_str());
                }
                else
                {
                    printf("[ckpt] %s: %lld steps, K=%d, %s, %.1f MB\n", g_ckpt_out, step,
                           J.K, (sizeof(Real) < sizeof(double)) ? "fp32" : "fp64",
                           (double)(160 + payload) / 1048576.0);
                    fflush(stdout);
                }
            }
        }
    }
    J.bar.wait();
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
template <class Real>
constexpr int mig_n = 20 + (sizeof(Real) < sizeof(double) ? 7 : 0);   // + u3, xc3, rhoc

template <class Real>
static void build_send_lists(Job<Real>& J, Rank<Real>& R);   // defined below

template <class Real>
static void xch_free(Rank<Real>& R)
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
template <class Real>
static void monitor_init(Rank<Real>& R)
{
    if (R.dXBuild)  { cudaFree(R.dXBuild);  R.dXBuild = nullptr; }
    if (R.dMaxDisp) { cudaFree(R.dMaxDisp); R.dMaxDisp = nullptr; }
    const int n = R.S.g.d.num, N = R.S.g.d.numP;
    if (n <= 0) return;
    CUDA_CHECK(cudaMalloc((void**)&R.dXBuild, (size_t)3 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&R.dMaxDisp, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(R.dMaxDisp, 0, sizeof(unsigned long long)));
    // The snapshot is double even for Real = float: k_disp_max widens the live
    // coordinate to double before subtracting, so both precisions measure the
    // same metric.  One host round trip, once per (re)build.
    std::vector<Real> hx((size_t)n);
    std::vector<double> dx((size_t)n);
    for (int c = 0; c < 3; ++c)
    {
        CUDA_CHECK(cudaMemcpy(hx.data(), R.S.g.x + (size_t)c * N,
                              (size_t)n * sizeof(Real), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) dx[i] = (double)hx[i];
        CUDA_CHECK(cudaMemcpy(R.dXBuild + (size_t)c * n, dx.data(),
                              (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    }
}

// P17: the displacement is reduced into the tail of the E1 staging block (xch_pack
// does that), and after E1's barrier every rank reads the maximum over the peers'
// blocks -- the same relay it already performs, no readback of its own.
template <class Real>
static void disp_collect(Job<Real>& J, Rank<Real>& R)
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

template <class Real>
static void mig_snapshot(Job<Real>& J, Rank<Real>& R)
{
    const SphDims& d = R.S.g.d;
    const int n = d.num, N = d.numP;
    std::vector<const Real*> src;
    src.reserve((size_t)mig_n<Real>);
    src.push_back(R.S.g.x); src.push_back(R.S.g.x + N); src.push_back(R.S.g.x + 2 * N);
    src.push_back(R.S.g.vx); src.push_back(R.S.g.vx + N); src.push_back(R.S.g.vx + 2 * N);
    src.push_back(R.S.g.rho); src.push_back(R.S.g.SurF);
    for (int c = 0; c < 9; ++c) src.push_back(R.S.g.A + (size_t)c * N);
    src.push_back(R.S.g.vecn); src.push_back(R.S.g.vecn + N); src.push_back(R.S.g.vecn + 2 * N);
    if constexpr (sizeof(Real) < sizeof(double))
    {
        // fp32: x of a solid is X0 + u, and the Kahan compensations are the only
        // record of the sub-ulp motion.  Losing them in a repartition would move
        // the solid back by the accumulated rounding.
        src.push_back(R.S.g.u); src.push_back(R.S.g.u + N); src.push_back(R.S.g.u + 2 * N);
        src.push_back(R.S.g.xc); src.push_back(R.S.g.xc + N); src.push_back(R.S.g.xc + 2 * N);
        src.push_back(R.S.g.rhoc);
    }
    std::vector<Real> buf((size_t)n);
    for (int k = 0; k < (int)src.size(); ++k)
    {
        CUDA_CHECK(cudaMemcpy(buf.data(), src[(size_t)k], (size_t)n * sizeof(Real),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i)
            J.mig[(size_t)R.gid[i] * mig_n<Real> + k] = (double)buf[i];
    }
}

template <class Real>
static void mig_restore(Job<Real>& J, Rank<Real>& R)
{
    const SphDims& d = R.S.g.d;
    const int n = d.num, N = d.numP;
    std::vector<Real*> dst;
    dst.reserve((size_t)mig_n<Real>);
    dst.push_back(R.S.g.x); dst.push_back(R.S.g.x + N); dst.push_back(R.S.g.x + 2 * N);
    dst.push_back(R.S.g.vx); dst.push_back(R.S.g.vx + N); dst.push_back(R.S.g.vx + 2 * N);
    dst.push_back(R.S.g.rho); dst.push_back(R.S.g.SurF);
    for (int c = 0; c < 9; ++c) dst.push_back(R.S.g.A + (size_t)c * N);
    dst.push_back(R.S.g.vecn); dst.push_back(R.S.g.vecn + N); dst.push_back(R.S.g.vecn + 2 * N);
    if constexpr (sizeof(Real) < sizeof(double))
    {
        dst.push_back(R.S.g.u); dst.push_back(R.S.g.u + N); dst.push_back(R.S.g.u + 2 * N);
        dst.push_back(R.S.g.xc); dst.push_back(R.S.g.xc + N); dst.push_back(R.S.g.xc + 2 * N);
        dst.push_back(R.S.g.rhoc);
    }
    std::vector<Real> buf((size_t)n);
    for (int k = 0; k < (int)dst.size(); ++k)
    {
        for (int i = 0; i < n; ++i)
            buf[i] = (Real)J.mig[(size_t)R.gid[i] * mig_n<Real> + k];
        CUDA_CHECK(cudaMemcpy(dst[(size_t)k], buf.data(), (size_t)n * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }
}

template <class Real>
static void repartition(Job<Real>& J, Rank<Real>& R, int itime, const char* why, bool recut)
{
    const double t0 = now_s();
    const int n = R.S.g.d.num, N = R.S.g.d.numP;
    if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: repartition(%s, recut=%d) start\n", R.id, why, (int)recut);

    // 1) publish my positions and my dynamic state, both indexed by gid
    {
        std::vector<Real> buf((size_t)n);
        for (int c = 0; c < 3; ++c)
        {
            CUDA_CHECK(cudaMemcpy(buf.data(), R.S.g.x + (size_t)c * N,
                                  (size_t)n * sizeof(Real), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i)
                J.pos[(size_t)R.gid[i] * 3 + c] = (double)buf[i];
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
template <class Real>
static void build_send_lists(Job<Real>& J, Rank<Real>& R)
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
template <class Real>
static void rank_main(Job<Real>* Jp, int r)
{
    Job<Real>& J = *Jp;
    Rank<Real>& R = J.ranks[r];
    tl_job<Real> = &J;
    tl_rank = r;
    CUDA_CHECK(cudaSetDevice(R.gpu));
    // Each device has its own copy of the __device__ clamp symbols.
    sph_apply_soldbox();

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

    if (J.resume)
    {
        // The checkpoint holds global state by deck id, not a decomposition.
        // Load the owned rows into the initial build_job() ranks, then refresh
        // the halo from the restored current positions: that re-selects every
        // ghost the run actually needs now, and the gid summation order still
        // matches a straight run.
        upload_owned_state(J, R);
        repartition(J, R, (int)J.step0, "resume", false);
        J.bar.wait();
        if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: resume rebuild ok\n", r);
    }
    else
    {
        launch_copy_xf(R.S.g, 0);
        if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: deck loaded, numP=%d\n", r, R.S.g.d.numP);
        monitor_init(R);                  // C9: the reference for the displacement trigger
        R.S.build_cells();
        if (getenv("SPH_VERIFY")) R.S.verify_cells();
        R.S.init_reference_state();
        CUDA_CHECK(cudaDeviceSynchronize());
        if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: reference state ok\n", r);
        J.bar.wait();

        // the ghost block's static fields, then the solid pack/list that needs them
        xch_static(J, R);
        launch_build_solid_pack(R.S.g, 0);
        if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: static exchange + pack ok\n", r);
        if (sph_use_static_solid) R.S.build_solid_lists();
        if (getenv("SPH_VERIFY")) R.S.verify_cells();
        if (getenv("SPH_D1_DBG")) fprintf(stderr, "[dbg] rank %d: solid lists ok (npair %d, nsolC %d)\n", r, R.S.g.npair, R.S.g.d.nsolC);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    {
        const int nfb = gra_fallback_count();
        if (nfb != 0)
        {
            std::lock_guard<std::mutex> lk(J.print_m);
            fprintf(stderr, "[D1] rank %d: warning: %d particle(s) fell back to A = I\n", r, nfb);
            if constexpr (sizeof(Real) < sizeof(double))
            {
                fprintf(stderr, "[D1] rank %d: fp32: the F - I displacement form is invalid "
                                "for this deck; re-run with --fp64\n", r);
                std::exit(4);
            }
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
        const long long step = J.step0 + (long long)itime;
        if (g_ckpt_out != nullptr && g_ckpt_every > 0 && step % g_ckpt_every == 0)
            save_d1_checkpoint(J, R, step);

        const auto s0 = std::chrono::steady_clock::now();
        step_d1(J, R, (int)step);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto s1 = std::chrono::steady_clock::now();
        R.step_s += std::chrono::duration<double>(s1 - s0).count();

        if (g_out_cadence > 0 && step % g_out_cadence == 0)
        {
            // P14: hand the file to a writer thread and keep stepping.  Two stages
            // are enough: a stage is only reused two results later, and rank 0
            // waits for that writer before the barrier that releases it.
            // C20: derived, not stored -- every rank computes the same value
            const int s = (int)((step / g_out_cadence) & 1);
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
                const int idx = (int)(step / g_out_cadence);
                typename Job<Real>::OutBuf& O = J.out[s];
                J.outFut[s] = std::async(std::launch::async,
                                         [&J, &O, idx] { write_global_result(J, O, idx); });
                J.outBusy[s] = true;
            }
        }

        // repartition at a step boundary, after the result write, so the file
        // still carries the decomposition it was fetched with
        bool due = false;
        if (J.repart > 0 && step > 0 && (step % J.repart) == 0) due = true;
        // C9: the skin has been spent -- rebuild before a pair can be lost.  The
        // decision is identical on every rank (same maximum, same threshold).
        const bool skin_due = R.rebuild_due;
        if ((due || skin_due) && step < J.step0 + J.maxStep - 1)
        {
            char why[128];
            if (skin_due)
                snprintf(why, sizeof(why), "skin spent: %.3f mm moved, skin %.2f mm",
                         1000.0 * R.last_disp, 1000.0 * J.skin);
            else
                snprintf(why, sizeof(why), "SPH_D1_REPART");
            repartition(J, R, (int)step, why, due);
        }
    }
    if (g_ckpt_out != nullptr)
        save_d1_checkpoint(J, R, J.step0 + (long long)J.maxStep);
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

template <class Real>
static int run(const char* deck, int maxStep, int K, const std::vector<int>& gpus);

// L1: rank the devices, choose K, then run the ordinary driver on that set.
template <class Real>
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
    int K = 0;
    if (g_ckpt_in != nullptr)
    {
        K = read_d1_checkpoint_K(g_ckpt_in);
        if (K < 1 || K > (int)devs.size())
        {
            fprintf(stderr, "[D1] checkpoint needs %d rank(s), %zu device(s) are usable; "
                            "set SPH_RANKS=K explicitly or free the devices\n", K, devs.size());
            return 2;
        }
        std::printf("[D1] resume: keeping K=%d from %s\n", K, g_ckpt_in);
    }
    else
    {
        K = choose_auto_k(D, (int)devs.size(), skin, pred);
    }
    std::vector<int> gpus;
    for (int i = 0; i < K; ++i) gpus.push_back(devs[(size_t)i].id);
    return run<Real>(deck, maxStep, K, gpus);
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------
template <class Real>
static int run(const char* deck, int maxStep, int K, const std::vector<int>& gpus)
{
    Job<Real> J;
    J.K = K;
    J.bar.K = K;
    J.maxStep = maxStep;
    J.deck = deck;
    J.prof = (getenv("SPH_D1PROF") != nullptr);
    // C12: from here on build_solid_lists() decides the A6 layout collectively
    sph_a3_fat_hook = &a3_fat_agreed<Real>;
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

    if (g_ckpt_wanted) J.deck_fp = sph_deck_fingerprint(rows, J.num);
    if (g_ckpt_in != nullptr)
    {
        if (!load_d1_checkpoint<Real>(g_ckpt_in, J.num, J.ntotal, J.solnum, K,
                                      J.deck_fp, J.ckState, J.step0))
            return 5;
        J.resume = true;
        if (J.step0 < 0) { fprintf(stderr, "[ckpt] negative step\n"); return 5; }
    }

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
    if (J.repart > 0 || J.skin > 0.0 || J.resume)
    {
        J.pos.assign((size_t)3 * J.num, 0.0);
        J.mig.assign((size_t)mig_n<Real> * J.num, 0.0);
    }
    if (g_ckpt_out != nullptr && J.ckState.empty())
        J.ckState.assign((size_t)d1_state_nvals<Real>() * (size_t)J.num, Real(0.0));
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
        const Rank<Real>& R = J.ranks[r];
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
    for (int r = 0; r < K; ++r) th.emplace_back(rank_main<Real>, &J, r);
    for (auto& t : th) t.join();

    double wall = 0.0;
    for (int r = 0; r < K; ++r) wall = std::max(wall, J.ranks[r].t_end - J.ranks[r].t_start);
    const int steps = maxStep > 0 ? maxStep : 1;
    for (int r = 0; r < K; ++r)
    {
        const Rank<Real>& R = J.ranks[r];
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
            const Rank<Real>& R = J.ranks[r];
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
#endif  // SPH_HAVE_FP64 || SPH_HAVE_FP32

// ---------------------------------------------------------------------------
// driver for one precision
// ---------------------------------------------------------------------------
template <class Real>
static int run_case(const char* input_path, int iMaxStep, int profile, const char* prec_name,
                    const char* ckpt_out, const char* ckpt_in, long long step0)
{
    Solver<Real> S;
    S.dump_step = (getenv("SPH_DUMPSTEP") != nullptr);
    S.dump_a    = (getenv("SPH_DUMPA") != nullptr);

    auto t0 = std::chrono::steady_clock::now();
    if (!S.load_deck(input_path)) return 1;   // parse + size + allocate + upload
    launch_copy_xf(S.g, 0);                    // A4: float copy before the first search

    S.build_cells();
    if (getenv("SPH_VERIFY")) S.verify_cells();

    long long itime0 = step0;
    if (ckpt_in != nullptr)
    {
        // A resume replaces both init-only arrays (A, vecn) with the state of
        // the run that wrote the checkpoint, so init_reference_state() must not
        // run: it would search in the restored, already deformed positions.
        long long done = 0;
        if (!S.load_checkpoint(ckpt_in, done)) return 5;
        itime0 = done;
        launch_copy_xf(S.g, 0);
        S.build_cells();
        if (getenv("SPH_LOADA"))
            fprintf(stderr, "[ckpt] SPH_LOADA ignored: A comes from the checkpoint\n");
        if (g_out_cadence > 0)
            printf("[ckpt] resumed at step %lld; the next result frame is written after step %lld "
                   "(SPH_MAXSTEP counts this run's steps only)\n",
                   itime0, ((itime0 + g_out_cadence - 1) / g_out_cadence) * g_out_cadence);
        else
            printf("[ckpt] resumed at step %lld\n", itime0);
        fflush(stdout);
    }
    else
    {
        S.init_reference_state();
        CUDA_CHECK(cudaDeviceSynchronize());
        // C8: the fp32 solid kernels accumulate F - I, which is only the same tensor
        // as F when every A is a true inverse (a * = I).  A particle whose A fell
        // back to the identity breaks that, so refuse fp32 rather than run wrong.
        S.nfb = gra_fallback_count();
        if (S.nfb != 0)
        {
            fprintf(stderr, "[A] warning: %d particle(s) have a singular gradient-correction "
                            "matrix and fell back to A = I (as 1.cpp does)\n", S.nfb);
            if constexpr (sizeof(Real) < sizeof(double))
            {
                fprintf(stderr, "[A] fp32: the F - I displacement form is invalid for this deck; "
                                "re-run in fp64\n");
                return 4;
            }
        }
        if (const char* pa = getenv("SPH_LOADA")) S.load_A_from_dump(pa);
    }
    launch_build_solid_pack(S.g, 0);            // after SPH_LOADA / the checkpoint: the pack holds a copy of A
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
    double pw[6] = {0, 0, 0, 0, 0, 0};      // per-window snapshot for [profwin]
    cudaEvent_t ev[8];
    if (profile) for (int k = 0; k < 8; ++k) CUDA_CHECK(cudaEventCreate(&ev[k]));

    // SPH_STEPTIME: print the wall time of every window of 500 steps.  A long run
    // gets slower as the flow develops; this shows the drift directly in the log.
    const bool steptime = (getenv("SPH_STEPTIME") != nullptr);
    double win_seconds = 0.0;

    for (long long itime = itime0; itime < itime0 + (long long)iMaxStep; ++itime)
    {
        // The checkpoint is taken *here*, before the step, so its `step` count
        // is the number of completed steps and resuming replays exactly the
        // iterations this run would have executed next -- same result frame
        // names, same printed diagnostics.
        if (ckpt_out != nullptr && g_ckpt_every > 0 && itime % g_ckpt_every == 0)
        {
            if (!S.save_checkpoint(ckpt_out, itime)) return 6;
        }

        if (itime % 500 == 0)
        {
            // the list built inside the previous step predates that step's position
            // update; rebuild it so the printed count belongs to the current positions
            if (itime > 0) S.build_cells();
            CUDA_CHECK(cudaMemset(S.d_niac, 0, sizeof(int)));
            launch_count_pairs(S.g, S.d_niac, 0);
            int niac = 0;
            CUDA_CHECK(cudaMemcpy(&niac, S.d_niac, sizeof(int), cudaMemcpyDeviceToHost));
            printf("T= %lld\tpairs=%d\n", itime / 500, niac / 2);
            if (steptime && itime > 0 && !profile)   // profile mode prints [profwin] instead
            {
                printf("[steptime] step %lld\t%.3f ms/step\n", itime, 1000.0 * win_seconds / 500.0);
                win_seconds = 0.0;
            }
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
            if (steptime && itime > 0 && itime % 500 == 0)
            {
                printf("[profwin] step %lld  prepare %.3f  cell %.3f  pass1 %.3f  "
                       "cons %.3f  pass2 %.3f  update %.3f  (ms/step)\n", itime,
                       1000.0 * (prof[0] - pw[0]) / 500.0, 1000.0 * (prof[1] - pw[1]) / 500.0,
                       1000.0 * (prof[2] - pw[2]) / 500.0, 1000.0 * (prof[3] - pw[3]) / 500.0,
                       1000.0 * (prof[4] - pw[4]) / 500.0, 1000.0 * (prof[5] - pw[5]) / 500.0);
                for (int k = 0; k < 6; ++k) pw[k] = prof[k];
            }
        }
        else
        {
            auto s0 = std::chrono::steady_clock::now();
            S.step((int)itime);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto s1 = std::chrono::steady_clock::now();
            step_seconds += std::chrono::duration<double>(s1 - s0).count();
            win_seconds  += std::chrono::duration<double>(s1 - s0).count();
        }

        if (g_out_cadence > 0 && itime % g_out_cadence == 0)
        {
            S.fetch_state();
            if (!S.write_result(itime)) return 7;
        }
    }

    // A run that stops (or is stopped between two periodic writes) still leaves
    // a checkpoint that is exactly its final state.
    if (ckpt_out != nullptr)
    {
        if (!S.save_checkpoint(ckpt_out, itime0 + (long long)iMaxStep)) return 6;
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
           "                [--resume=<checkpoint>] [--ckpt=<checkpoint>] [--step0=<n>]\n"
           "  env: SPH_MAXSTEP, SPH_OUTCADENCE, SPH_OUT_F32, SPH_SOLDBOX, SPH_PRECISION,\n"
           "       SPH_PROFILE, SPH_STEPTIME (per-500-step ms/step), SPH_BENCH,\n"
           "       SPH_CKPT (write), SPH_CKPT_IN (resume), SPH_CKPT_EVERY (steps,\n"
           "       0 = at the end), SPH_STEP0 (absolute step number of a fresh run),\n"
           "       SPH_SORTEDTHREADS=0/1 (single-GPU fluid-kernel thread order)\n"
           "       SPH_RANKS=K / SPH_RANKS=auto, SPH_RANKS_GPUS=0,1,... (multi-GPU)\n"
           "  a resume needs the same deck, precision and solver switches as the run that wrote\n"
           "  the checkpoint; the checkpoint supplies x, vx, rho, SurF, A, vecn (+ u, xc, rhoc in\n"
           "  fp32).  The single-GPU container is SPHCHK; a decomposed run writes SPHD1CK\n"
           "  (global state by deck id, no decomposition) and resumes by rebuilding from the\n"
           "  restored positions.  See cuda/docs/CHECKPOINT.md.\n");
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
    // Unbuffered: a long run that is killed (or that hangs) must still show how
    // far it got.  The per-step path prints at most once every 500 steps.
    setvbuf(stdout, nullptr, _IONBF, 0);

    const char* input = "input.dat";
    int prec = -1;
    const char* ckpt_out = nullptr;
    const char* ckpt_in = nullptr;
    long long step0 = 0;

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
        else if (!strncmp(a, "--resume=", 9)) ckpt_in = a + 9;
        else if (!strncmp(a, "--ckpt=", 7)) ckpt_out = a + 7;
        else if (!strncmp(a, "--step0=", 8)) step0 = atoll(a + 8);
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
    if (const char* e = getenv("SPH_MAXSTEP"))
    {
        iMaxStep = atoi(e);
        if (iMaxStep < 0) { fprintf(stderr, "SPH_MAXSTEP must be >= 0 (got %s)\n", e); return 2; }
    }
    // 0 is a meaningful value: no result frames at all.
    if (const char* e = getenv("SPH_OUTCADENCE"))
    {
        const int c = atoi(e);
        if (c < 0) { fprintf(stderr, "SPH_OUTCADENCE must be >= 0 (got %s)\n", e); return 2; }
        g_out_cadence = c;
    }
    const int profile = (getenv("SPH_PROFILE") != nullptr) ? 1 : 0;
    if (const char* e = getenv("SPH_SOLDDYN")) sph_use_static_solid = (atoi(e) == 0) ? 1 : 0;
    if (const char* e = getenv("SPH_LITERALFLUID")) sph_fast_fluid_loop = (atoi(e) == 0) ? 1 : 0;
    if (const char* e = getenv("SPH_FLUIDLIST")) sph_use_fluid_list = (atoi(e) == 0) ? 0 : 1;
    if (const char* e = getenv("SPH_SORTEDTHREADS")) sph_sorted_threads = (atoi(e) == 0) ? 0 : 1;
    // A6: which solid-list layout to use.  -1 = auto (lean when the fat list
    // would dominate VRAM), 0 = fat (sG/sB/sD stored, fastest), 1 = lean.
    // SPH_A3_FULL=0/1 forces lean/fat; SPH_A3_LIMIT_MB sets the auto threshold.
    sph_a3_lean = -1;
    if (const char* e = getenv("SPH_A3_FULL")) sph_a3_lean = (atoi(e) == 0) ? 1 : 0;
    // M8: the z = 1 reflection.  'flip' is 1.cpp's unconditional form.
    if (const char* e = getenv("SPH_REFLECT")) sph_reflect_guard = strcmp(e, "flip") != 0;

    // ---- checkpointing (single-GPU path; see cuda/docs/CHECKPOINT.md) ----
    if (ckpt_out == nullptr) ckpt_out = getenv("SPH_CKPT");
    if (ckpt_in  == nullptr) ckpt_in  = getenv("SPH_CKPT_IN");
    if (ckpt_out != nullptr && *ckpt_out == '\0') ckpt_out = nullptr;
    if (ckpt_in  != nullptr && *ckpt_in  == '\0') ckpt_in  = nullptr;
    if (const char* e = getenv("SPH_CKPT_EVERY")) { int c = atoi(e); if (c > 0) g_ckpt_every = c; }
    if (const char* e = getenv("SPH_STEP0")) step0 = atoll(e);
    if (step0 < 0) { fprintf(stderr, "SPH_STEP0 must be >= 0\n"); return 2; }
    if (g_ckpt_every > 0 && ckpt_out == nullptr)
    {
        fprintf(stderr, "SPH_CKPT_EVERY=%d needs SPH_CKPT=<file>\n", g_ckpt_every);
        return 2;
    }
    g_ckpt_wanted = (ckpt_out != nullptr) || (ckpt_in != nullptr);
    g_ckpt_out = ckpt_out;
    g_ckpt_in  = ckpt_in;
    if (ckpt_in != nullptr)
    {
        // the checkpoint carries the step count; --step0 only names a fresh run
        if (step0 != 0) fprintf(stderr, "[ckpt] --step0 ignored: the checkpoint carries the step count\n");
        step0 = 0;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    {
        // solid wall box clamp, see sph.h.  Defaults: the 3.7M deck's geometry
        // (1.cpp of that case: x[2]<=0.01 || x[1]<=0.00 || x[1]>=3.31).  The
        // 1.2M deck's box is "0.01,0.00,1.31"; the 237k deck "0.01,0.01,0.49".
        double bz = 0.01, by0 = 0.00, by1 = 3.31;
        int bon = 1;
        if (const char* e = getenv("SPH_SOLDBOX"))
        {
            if (!strcmp(e, "off") || !strcmp(e, "0")) bon = 0;
            else if (sscanf(e, "%lf,%lf,%lf", &bz, &by0, &by1) != 3)
            {
                fprintf(stderr, "SPH_SOLDBOX expects \"z,y0,y1\" or \"off\"\n");
                return 2;
            }
        }
        g_soldbox_z = bz; g_soldbox_y0 = by0; g_soldbox_y1 = by1;
        sph_set_soldbox(bon, bz, by0, by1);
        if (bon) printf("[soldbox] solid wall clamp on: z<=%.3f | y<=%.3f | y>=%.3f\n", bz, by0, by1);
        else     printf("[soldbox] solid wall clamp OFF\n");
    }

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
#if SPH_HAVE_FP32
            if (want_f32) return d1::run_auto<float>(input, iMaxStep);
#endif
#if SPH_HAVE_FP64
            return d1::run_auto<double>(input, iMaxStep);
#else
            fprintf(stderr, "this build has no fp64 path (SPH_ENABLE_FP64=OFF)\n");
            return 2;
#endif
        }
        if (K < 1) { fprintf(stderr, "SPH_RANKS must be >= 1 or 'auto' (got '%s')\n", e); return 2; }
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
        if (profile)
            fprintf(stderr, "[D1] note: SPH_PROFILE is not implemented on the decomposed path\n");
#if SPH_HAVE_FP32
        if (want_f32) return d1::run<float>(input, iMaxStep, K, gpus);
#endif
#if SPH_HAVE_FP64
        return d1::run<double>(input, iMaxStep, K, gpus);
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
    if (want_f32) return run_case<float>(input, iMaxStep, profile, "fp32",
                                         ckpt_out, ckpt_in, step0);
#endif
#if SPH_HAVE_FP64
    return run_case<double>(input, iMaxStep, profile, "fp64",
                            ckpt_out, ckpt_in, step0);
#else
    return 0;
#endif
}
























