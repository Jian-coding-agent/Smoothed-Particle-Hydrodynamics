"""Patch the scratch copy of cuda/src with the review experiments (D, E, F, G)."""
import pathlib, sys

src = pathlib.Path(sys.argv[1])

def sub(path, old, new, count=1):
    t = path.read_text()
    n = t.count(old)
    if n != count:
        raise SystemExit(f"{path.name}: expected {count} match(es) for {old[:60]!r}, found {n}")
    path.write_text(t.replace(old, new))

h = src / "sph.h"
sub(h, "    Real* rhoc;     // [NUM]   running compensation of the density sum\n",
       "    Real* rhoc;     // [NUM]   running compensation of the density sum\n"
       "    // review experiments\n"
       "    int   ellMax; int* eCnt; int* eQ; Real* eG; Real* eB; Real* eD;\n"
       "    int*  fnbr; int* fcnt;\n")
sub(h, "extern int sph_fast_fluid_loop;\n",
       "extern int sph_fast_fluid_loop;\n"
       "extern int sph_use_ell;\n"
       "#ifndef P1_MINB\n#define P1_MINB 4\n#endif\n"
       "#if P1_MINB > 0\n#define P1_LB __launch_bounds__(256, P1_MINB)\n#else\n#define P1_LB __launch_bounds__(256)\n#endif\n"
       "template <class Real> void launch_fluid_nbr_build(SphPtrsT<Real> s, int maxn, int* d_over, cudaStream_t st);\n"
       "template <class Real> void launch_pass1_fluid_list(SphPtrsT<Real> s, cudaStream_t st);\n"
       "template <class Real> void launch_pass1_solid_ell(SphPtrsT<Real> s, cudaStream_t st);\n"
       "template <class Real> void launch_pass2_solid_ell(SphPtrsT<Real> s, cudaStream_t st);\n")

k = src / "sph_kernels.cu"
sub(k, "__global__ void __launch_bounds__(256, 4) k_pass1_fluid_fast(", "__global__ void P1_LB k_pass1_fluid_fast(")
sub(k, "    if (sph_use_static_solid && s.npair > 0)\n        k_pass1_solid_static<Real>",
       "    if (sph_use_ell && s.ellMax > 0) { launch_pass1_solid_ell(s, st); return; }\n"
       "    if (sph_use_static_solid && s.npair > 0)\n        k_pass1_solid_static<Real>")
sub(k, "    if (sph_use_static_solid && s.npair > 0)\n        k_pass2_solid_static<Real>",
       "    if (sph_use_ell && s.ellMax > 0) { launch_pass2_solid_ell(s, st); return; }\n"
       "    if (sph_use_static_solid && s.npair > 0)\n        k_pass2_solid_static<Real>")
k.write_text(k.read_text() + '\n#include "../review_ext_kernels.inc"\n')

c = src / "sph_host.cu"
sub(c, "    // ---------------- one time step ----------------\n", r'''
    // ---------------- review experiments ----------------
    static const int FMAXN = 224;
    int* d_over = nullptr;
    void alloc_fluid_list()
    {
        g.fnbr = (int*)dev_alloc_raw((size_t)FMAXN * SPH_NTOTAL * sizeof(int));
        g.fcnt = (int*)dev_alloc((size_t)SPH_NTOTAL * sizeof(int));
        d_over = (int*)dev_alloc(sizeof(int));
        printf("[review] fluid neighbour list: %d slots, %.1f MB\n", FMAXN,
               FMAXN * (double)SPH_NTOTAL * 4.0 / 1048576.0);
    }
    void dump_solid_list()
    {
        std::vector<int> st(SPH_SOLNUM + 1), q((size_t)g.npair);
        CUDA_CHECK(cudaMemcpy(st.data(), g.sStart, st.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q.data(), g.sQ, q.size() * sizeof(int), cudaMemcpyDeviceToHost));
        FILE* f = fopen("solidlist_start.bin", "wb"); fwrite(st.data(), 4, st.size(), f); fclose(f);
        f = fopen("solidlist_q.bin", "wb"); fwrite(q.data(), 4, q.size(), f); fclose(f);
        printf("[review] solid list dumped (%d pairs)\n", g.npair);
    }
    void build_ell(bool dropEdge)
    {
        const int S = SPH_SOLNUM; const size_t P = (size_t)g.npair;
        std::vector<int> st(S + 1), q(P);
        std::vector<Real> G(3 * P), B(3 * P), D(3 * P);
        CUDA_CHECK(cudaMemcpy(st.data(), g.sStart, st.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(q.data(), g.sQ, P * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(G.data(), g.sG, 3 * P * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(B.data(), g.sB, 3 * P * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(D.data(), g.sD, 3 * P * sizeof(Real), cudaMemcpyDeviceToHost));
        auto keep = [&](size_t a) {
            if (!dropEdge) return true;
            const double m = fabs((double)G[3 * a]) + fabs((double)G[3 * a + 1]) + fabs((double)G[3 * a + 2]);
            return m > 1e-20;
        };
        std::vector<int> cnt(S, 0); int maxn = 0; size_t kept = 0;
        for (int k = 0; k < S; ++k) {
            for (int a = st[k]; a < st[k + 1]; ++a) if (keep(a)) ++cnt[k];
            if (cnt[k] > maxn) maxn = cnt[k];
            kept += cnt[k];
        }
        const size_t M = (size_t)maxn * S;
        std::vector<int> eQ(M, 0); std::vector<Real> eG(3 * M, Real(0)), eB(3 * M, Real(0)), eD(3 * M, Real(0));
        for (int k = 0; k < S; ++k) {
            int j = 0;
            for (int a = st[k]; a < st[k + 1]; ++a) {
                if (!keep(a)) continue;
                const size_t e = (size_t)j * S + k;
                eQ[e] = q[a];
                for (int c = 0; c < 3; ++c) { eG[c * M + e] = G[3 * a + c]; eB[c * M + e] = B[3 * a + c]; eD[c * M + e] = D[3 * a + c]; }
                ++j;
            }
        }
        g.ellMax = maxn;
        g.eCnt = (int*)dev_alloc_raw(S * sizeof(int));
        g.eQ = (int*)dev_alloc_raw(M * sizeof(int));
        g.eG = dev_alloc_real_raw(3 * M); g.eB = dev_alloc_real_raw(3 * M); g.eD = dev_alloc_real_raw(3 * M);
        CUDA_CHECK(cudaMemcpy(g.eCnt, cnt.data(), S * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.eQ, eQ.data(), M * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.eG, eG.data(), 3 * M * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.eB, eB.data(), 3 * M * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.eD, eD.data(), 3 * M * sizeof(Real), cudaMemcpyHostToDevice));
        printf("[review] ELL: maxn=%d kept=%zu dropped=%zu padding=%.1f%% %.1f MB\n", maxn, kept, P - kept,
               100.0 * ((double)M / (double)kept - 1.0), (double)M * (sizeof(int) + 9 * sizeof(Real)) / 1048576.0);
    }

    // ---------------- one time step ----------------
''')
sub(c, "        launch_pass1(g, 0);\n        if (dump)",
       "        if (g.fnbr) { launch_fluid_nbr_build(g, FMAXN, d_over, 0); launch_pass1_fluid_list(g, 0); launch_pass1_solid(g, 0); }\n"
       "        else launch_pass1(g, 0);\n        if (dump)")
sub(c, "    if (S.dump_a) S.dump_reference_arrays();\n",
       "    if (S.dump_a) S.dump_reference_arrays();\n"
       "    if (getenv(\"SPH_DUMPSOLIDLIST\") && S.g.npair > 0) S.dump_solid_list();\n"
       "    if (getenv(\"SPH_ELL\") && S.g.npair > 0) { S.build_ell(getenv(\"SPH_ELL_DROPEDGE\") != nullptr); sph_use_ell = 1; }\n"
       "    if (getenv(\"SPH_FLUIDLIST\")) S.alloc_fluid_list();\n")
sub(c, "        bench(\"pass1_solid\",             [&] { launch_pass1_solid(S.g, 0); });\n",
       "        bench(\"pass1_solid\",             [&] { launch_pass1_solid(S.g, 0); });\n"
       "        if (S.g.fnbr) {\n"
       "            bench(\"fluid list build\",      [&] { launch_fluid_nbr_build(S.g, S.FMAXN, S.d_over, 0); });\n"
       "            bench(\"pass1_fluid over list\", [&] { launch_pass1_fluid_list(S.g, 0); });\n"
       "            std::vector<int> fc(SPH_NTOTAL); int ov = 0;\n"
       "            CUDA_CHECK(cudaMemcpy(fc.data(), S.g.fcnt, SPH_NTOTAL * sizeof(int), cudaMemcpyDeviceToHost));\n"
       "            CUDA_CHECK(cudaMemcpy(&ov, S.d_over, sizeof(int), cudaMemcpyDeviceToHost));\n"
       "            long long tot = 0; int mx = 0, mn = 1 << 30; for (int v : fc) { tot += v; if (v > mx) mx = v; if (v < mn) mn = v; }\n"
       "            printf(\"[review] fluid neighbours: total=%lld mean=%.1f min=%d max=%d overflow=%d\\n\", tot, (double)tot / SPH_NTOTAL, mn, mx, ov);\n"
       "        }\n")
sub(c, "    double step_seconds = 0.0;\n",
       "    double step_seconds = 0.0, out_seconds = 0.0;\n"
       "    const bool nosync = getenv(\"SPH_NOSYNC\") != nullptr;\n"
       "    CUDA_CHECK(cudaDeviceSynchronize());\n"
       "    auto L0 = std::chrono::steady_clock::now();\n")
sub(c, "            S.step(itime);\n            CUDA_CHECK(cudaDeviceSynchronize());\n",
       "            S.step(itime);\n            if (!nosync) CUDA_CHECK(cudaDeviceSynchronize());\n")
sub(c, "        if (itime % 50 == 0)\n        {\n            S.fetch_state();\n            S.write_result(itime / 50);\n        }\n",
       "        if (itime % 50 == 0)\n        {\n            auto o0 = std::chrono::steady_clock::now();\n"
       "            S.fetch_state();\n            S.write_result(itime / 50);\n"
       "            out_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - o0).count();\n        }\n")
sub(c, "    if (profile)\n    {\n        const double n = ",
       "    CUDA_CHECK(cudaDeviceSynchronize());\n"
       "    {\n        const double L = std::chrono::duration<double>(std::chrono::steady_clock::now() - L0).count();\n"
       "        printf(\"[review] loop wall %.3f s, output %.3f s, compute %.3f ms/step (%s)\\n\", L, out_seconds,\n"
       "               iMaxStep > 0 ? 1000.0 * (L - out_seconds) / iMaxStep : 0.0, nosync ? \"no per-step sync\" : \"per-step sync\");\n    }\n"
       "    if (profile)\n    {\n        const double n = ")
print("patched OK")
