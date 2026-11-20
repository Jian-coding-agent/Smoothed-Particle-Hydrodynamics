#!/usr/bin/env python3
"""Instrument a scratch copy of cuda/src/sph_host.cu for the D1 review.

Adds (all off by default, so an unset environment reproduces the original):
  SPH_HALOCHK=M         every M steps (and right after each repartition) rank 0
                        checks on the host that each rank holds every particle
                        within 3h of each particle it owns (current x), and every
                        reference-configuration solid partner of each owned solid
                        (|X0_p - X0_q| < 3h, the A3 list's rule).
  SPH_NO_E3=1           skip exchange point E3 (fluid PIJW).
  SPH_FIX_REPART_POS=1  repartition builds ghost sets / per-rank grids from the
                        current positions instead of the t=0 deck rows.
  SPH_D1_SKIN=s         widen the ghost shell from 3h to 3h + s (metres).

usage (from the repository root):
    cp cuda/src/* <scratch>/ && python3 review/Claude/d1_check/patch_inst.py <scratch>
The repository itself is never modified.
"""
import sys, os

path = os.path.join(sys.argv[1], "sph_host.cu")
src = open(path).read()


def rep(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:80])
    src = src.replace(old, new)


# ---- switches --------------------------------------------------------------
rep("static int g_barrier_mode = 0;      // 0 = adaptive (default), 1 = pure spin, 2 = block\n",
    "static int g_barrier_mode = 0;      // 0 = adaptive (default), 1 = pure spin, 2 = block\n"
    "static int g_no_e3 = 0, g_halochk = 0, g_fix_repart_pos = 0;\n"
    "static double g_skin = 0.0;\n")

rep("    if (const char* e = getenv(\"SPH_D1_REPART\")) J.repart = atoi(e);\n",
    "    if (const char* e = getenv(\"SPH_D1_REPART\")) J.repart = atoi(e);\n"
    "    if (const char* e = getenv(\"SPH_NO_E3\")) g_no_e3 = atoi(e);\n"
    "    if (const char* e = getenv(\"SPH_HALOCHK\")) g_halochk = atoi(e);\n"
    "    if (const char* e = getenv(\"SPH_FIX_REPART_POS\")) g_fix_repart_pos = atoi(e);\n"
    "    if (const char* e = getenv(\"SPH_D1_SKIN\")) g_skin = atof(e);\n"
    "    std::printf(\"[inst] no_e3=%d halochk=%d fix_repart_pos=%d skin=%g\\n\", g_no_e3, g_halochk, g_fix_repart_pos, g_skin);\n")

# ---- skin: widen the ghost band in both make_part and build_ranks ---------
rep("    const double band = 3.0 * SPH_HSML;\n", "    const double band = 3.0 * SPH_HSML + g_skin;\n", count=2)

# ---- E3 switch -------------------------------------------------------------
rep("    xch(J, R, 2);                    // E3: fluid PIJW\n",
    "    if (!g_no_e3) xch(J, R, 2);      // E3: fluid PIJW\n")

# ---- repartition fix: current positions into the rows -----------------------
rep("        build_ranks(J, J.deckRows, J.owner, J.box);\n",
    "        if (g_fix_repart_pos)\n"
    "        {\n"
    "            std::vector<double> now = J.deckRows;\n"
    "            for (int g = 0; g < J.num; ++g)\n"
    "                for (int c = 0; c < 3; ++c) now[(size_t)g * 14 + c] = J.pos[(size_t)g * 3 + c];\n"
    "            build_ranks(J, now, J.owner, J.box);\n"
    "        }\n"
    "        else\n"
    "            build_ranks(J, J.deckRows, J.owner, J.box);\n")

# ---- the checker ---------------------------------------------------------------
checker = r'''
// [inst] halo completeness check (rank 0, host, after every rank ran fetch_own)
static void check_halo(Job& J, int itime, const char* tag)
{
    const int num = J.num;
    const double h3 = 3.0 * SPH_HSML, r2max = h3 * h3;
    auto build = [&](const double* X, const double* Y, const double* Z, size_t stride,
                     std::vector<int>& start, std::vector<int>& ord, double mn[3], int dim[3])
    {
        double mx[3] = { -1e300, -1e300, -1e300 };
        for (int c = 0; c < 3; ++c) mn[c] = 1e300;
        for (int g = 0; g < num; ++g)
        {
            const double v[3] = { X[g * stride], Y[g * stride], Z[g * stride] };
            for (int c = 0; c < 3; ++c) { mn[c] = std::min(mn[c], v[c]); mx[c] = std::max(mx[c], v[c]); }
        }
        for (int c = 0; c < 3; ++c) dim[c] = (int)((mx[c] - mn[c]) / h3) + 1;
        const size_t nc = (size_t)dim[0] * dim[1] * dim[2];
        start.assign(nc + 1, 0); ord.assign(num, 0);
        std::vector<int> cell(num);
        for (int g = 0; g < num; ++g)
        {
            int i0 = (int)((X[g * stride] - mn[0]) / h3), i1 = (int)((Y[g * stride] - mn[1]) / h3),
                i2 = (int)((Z[g * stride] - mn[2]) / h3);
            i0 = std::min(std::max(i0, 0), dim[0] - 1); i1 = std::min(std::max(i1, 0), dim[1] - 1);
            i2 = std::min(std::max(i2, 0), dim[2] - 1);
            cell[g] = (i2 * dim[1] + i1) * dim[0] + i0;
            ++start[cell[g] + 1];
        }
        for (size_t c = 0; c < nc; ++c) start[c + 1] += start[c];
        std::vector<int> cur(start.begin(), start.end() - 1);
        for (int g = 0; g < num; ++g) ord[cur[cell[g]]++] = g;
    };
    auto visit = [&](const double* X, const double* Y, const double* Z, size_t stride,
                     const std::vector<int>& start, const std::vector<int>& ord,
                     const double mn[3], const int dim[3], int p, auto&& fn)
    {
        int i0 = (int)((X[p * stride] - mn[0]) / h3), i1 = (int)((Y[p * stride] - mn[1]) / h3),
            i2 = (int)((Z[p * stride] - mn[2]) / h3);
        i0 = std::min(std::max(i0, 0), dim[0] - 1); i1 = std::min(std::max(i1, 0), dim[1] - 1);
        i2 = std::min(std::max(i2, 0), dim[2] - 1);
        for (int a = std::max(i2 - 1, 0); a <= std::min(i2 + 1, dim[2] - 1); ++a)
            for (int b = std::max(i1 - 1, 0); b <= std::min(i1 + 1, dim[1] - 1); ++b)
                for (int c = std::max(i0 - 1, 0); c <= std::min(i0 + 1, dim[0] - 1); ++c)
                {
                    const int cc = (a * dim[1] + b) * dim[0] + c;
                    for (int k = start[cc]; k < start[cc + 1]; ++k) fn(ord[k]);
                }
    };

    std::vector<int> st, od, st0, od0;
    double mn[3], mn0[3];
    int dim[3], dim0[3];
    build(J.gx.data(), J.gy.data(), J.gz.data(), 1, st, od, mn, dim);
    const double* D = J.deckRows.data();
    build(D + 10, D + 11, D + 12, 14, st0, od0, mn0, dim0);       // X0 (reference)

    std::vector<int> present(num, -1);
    long long totMiss = 0, totRef = 0;
    for (int r = 0; r < J.K; ++r)
    {
        const Rank& R = J.ranks[r];
        for (int i = 0; i < R.numP; ++i) present[R.gid[i]] = r;
        long long miss[3] = { 0, 0, 0 }, affected = 0, refMiss = 0;
        double minr = 1e300;
        for (int i = 0; i < R.ownNum; ++i)
        {
            const int p = R.gid[i];
            bool aff = false;
            visit(J.gx.data(), J.gy.data(), J.gz.data(), 1, st, od, mn, dim, p, [&](int q)
            {
                if (q == p || present[q] == r) return;
                const double d0 = J.gx[p] - J.gx[q], d1 = J.gy[p] - J.gy[q], d2 = J.gz[p] - J.gz[q];
                const double r2 = (d0 * d0 + d1 * d1) + d2 * d2;
                if (r2 >= r2max) return;
                const double t = D[(size_t)q * 14 + 9];
                ++miss[t > 0.0 ? 0 : (t < -10.0 ? 1 : 2)];
                minr = std::min(minr, std::sqrt(r2));
                aff = true;
            });
            if (aff) ++affected;
            if (D[(size_t)p * 14 + 9] < -10.0)
                visit(D + 10, D + 11, D + 12, 14, st0, od0, mn0, dim0, p, [&](int q)
                {
                    if (q == p || present[q] == r || D[(size_t)q * 14 + 9] >= -10.0) return;
                    const double d0 = D[(size_t)p * 14 + 10] - D[(size_t)q * 14 + 10];
                    const double d1 = D[(size_t)p * 14 + 11] - D[(size_t)q * 14 + 11];
                    const double d2 = D[(size_t)p * 14 + 12] - D[(size_t)q * 14 + 12];
                    const double Dr = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
                    if (Dr < h3 * (1.0 - 1e-9)) ++refMiss;
                });
        }
        const long long m = miss[0] + miss[1] + miss[2];
        totMiss += m; totRef += refMiss;
        if (m || refMiss)
            std::printf("[halochk] step %d %s rank %d: MISSING %lld neighbour pairs (fluid %lld solid %lld wall %lld) "
                        "for %lld owned particles, closest r = %.4f*3h; A3 ref partners missing %lld\n",
                        itime, tag, r, m, miss[0], miss[1], miss[2], affected, minr / h3, refMiss);
    }
    std::printf("[halochk] step %d %s: total missing pairs %lld, missing A3 ref partners %lld\n",
                itime, tag, totMiss, totRef);
}
'''
rep("// ---------------------------------------------------------------------------\n"
    "// Repartition (SPH_D1_REPART=N): recompute the blocks",
    checker +
    "// ---------------------------------------------------------------------------\n"
    "// Repartition (SPH_D1_REPART=N): recompute the blocks")

# ---- call sites in rank_main -------------------------------------------------
rep("        if (J.repart > 0 && itime > 0 && (itime % J.repart) == 0 && itime < J.maxStep - 1)\n"
    "            repartition(J, R, itime);\n",
    "        if (J.repart > 0 && itime > 0 && (itime % J.repart) == 0 && itime < J.maxStep - 1)\n"
    "        {\n"
    "            repartition(J, R, itime);\n"
    "            if (g_halochk > 0)\n"
    "            {\n"
    "                fetch_own(J, R); J.bar.wait();\n"
    "                if (r == 0) check_halo(J, itime, \"after-repartition\");\n"
    "                J.bar.wait();\n"
    "            }\n"
    "        }\n"
    "        else if (g_halochk > 0 && (itime % g_halochk) == 0)\n"
    "        {\n"
    "            fetch_own(J, R); J.bar.wait();\n"
    "            if (r == 0) check_halo(J, itime, \"periodic\");\n"
    "            J.bar.wait();\n"
    "        }\n")

# the checker needs the deck rows even without repartition
rep("    J.deckRows = std::move(rows);\n", "    J.deckRows = std::move(rows);\n    (void)0;\n")
rep("#include <string>\n", "#include <string>\n#include <cmath>\n")

open(path, "w").write(src)
print("patched", path)
