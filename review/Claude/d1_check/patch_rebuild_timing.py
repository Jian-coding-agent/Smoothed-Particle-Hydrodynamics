#!/usr/bin/env python3
"""Add phase timers to d1::repartition() in a scratch copy of sph_host.cu (HEAD c146aa7).

Prints, per rebuild event:
  rank 0 layout: deck arrays, 3 axis sorts, make_part (+regrid), the `cur` deck copy,
                 build_ranks
  every rank:    snapshot (D2H + gid scatter), state rebuild (free/finish_deck/upload/
                 mig_restore/send lists), static exchange, solid pack + A3 lists
usage (from the repository root):
    cp cuda/src/* <scratch>/ && python3 review/Claude/d1_check/patch_rebuild_timing.py <scratch>
"""
import sys, os

path = os.path.join(sys.argv[1], "sph_host.cu")
src = open(path).read()


def rep(old, new):
    global src
    n = src.count(old)
    assert n == 1, (n, old[:70])
    src = src.replace(old, new)


rep("    mig_snapshot(J, R);\n    J.bar.wait();\n\n    // 2) the new layout",
    "    mig_snapshot(J, R);\n    const double tA = now_s();\n    J.bar.wait();\n    const double tA2 = now_s();\n\n    // 2) the new layout")

rep("    if (R.id == 0)\n    {\n        Deck D;\n        D.num = J.num;",
    "    if (R.id == 0)\n    {\n        const double tB0 = now_s();\n        Deck D;\n        D.num = J.num;")

rep("        D.ordZ = sort_axis(D.z);\n\n        std::vector<int> ixOf(J.num), iyOf(J.num), izOf(J.num), acc;\n"
    "        Part P = make_part(D, J.gnx, J.gny, J.gnz, J.skin, ixOf, iyOf, izOf, acc);",
    "        D.ordZ = sort_axis(D.z);\n        const double tB1 = now_s();\n\n"
    "        std::vector<int> ixOf(J.num), iyOf(J.num), izOf(J.num), acc;\n"
    "        Part P = make_part(D, J.gnx, J.gny, J.gnz, J.skin, ixOf, iyOf, izOf, acc);")

rep("        J.owner = P.owner;\n        J.box = P.box;\n",
    "        const double tB2 = now_s();\n        J.owner = P.owner;\n        J.box = P.box;\n")

rep("        build_ranks(J, cur, J.owner, J.box);\n",
    "        const double tB3 = now_s();\n        build_ranks(J, cur, J.owner, J.box);\n"
    "        const double tB4 = now_s();\n"
    "        std::printf(\"[tim] rank0 layout: deck %.1f+sorts %.1f ms, make_part %.1f ms, cur copy %.1f ms, \"\n"
    "                    \"build_ranks %.1f ms\\n\", 0.0, 1e3 * (tB1 - tB0), 1e3 * (tB2 - tB1), 1e3 * (tB3 - tB2), 1e3 * (tB4 - tB3));\n")

rep("    J.bar.wait();\n\n    // 3) rebuild this rank",
    "    J.bar.wait();\n    const double tC0 = now_s();\n\n    // 3) rebuild this rank")

rep("    build_send_lists(J, R);\n    launch_copy_xf(R.S.g, 0);\n    xch_static(J, R);                     // ghost A/vecn/p/irho/SurFD\n"
    "    launch_build_solid_pack(R.S.g, 0);\n    if (sph_use_static_solid) R.S.build_solid_lists();\n    CUDA_CHECK(cudaDeviceSynchronize());\n",
    "    build_send_lists(J, R);\n    launch_copy_xf(R.S.g, 0);\n    CUDA_CHECK(cudaDeviceSynchronize());\n    const double tC1 = now_s();\n"
    "    xch_static(J, R);                     // ghost A/vecn/p/irho/SurFD\n    const double tD = now_s();\n"
    "    launch_build_solid_pack(R.S.g, 0);\n    if (sph_use_static_solid) R.S.build_solid_lists();\n    CUDA_CHECK(cudaDeviceSynchronize());\n"
    "    const double tE = now_s();\n"
    "    { std::lock_guard<std::mutex> lk(J.print_m);\n"
    "      std::printf(\"[tim] rank %d: snapshot %.1f ms, wait-for-layout %.1f ms, state rebuild %.1f ms, \"\n"
    "                  \"static xch %.1f ms, pack+A3 %.1f ms (numP %d)\\n\", R.id, 1e3 * (tA - t0), 1e3 * (tC0 - tA2),\n"
    "                  1e3 * (tC1 - tC0), 1e3 * (tD - tC1), 1e3 * (tE - tD), R.S.g.d.numP); }\n")

open(path, "w").write(src)
print("patched", path)
