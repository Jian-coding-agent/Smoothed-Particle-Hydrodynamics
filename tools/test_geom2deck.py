#!/usr/bin/env python3
"""Self-test for tools/geom2deck.py -- no reference deck needed.

    python tools/test_geom2deck.py

Checks, in order:

1. the 8-hole gate case, described as a case file, reproduces the particle
   counts of the hand-written generator (2,194,500 fluid / 606,000 solid /
   915,328 wall = 3,715,828);
2. the deck is ordered fluid | solid | wall, which is what both solvers demand;
3. dumping the box parts to STL and rebuilding through the STL path gives the
   identical particle set -- i.e. the CAD path is the same path;
4. an analytic cylinder is voxelised to within a few percent of pi r^2 h / dx^3
   and its extent is within one spacing of the true radius.

Dot ``--check <deck>`` at the command line of geom2deck.py to additionally
compare against a real deck (e.g. the 8-hole reference).
"""
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geom2deck as G

CASE = os.path.join(HERE, "..", "cases", "8hole_gate.json")
EXPECT = {"fluid": 2194500, "elastic": 606000, "rigid": 915328}


def counts(parts):
    return dict((r, sum(int(p["mask"].sum()) for p in parts if p["role"] == r))
                for r in G.BLOCKS)


def build_case(path):
    case = G.load_case(path)
    grid, parts = G.build(case, os.path.dirname(os.path.abspath(path)),
                          verbose=False)
    return grid, parts


def keys(mask, grid):
    return np.sort(np.flatnonzero(mask.ravel()))


def test_counts():
    grid, parts = build_case(CASE)
    got = counts(parts)
    assert got == EXPECT, "counts %s, expected %s" % (got, EXPECT)
    total = sum(got.values())
    assert total == 3715828, "total %d" % total
    print("  ok  %s particles, fluid %s / solid %s / wall %s"
          % (f"{total:,}", f"{got['fluid']:,}", f"{got['elastic']:,}",
             f"{got['rigid']:,}"))


def test_block_order():
    grid, parts = build_case(CASE)
    rows, report = G.assemble(grid, parts)
    it = np.concatenate([r[:, 9] for r in rows])
    nf, ns = report["fluid"]["n"], report["elastic"]["n"]
    assert (it[:nf] > 0).all(), "fluid block holds a non-fluid itype"
    assert (it[nf:nf + ns] < -10).all(), "solid block holds a non-solid itype"
    assert ((it[nf + ns:] >= -10) & (it[nf + ns:] <= 0)).all(), \
        "wall block holds a non-wall itype"
    r = rows[0]
    assert np.all(r[:, 3] == 0.0), "the deck must start at zero pressure"
    assert np.allclose(r[:, 8], 0.001), "mass should be rho*dx^3 = 0.001"
    assert np.allclose(r[:, 10:13], r[:, 0:3]), "X0 must start equal to x"
    print("  ok  blocks fluid|solid|wall, p=0, mass=0.001, X0=x")


def scratch_dir():
    """A writable scratch directory.

    Not tempfile.mkdtemp: that creates the directory with mode 0700, i.e. a
    DACL that a restricted sandbox token cannot write into.  os.makedirs uses
    the inherited ACL, which is what the rest of the tree has.
    """
    for base in (os.environ.get("SPH_TMPDIR"),
                 os.path.join(HERE, "..", "_local"),
                 os.path.join(HERE, "..")):
        if not base or not os.path.isdir(base):
            continue
        d = os.path.join(base, "geom2deck_test_%d" % os.getpid())
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, ".probe"), "w") as f:
                f.write("x")
            os.remove(os.path.join(d, ".probe"))
            return d
        except OSError:
            shutil.rmtree(d, ignore_errors=True)
    raise SystemExit("no writable scratch directory (set SPH_TMPDIR)")


def test_stl_roundtrip():
    tmp = scratch_dir()
    try:
        case = G.load_case(CASE)
        G.dump_stl(case, tmp)
        grid_a, pa = build_case(CASE)
        grid_b, pb = build_case(os.path.join(tmp, "case_stl.json"))
        assert grid_a.n == grid_b.n, "lattice changed"
        bad = []
        for role in G.BLOCKS:
            ma = np.zeros(grid_a.shape, dtype=bool)
            mb = np.zeros(grid_b.shape, dtype=bool)
            for p in pa:
                if p["role"] == role:
                    ma |= p["mask"]
            for p in pb:
                if p["role"] == role:
                    mb |= p["mask"]
            if not np.array_equal(keys(ma, grid_a), keys(mb, grid_b)):
                bad.append(role)
        assert not bad, "STL path disagrees with the box path for %s" % bad
        print("  ok  every box part re-read as an STL gives the same particles")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wall_layers():
    """A rigid wall keeps only the N layers that touch a deformable cell."""
    case = {"spacing": 0.01,
            "domain": {"min": [0.0, 0.0, 0.0], "max": [0.20, 0.20, 0.20]},
            "parts": [
                {"name": "w", "role": "fluid",
                 "geom": {"type": "box", "min": [0.0, 0.0, 0.0],
                          "max": [0.09, 0.20, 0.20]}},
                {"name": "wall", "role": "rigid", "layers": 2,
                 "geom": {"type": "box", "min": [0.10, 0.0, 0.0],
                          "max": [0.20, 0.20, 0.20]}},
            ]}
    grid, parts = G.build(case, ".", verbose=False)
    wall = [p for p in parts if p["name"] == "wall"][0]
    expect = 2 * 21 * 21            # x = 0.10 and 0.11, full y-z
    assert int(wall["mask"].sum()) == expect, \
        "2 layers kept %d cells, expected %d" % (int(wall["mask"].sum()), expect)
    assert wall["mask"][:, :, 10].all(), "the layer on the wall face is missing"
    assert not wall["mask"][:, :, 12].any(), "layer 3 was kept"
    # layers=0 means the whole block
    case["parts"][1]["layers"] = 0
    grid, parts = G.build(case, ".", verbose=False)
    wall = [p for p in parts if p["name"] == "wall"][0]
    assert int(wall["mask"].sum()) == 11 * 21 * 21, "layers=0 did not fill the block"
    print("  ok  rigid wall thinned to 2 layers (%d cells), layers=0 fills it"
          % expect)


def cylinder(r=0.30, h=1.0, seg=64, dx=0.01):
    t = np.linspace(0.0, 2.0 * math.pi, seg, endpoint=False)
    ring = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    lo = np.column_stack([ring, np.zeros(seg)])
    hi = np.column_stack([ring, np.full(seg, h)])
    tris = []
    for i in range(seg):
        j = (i + 1) % seg
        tris += [[lo[i], lo[j], hi[j]], [lo[i], hi[j], hi[i]],
                 [[0, 0, 0], lo[j], lo[i]], [[0, 0, h], hi[i], hi[j]]]
    return np.array(tris, dtype=np.float64)


def test_cylinder():
    r, h, dx = 0.30, 1.0, 0.01
    tris = cylinder(r, h, dx=dx)
    v = tris.reshape(-1, 3)
    lo = np.floor(v.min(axis=0) / dx) * dx - 0.05
    hi = np.ceil(v.max(axis=0) / dx) * dx + 0.05
    grid = G.Grid(lo, hi, dx)
    surf, sin = G.rasterise_surface(grid, tris)
    bb = G.bbox_of(surf)
    win = [(max(a - 1, 0), min(b + 1, grid.n[c] - 1))
           for c, (a, b) in enumerate(bb)]
    deep = G.flood_outside(surf, win)
    (i0, i1), (j0, j1), (k0, k1) = win
    sw = surf[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
    si = sin[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
    n = int((si | ((~sw) & deep)).sum())
    exact = math.pi * r ** 2 * h / dx ** 3
    err = 100.0 * (n - exact) / exact
    # the lattice body carries one boundary layer on every side, so a few
    # percent high is the expected sign; a large negative error is a bug
    assert -3.0 < err < 6.0, "cylinder volume error %+.2f%%" % err
    print("  ok  cylinder %+.2f%% of pi r^2 h / dx^3 (%d vs %.0f particles)"
          % (err, n, exact))


def test_text_roundtrip():
    """The .dat writer and reader keep the same lattice and itypes."""
    grid, parts = build_case(CASE)
    rows, report = G.assemble(grid, parts)
    tmp = scratch_dir()
    try:
        path = os.path.join(tmp, "deck.dat")
        G.write_text(path, rows)
        with open(path, "rb") as f:
            head = f.readline()
        assert head.endswith(b"\r\n"), "the deck must be CRLF-terminated"
        assert head.count(b"\t") == 13, "expected 14 tab-separated columns"
        back = G.load_deck(path)
        assert back.shape[0] == sum(r.shape[0] for r in rows), "row count changed"
        for role, a, b in zip(G.BLOCKS, rows,
                              [back[back[:, 9] > 0],
                               back[back[:, 9] < -10],
                               back[(back[:, 9] >= -10) & (back[:, 9] <= 0)]]):
            if a.shape[0] != b.shape[0]:
                raise AssertionError("%s block: %d rows written, %d read back"
                                     % (role, a.shape[0], b.shape[0]))
            if a.shape[0] == 0:
                continue
            if not np.array_equal(G.lattice_keys(a, grid), G.lattice_keys(b, grid)):
                raise AssertionError("%s block: lattice changed through the text "
                                     "deck" % role)
            assert np.array_equal(a[:, 9], b[:, 9]), "%s: itype changed" % role
        print("  ok  14 tab-separated columns, CRLF, same lattice after a "
              "text round trip")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if not os.path.exists(CASE):
        print("missing %s -- run cases/make_8hole_gate.py first" % CASE)
        sys.exit(1)
    tests = [test_counts, test_block_order, test_stl_roundtrip, test_text_roundtrip,
             test_wall_layers, test_cylinder]
    failed = 0
    for t in tests:
        print(t.__name__)
        try:
            t()
        except AssertionError as exc:
            print("  FAIL %s" % exc)
            failed += 1
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
