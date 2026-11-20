#!/usr/bin/env python3
"""Test decks used by REVIEW-2026-11-19-d1-followup.md, built from input.dat.

  make_decks.py lone  <input.dat> <out.dat>      one solid particle moved into the fluid,
                                                 > 0.4 m from every other solid: at K=2 the
                                                 rank that owns it has a computing solid with
                                                 no A3 partner (C16)
  make_decks.py rep   <input.dat> <out.sphbin> R R copies along x with 5 cm gaps, written as
                                                 .sphbin (block order fluid|solid|wall kept);
                                                 the construction the D1 notes use for 9.6 M /
                                                 19.3 M, here to time rebuilds at 1.9 M (R=8)
"""
import sys, struct
import numpy as np


def lone(src, dst):
    a = np.loadtxt(src)
    so = np.flatnonzero(a[:, 9] < -10)
    i = int(so[0])
    p = np.array([0.105, 0.255, 0.105])          # inside the fluid, off the lattice
    a[i, 0:3] = p
    a[i, 10:13] = p
    a[i, 4:7] = 0.0
    np.savetxt(dst, a, delimiter="\t", fmt="%.10g")
    print(f"{dst}: solid row {i} moved to {p.tolist()}")


def rep(src, dst, R):
    a = np.loadtxt(src)
    it = a[:, 9]
    cls = np.where(it > 0, 0, np.where(it < -10, 1, 2))
    w = a[:, 0].max() - a[:, 0].min() + 0.05
    blocks = []
    for c in (0, 1, 2):
        part = a[cls == c]
        for k in range(R):
            b = part.copy()
            b[:, 0] += k * w
            b[:, 10] += k * w
            blocks.append(b)
    d = np.vstack(blocks)
    n, nt, ns = len(d), int((cls == 0).sum()) * R, int((cls == 1).sum()) * R
    g = (d[:, 0].min() - .05, d[:, 1].min() - .05, d[:, 2].min() - .05,
         d[:, 0].max() + .05, d[:, 1].max() + .05, d[:, 2].max() + .05)
    hdr = struct.pack("<8sIIIIIIII dd 6d 24x", b"SPHBIN\0\0", 1, 128, n, nt, ns, 14, 0, 1,
                      0.01, 1e-5, *g)
    with open(dst, "wb") as f:
        f.write(hdr)
        f.write(np.ascontiguousarray(d, dtype="<f8").tobytes())
    print(f"{dst}: num={n} fluid={nt} solid={ns} wall={n - nt - ns}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "lone":
        lone(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 5 and sys.argv[1] == "rep":
        rep(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        print(__doc__)
        sys.exit(1)
