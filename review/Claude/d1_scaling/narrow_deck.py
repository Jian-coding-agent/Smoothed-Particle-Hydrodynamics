#!/usr/bin/env python3
"""Narrow copy of input.dat that fits in ~190 MB of free VRAM.

usage: narrow_deck.py <input.dat> <out.dat> <ycut> <zcut> [xcol]

* keeps every particle with y < ycut (fluid, solid, wall);
* translates the far y-wall and the plate's far edge (y >= 0.505) down to
  just past the cut, so the tank stays closed;
* drops walls/solids above zcut (the fluid stays below 0.45 m);
* xcol (optional): keep only fluid with x < xcol -> a dam-break column.
Block order fluid|solid|wall is preserved; X0 moves with x.
"""
import sys
import numpy as np

src, dst, ycut, zcut = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
xcol = float(sys.argv[5]) if len(sys.argv) > 5 else None
a = np.loadtxt(src)
it = a[:, 9]
cls = np.where(it > 0, 0, np.where(it < -10, 1, 2))
y = a[:, 1]
shift = (np.floor((ycut - 0.005) / 0.01) * 0.01 + 0.01) - 0.51   # far wall -> first lattice row past the kept walls
out = []
for c in (0, 1, 2):
    b = a[cls == c]
    keep = b[:, 1] < ycut - 0.005
    if c == 0 and xcol is not None:
        keep &= b[:, 0] < xcol
    if c > 0:
        keep &= b[:, 2] < zcut
        far = (b[:, 1] > 0.505) & (b[:, 2] < zcut)
        f = b[far].copy()
        f[:, 1] += shift
        f[:, 11] += shift
        out.append(np.vstack([b[keep], f]))
    else:
        out.append(b[keep])
d = np.vstack(out)
np.savetxt(dst, d, delimiter="\t", fmt="%.17g")
n = [len(o) for o in out]
print(f"{dst}: num={len(d)} fluid={n[0]} solid={n[1]} wall={n[2]} (far wall shifted {shift:+.3f} m)")
