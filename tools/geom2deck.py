#!/usr/bin/env python3
"""geom2deck -- build an SPH input deck from CAD geometry (STL) or boxes.

The original preprocessor (``dist/Input_3D/1.cpp``) places particles with one
hand-written loop nest per block: 1300 lines of ``for(i=151;i<=160;i++)`` for
the 8-hole gate case, and a new case means editing C++ index ranges.  This tool
takes the same information the way a CFD preprocessor takes it -- a domain, a
particle spacing, a fluid region and a set of CAD parts -- voxelises the parts
onto the particle lattice and writes the 14-column deck (or the ``.sphbin``
container) that ``cpu/`` and ``cuda/`` read.

The pipeline is the one FLOW-3D uses for FAVOR, minus the area/volume fractions:

    read the parts (STL triangles, or analytic boxes)
    rasterise each part's surface onto the lattice
    flood fill from the part's bounding box -> whatever is left is its volume
    give every cell a role (fluid / elastic solid / rigid wall)
    thin a rigid wall to the N layers that touch a deformable region
    emit the deck, ordered fluid | solid | wall as the solvers require

Roles map onto the ``itype`` values the solver already switches on:

    role      itype   meaning in the solver
    fluid       2     Riemann fluid pair, density from the continuity equation
    elastic   -20     total-Lagrangian solid, own stress update
    rigid      -2     Monaghan boundary particle, held fixed

Usage
-----
    python tools/geom2deck.py cases/8hole_gate.json --out input.dat
    python tools/geom2deck.py cases/8hole_gate.json --sphbin input.sphbin
    python tools/geom2deck.py cases/8hole_gate.json --check ref.sphbin
    python tools/geom2deck.py cases/8hole_gate.json --dump-stl out/stl

``--check`` compares the generated particle set against an existing deck, which
is how the CAD path is validated against the hand-written one.  ``--dump-stl``
writes every box part out as an ASCII STL, so the same case can be regenerated
through the STL path and the two compared (see docs/GEOMETRY_INPUT.md).

Case file
---------
    {
      "spacing": 0.01,                       // particle spacing [m], = hsml
      "domain": {"min": [-0.03,-0.03,-0.03], "max": [3.60, 3.34, 1.00]},
      "parts": [
        {"name": "water", "role": "fluid", "geom": {"type": "box",
         "min": [0.01,0.01,0.01], "max": [0.70,3.30,0.95]}},

        {"name": "gate", "role": "elastic", "geom": {"type": "boxes",
         "boxes": [[1.51,0.01,0.01, 1.60,3.30,1.00], ...]}},

        {"name": "floor", "role": "rigid", "layers": 0, "geom": {"type": "box", ...}},

        {"name": "sluice", "role": "rigid", "layers": 3,
         "geom": {"type": "stl", "file": "sluice.stl", "scale": 0.001,
                  "rotate": {"axis": [0,0,1], "angle_deg": 90},
                  "translate": [0.1, 0, 0], "seal": 0}}
      ]
    }

    role      "fluid" | "elastic" | "rigid"
    layers    rigid only: 0/absent = fill the part solid; N>0 = keep only the N
              lattice layers that touch a fluid or elastic cell.  A wall only
              needs ceil(3*hsml/dx)+1 layers of particles, so this is what
              keeps a concrete block from costing millions of particles.
    seal      STL only: grow the surface mask by N cells before the flood fill,
              to close holes in a mesh that leaks (default 0)
    itype     override the itype (default from the role)
    rho       per-particle density (default 1000)
    velocity  initial velocity [vx,vy,vz] (default 0)

Exit status 0 on success, 1 on a bad case file, 2 when ``--check`` disagrees.
"""
import argparse
import json
import math
import os
import re
import struct
import sys
import time

import numpy as np

MAGIC = b"SPHBIN\0\0"
SPHBIN_HDR = struct.Struct("<8sIIIIIIII dd 6d 24x")   # see cuda/tools/sphbin.py
assert SPHBIN_HDR.size == 128

ROLE_ID = {"fluid": 1, "elastic": 2, "rigid": 3}
ITYPE_BY_ROLE = {"fluid": 2.0, "elastic": -20.0, "rigid": -2.0}
BLOCKS = ("fluid", "elastic", "rigid")


def log(*a):
    print(*a, flush=True)


def warn(msg):
    print("warning: " + msg, file=sys.stderr, flush=True)


def human(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return "%.1f %s" % (n, u)
        n /= 1024.0


# --------------------------------------------------------------------------
# geometry sources
# --------------------------------------------------------------------------

def read_stl(path):
    """(n,3,3) float64 triangles from a binary or ASCII STL."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 15:
        raise ValueError("%s: too short to be an STL" % path)
    n = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
    if n > 0 and 84 + 50 * n == len(data):
        rec = np.frombuffer(data, dtype=np.uint8, count=50 * n, offset=84)
        v = rec.reshape(n, 50)[:, 12:48].copy().view("<f4")
        return v.reshape(n, 3, 3).astype(np.float64)
    # ASCII.  ("solid" is not a test: binary files usually start with it.)
    txt = data.decode("ascii", "replace")
    if "facet" not in txt:
        raise ValueError("%s: not an STL (no binary count, no ASCII facets)" % path)
    v = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt)
    if not v or len(v) % 3:
        raise ValueError("%s: %d ASCII vertices (not a multiple of 3)"
                         % (path, len(v)))
    return np.array(v, dtype=np.float64).reshape(-1, 3, 3)


# 12 outward-oriented triangles of the unit box, as index triples
_BOX_TRI = np.array([
    [[0, 0, 0], [0, 1, 0], [0, 1, 1]], [[0, 0, 0], [0, 1, 1], [0, 0, 1]],   # -x
    [[1, 0, 0], [1, 1, 1], [1, 1, 0]], [[1, 0, 0], [1, 0, 1], [1, 1, 1]],   # +x
    [[0, 0, 0], [0, 0, 1], [1, 0, 1]], [[0, 0, 0], [1, 0, 1], [1, 0, 0]],   # -y
    [[0, 1, 0], [1, 1, 0], [1, 1, 1]], [[0, 1, 0], [1, 1, 1], [0, 1, 1]],   # +y
    [[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 1, 0], [0, 1, 0]],   # -z
    [[0, 0, 1], [0, 1, 1], [1, 1, 1]], [[0, 0, 1], [1, 1, 1], [1, 0, 1]],   # +z
], dtype=np.float64)


def box_triangles(lo, hi):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return lo + _BOX_TRI * (hi - lo)


def as_box(b):
    """[x0,y0,z0,x1,y1,z1] or {"min":[..],"max":[..]} -> (lo, hi)."""
    if isinstance(b, dict):
        return b["min"], b["max"]
    v = list(b)
    if len(v) != 6:
        raise ValueError("a box needs 6 numbers [x0,y0,z0,x1,y1,z1], got %r" % (b,))
    return v[:3], v[3:]


def part_geom(part):
    g = part.get("geom", {})
    return g.get("type", "box"), g


def part_boxes(part):
    """List of (lo,hi) for box/boxes parts, else None."""
    kind, g = part_geom(part)
    if kind == "box":
        return [as_box(g)]
    if kind == "boxes":
        return [as_box(b) for b in g["boxes"]]
    return None


def rot_matrix(rot):
    if not rot:
        return None
    ax = np.asarray(rot["axis"], dtype=np.float64)
    n = np.linalg.norm(ax)
    if n <= 0:
        raise ValueError("rotate.axis must be non-zero")
    ax = ax / n
    t = math.radians(float(rot.get("angle_deg", 0.0)))
    c, s = math.cos(t), math.sin(t)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return c * np.eye(3) + s * K + (1 - c) * np.outer(ax, ax)


def part_triangles(part, base_dir):
    kind, g = part_geom(part)
    if kind in ("box", "boxes"):
        return np.concatenate([box_triangles(lo, hi) for lo, hi in part_boxes(part)])
    if kind != "stl":
        raise ValueError("part %r: unknown geom type %r" % (part.get("name"), kind))
    f = g["file"]
    if not os.path.isabs(f):
        f = os.path.join(base_dir, f)
    if not os.path.exists(f):
        raise ValueError("part %r: STL %s does not exist" % (part.get("name"), f))
    tris = read_stl(f)
    s = float(g.get("scale", 1.0))
    if s != 1.0:
        tris = tris * s
    R = rot_matrix(g.get("rotate"))
    if R is not None:
        tris = tris @ R.T
    if g.get("translate") is not None:
        tris = tris + np.asarray(g["translate"], dtype=np.float64)
    return tris


# --------------------------------------------------------------------------
# lattice
# --------------------------------------------------------------------------

class Grid(object):
    """Particle lattice: index (i,j,k) <-> position x0 + (i,j,k)*dx.

    Fields are (nz, ny, nx) so the fastest axis is x and a flat index
    ``i + nx*j + nx*ny*k`` is already sorted by (k,j,i).
    """

    def __init__(self, lo, hi, dx):
        self.dx = float(dx)
        if self.dx <= 0:
            raise ValueError("spacing must be positive")
        self.x0 = np.asarray(lo, dtype=np.float64)
        span = np.asarray(hi, dtype=np.float64) - self.x0
        n = np.rint(span / self.dx).astype(np.int64) + 1
        if np.any(n < 2) or np.any(span <= 0):
            raise ValueError("domain min must be below max and span at least 2 spacings")
        self.nx, self.ny, self.nz = (int(v) for v in n)
        self.n = (self.nx, self.ny, self.nz)
        self.shape = (self.nz, self.ny, self.nx)
        self.size = self.nx * self.ny * self.nz
        err = float(np.abs(n - 1 - span / self.dx).max())
        if err > 1e-6:
            warn("domain extent %s is not a whole number of %.6g m spacings; "
                 "snapped to %dx%dx%d (edges moved by up to %.3g m)"
                 % (np.round(span, 9).tolist(), self.dx,
                    self.nx, self.ny, self.nz, err * self.dx))

    def empty(self, dtype=bool):
        return np.zeros(self.shape, dtype=dtype)

    def clip_span(self, lo, hi):
        """Index box of [lo,hi] widened by up to one cell, clipped, or None."""
        r = []
        for c in range(3):
            a = (float(lo[c]) - self.x0[c]) / self.dx
            b = (float(hi[c]) - self.x0[c]) / self.dx
            r.append((max(int(math.floor(a)) - 1, 0),
                      min(int(math.ceil(b)) + 1, self.n[c] - 1)))
        (i0, i1), (j0, j1), (k0, k1) = r
        if i0 > i1 or j0 > j1 or k0 > k1:
            return None
        return i0, i1, j0, j1, k0, k1

    def index_range(self, lo, hi, tol=1e-6):
        """Inclusive index box of the closed coordinate box [lo,hi]."""
        r = []
        for c in range(3):
            a = (float(lo[c]) - self.x0[c]) / self.dx
            b = (float(hi[c]) - self.x0[c]) / self.dx
            ia, ib = int(round(a)), int(round(b))
            if max(abs(a - ia), abs(b - ib)) > tol:
                warn("box edge %s=[%.6g,%.6g] is off the lattice by at most "
                     "%.3g spacings; snapped to [%.6g,%.6g]"
                     % ("xyz"[c], lo[c], hi[c], max(abs(a - ia), abs(b - ib)),
                        self.x0[c] + ia * self.dx, self.x0[c] + ib * self.dx))
            r.append((ia, ib))
        (i0, i1), (j0, j1), (k0, k1) = r
        i0, i1 = max(i0, 0), min(i1, self.nx - 1)
        j0, j1 = max(j0, 0), min(j1, self.ny - 1)
        k0, k1 = max(k0, 0), min(k1, self.nz - 1)
        if i0 > i1 or j0 > j1 or k0 > k1:
            return None
        return i0, i1, j0, j1, k0, k1

    def centers(self, i0, i1, j0, j1, k0, k1):
        x = self.x0[0] + np.arange(i0, i1 + 1) * self.dx
        y = self.x0[1] + np.arange(j0, j1 + 1) * self.dx
        z = self.x0[2] + np.arange(k0, k1 + 1) * self.dx
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        return np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)

    def positions(self, flat):
        i = flat % self.nx
        j = (flat // self.nx) % self.ny
        k = flat // (self.nx * self.ny)
        return self.x0[:, None] + np.stack((i, j, k)) * self.dx


# --------------------------------------------------------------------------
# voxelisation: surface rasterisation + bounding-box flood fill
# --------------------------------------------------------------------------

def _pt_tri_dist2(P, A, B, C):
    """Squared distance from each row of P (M,3) to triangle (A,B,C).

    Ericson, Real-Time Collision Detection, 5.1.5, vectorised over the points.
    """
    AB = B - A
    AC = C - A
    AP = P - A
    d1 = AP @ AB
    d2 = AP @ AC
    BP = P - B
    d3 = BP @ AB
    d4 = BP @ AC
    CP = P - C
    d5 = CP @ AB
    d6 = CP @ AC
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-300, 1e-300, den)
    closest = A + AB * (vb / den)[:, None] + AC * (vc / den)[:, None]

    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)                  # edge AB
    t = d1 / np.where(np.abs(d1 - d3) < 1e-300, 1e-300, d1 - d3)
    closest[m] = A + AB * t[m][:, None]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)                  # edge AC
    t = d2 / np.where(np.abs(d2 - d6) < 1e-300, 1e-300, d2 - d6)
    closest[m] = A + AC * t[m][:, None]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)    # edge BC
    u = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-300, 1e-300,
                             (d4 - d3) + (d5 - d6))
    closest[m] = B + (C - B) * u[m][:, None]
    m = (d1 <= 0) & (d2 <= 0)                              # vertex A
    closest[m] = A
    m = (d3 >= 0) & (d4 <= d3)                             # vertex B
    closest[m] = B
    m = (d6 >= 0) & (d5 <= d6)                             # vertex C
    closest[m] = C

    d = P - closest
    return np.einsum("ij,ij->i", d, d)


def _mark(grid, best, side, A, B, C, box, thresh):
    """Keep the nearest triangle per cell and whether the cell is inside it."""
    i0, i1, j0, j1, k0, k1 = box
    p = grid.centers(i0, i1, j0, j1, k0, k1)
    d2 = _pt_tri_dist2(p, A, B, C)
    n = np.cross(B - A, C - A)
    ln = math.sqrt(float(n @ n))
    ins = ((p - A) @ n) >= -1e-9 * grid.dx * ln if ln > 0 else np.ones(len(p), bool)
    shp = (k1 - k0 + 1, j1 - j0 + 1, i1 - i0 + 1)
    d2 = d2.reshape(shp)
    sl = best[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
    take = (d2 <= thresh) & (d2 < sl)
    sl[take] = d2[take]
    side[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1][take] = ins.reshape(shp)[take]


def rasterise_surface(grid, tris, budget=2000000):
    """Classify the lattice cells the triangle surface passes through.

    Returns ``(surf, inside)``: ``surf`` is every cell whose lattice point is
    within half a spacing of some triangle; ``inside`` is the subset for which
    the *nearest* triangle puts the point on its inward side.

    For geometry that sits on the lattice (the usual case: a CAD face at a round
    coordinate) the signed test is exactly zero and resolves to "inside", which
    reproduces the closed convention of the hand-written generator: the cell on
    the face belongs to the part, the cell one spacing away does not.  A curved
    or slanted face is therefore classified by where its cell centres fall,
    which is unbiased -- see docs/GEOMETRY_INPUT.md for the measured error.
    """
    best = np.full(grid.shape, np.inf, dtype=np.float32)
    side = np.zeros(grid.shape, dtype=bool)
    thresh = (0.5 * grid.dx * (1.0 + 1e-9)) ** 2
    for t in range(len(tris)):
        A, B, C = tris[t]
        n = np.cross(B - A, C - A)
        if not (n @ n) > 0:
            continue                      # zero-area sliver: nothing to mark
        box = grid.clip_span(np.minimum(np.minimum(A, B), C) - 0.5 * grid.dx,
                             np.maximum(np.maximum(A, B), C) + 0.5 * grid.dx)
        if box is None:
            continue
        i0, i1, j0, j1, k0, k1 = box
        ncell = ((i1 - i0 + 1) * (j1 - j0 + 1) * (k1 - k0 + 1))
        if ncell <= budget:
            _mark(grid, best, side, A, B, C, box, thresh)
            continue
        # a huge triangle: walk it in slabs of k (and j) to bound the temporaries
        nj, nk = j1 - j0 + 1, k1 - k0 + 1
        step = max(1, int(math.ceil(ncell / float(budget))))
        if (i1 - i0 + 1) * nj <= budget:            # split along k
            for s in range(0, nk, step):
                _mark(grid, best, side, A, B, C,
                      (i0, i1, j0, j1, k0 + s, min(k0 + s + step, nk + k0) - 1),
                      thresh)
        else:                                        # split along k, then j
            js = max(1, int(math.ceil((i1 - i0 + 1) * nj / float(budget))))
            for s in range(0, nk):
                for q in range(0, nj, js):
                    _mark(grid, best, side, A, B, C,
                          (i0, i1, j0 + q, min(j0 + q + js, nj + j0) - 1,
                           k0 + s, k0 + s), thresh)
    surf = best <= thresh
    return surf, surf & side


def dilate6(m):
    out = m.copy()
    out[1:, :, :] |= m[:-1, :, :]
    out[:-1, :, :] |= m[1:, :, :]
    out[:, 1:, :] |= m[:, :-1, :]
    out[:, :-1, :] |= m[:, 1:, :]
    out[:, :, 1:] |= m[:, :, :-1]
    out[:, :, :-1] |= m[:, :, 1:]
    return out


def bbox_of(mask):
    """Index ranges [(i0,i1),(j0,j1),(k0,k1)] of the nonzero part of a mask.

    The fields are (nz, ny, nx): axis 0 is k, 1 is j, 2 is i.
    """
    i = np.flatnonzero(mask.any(axis=(0, 1)))
    j = np.flatnonzero(mask.any(axis=(0, 2)))
    k = np.flatnonzero(mask.any(axis=(1, 2)))
    if i.size == 0:
        return None
    return [(int(v.min()), int(v.max())) for v in (i, j, k)]


def flood_outside(surf, window, seal=0):
    """Cells inside ``window`` that ``surf`` encloses (the flood cannot reach).

    The window is the part's own bounding box (plus a cell), not the domain:
    a wall that runs across the whole domain would otherwise cut the domain in
    two and the far half would be swallowed as "solid".
    """
    blocked = surf
    if seal > 0:
        blocked = surf.copy()
        for _ in range(int(seal)):
            blocked = dilate6(blocked)
    (i0, i1), (j0, j1), (k0, k1) = window
    free = ~blocked[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
    reach = np.zeros_like(free)
    reach[0, :, :] = free[0, :, :]
    reach[-1, :, :] = free[-1, :, :]
    reach[:, 0, :] = free[:, 0, :]
    reach[:, -1, :] = free[:, -1, :]
    reach[:, :, 0] = free[:, :, 0]
    reach[:, :, -1] = free[:, :, -1]
    n_reach = int(reach.sum())
    while True:
        grow = dilate6(reach) & free
        n = int(grow.sum())
        if n == n_reach:
            return ~reach
        reach, n_reach = grow, n


def mesh_volume(tris):
    """Signed volume enclosed by a closed, outward-oriented triangle set."""
    A, B, C = tris[:, 0], tris[:, 1], tris[:, 2]
    return abs(float(np.einsum("ij,ij->i", A, np.cross(B, C)).sum())) / 6.0


def part_mask(grid, part, base_dir, verbose=True):
    """Occupancy (closed convention) of one part, plus a one-line statistic."""
    t0 = time.time()
    boxes = part_boxes(part)
    if boxes is not None:
        mask = grid.empty()
        for lo, hi in boxes:
            r = grid.index_range(lo, hi)
            if r is not None:
                i0, i1, j0, j1, k0, k1 = r
                mask[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1] = True
        if verbose:
            log("    %-16s box x%d          -> %12s cells  (%.2f s)"
                % (part["name"], len(boxes), f"{int(mask.sum()):,}",
                   time.time() - t0))
        return mask, dict(tris=0, cells=int(mask.sum()))

    tris = part_triangles(part, base_dir)
    if len(tris) == 0:
        raise ValueError("part %r: geometry has no triangles" % part["name"])
    surf, surf_in = rasterise_surface(grid, tris)
    bb = bbox_of(surf)
    if bb is None:
        if verbose:
            log("    %-16s stl %-9s   -> %12s cells  (off the domain)"
                % (part["name"], f"{len(tris):,}", "0"))
        return grid.empty(), dict(tris=len(tris), cells=0)
    # one cell of margin around the bounding box, clipped to the domain
    win = []
    for c in range(3):
        a, b = bb[c]
        win.append((max(a - 1, 0), min(b + 1, grid.n[c] - 1)))
    seal = int(part_geom(part)[1].get("seal", 0))
    deep = flood_outside(surf, win, seal)
    (i0, i1), (j0, j1), (k0, k1) = win
    sw = surf[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
    # a cell is in the part when the nearest triangle puts it inside, or when it
    # is off the surface and the flood cannot get to it
    inside = surf_in[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1] | ((~sw) & deep)
    mask = grid.empty()
    mask[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1] = inside
    if verbose:
        vox = int(mask.sum()) * grid.dx ** 3
        vol = mesh_volume(tris)
        # the lattice body is the mesh volume plus one boundary layer on every
        # side, so a thin plate reads high by design and a LOSS means the flood
        # fill escaped -- that is the leak detector
        ratio = vox / vol if vol > 0 else 0.0
        log("    %-16s stl %-9s   -> %12s cells  vol x%.2f  (%.1f s)"
            % (part["name"], f"{len(tris):,}", f"{int(mask.sum()):,}", ratio,
               time.time() - t0))
        if vol > 0 and ratio < 0.9:
            warn("part %r: the lattice body is %.0f%% of the mesh volume -- the "
                 "surface is probably open (try \"seal\": 1) or the part is "
                 "thinner than one spacing" % (part["name"], 100.0 * ratio))
    return mask, dict(tris=len(tris), cells=int(mask.sum()))


# --------------------------------------------------------------------------
# case -> parts -> deck
# --------------------------------------------------------------------------

def load_case(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        case = json.load(f)
    for k in ("spacing", "domain", "parts"):
        if k not in case:
            raise ValueError("%s: missing %r" % (path, k))
    return case


def build(case, base_dir, verbose=True):
    t0 = time.time()
    dx = float(case["spacing"])
    grid = Grid(case["domain"]["min"], case["domain"]["max"], dx)
    if verbose:
        log("lattice %d x %d x %d = %s cells (%s as bool), spacing %.6g m"
            % (grid.nx, grid.ny, grid.nz, f"{grid.size:,}",
               human(grid.size), dx))

    parts = []
    for spec in case["parts"]:
        p = dict(name=str(spec.get("name", "part%d" % (len(parts) + 1))),
                 role=spec.get("role", "fluid"),
                 geom=spec.get("geom", {}))
        if p["role"] not in ROLE_ID:
            raise ValueError("part %r: role must be one of %s"
                             % (p["name"], list(ROLE_ID)))
        p["itype"] = float(spec.get("itype", ITYPE_BY_ROLE[p["role"]]))
        p["rho"] = float(spec.get("rho", 1000.0))
        p["vel"] = [float(v) for v in spec.get("velocity", (0.0, 0.0, 0.0))]
        p["layers"] = int(spec.get("layers", 0))
        if p["layers"] and p["role"] != "rigid":
            warn("part %r: \"layers\" only applies to a rigid part, ignored"
                 % p["name"])
        if p["role"] == "fluid" and p["itype"] <= 0:
            raise ValueError("part %r: a fluid itype must be > 0" % p["name"])
        if p["role"] == "elastic" and p["itype"] >= -10:
            raise ValueError("part %r: an elastic itype must be < -10" % p["name"])
        if p["role"] == "rigid" and not (-10 <= p["itype"] <= 0):
            raise ValueError("part %r: a rigid itype must be in [-10,0]" % p["name"])
        mask, stat = part_mask(grid, p, base_dir, verbose)
        p["mask"] = mask
        p["raw_cells"] = stat["cells"]
        p["tris"] = stat["tris"]
        parts.append(p)
    for p in parts:
        if p["raw_cells"] == 0:
            warn("part %r produced no cells -- is it outside the domain?"
                 % p["name"])

    # a rigid wall only needs the layers that can feel a deformable particle
    deform = np.zeros(grid.shape, dtype=bool)
    for p in parts:
        if p["role"] in ("fluid", "elastic"):
            deform |= p["mask"]
    for p in parts:
        if p["role"] != "rigid" or p["layers"] <= 0:
            continue
        m = p["mask"]
        keep = m & dilate6(deform)
        for _ in range(p["layers"] - 1):
            keep |= dilate6(keep) & m
        if verbose and int(keep.sum()) != int(m.sum()):
            log("    %-16s rigid shell, %d layers: %s -> %s cells"
                % (p["name"], p["layers"], f"{int(m.sum()):,}",
                   f"{int(keep.sum()):,}"))
        if int(keep.sum()) == 0:
            log("    %-16s rigid shell: no layer touches a fluid or elastic "
                "cell, dropped" % p["name"])
        p["mask"] = keep

    # claimed space, in the order the solver's pair branches care about:
    # an elastic solid owns its volume, a wall owns the rest, the fluid gets
    # what is left.  First part wins inside a role.
    claimed = np.zeros(grid.shape, dtype=bool)
    for role in ("elastic", "rigid", "fluid"):
        for p in parts:
            if p["role"] != role:
                continue
            before = int(p["mask"].sum())
            p["mask"] &= ~claimed
            claimed |= p["mask"]
            if verbose and int(p["mask"].sum()) != before:
                log("    %-16s %s: %s cells hidden by a solid body"
                    % (p["name"], role, f"{before - int(p['mask'].sum()):,}"))
    if verbose:
        log("roles resolved in %.1f s" % (time.time() - t0))
    return grid, parts


def assemble(grid, parts):
    """(rows_by_block, report) with rows ordered fluid | solid | wall."""
    n = len(parts)
    rho_lut = np.full(n + 1, 1000.0)
    itype_lut = np.zeros(n + 1)
    vel_lut = np.zeros((n + 1, 3))
    pid = grid.empty(np.int16)
    for c, p in enumerate(parts):
        pid[p["mask"]] = c + 1
        rho_lut[c + 1] = p["rho"]
        itype_lut[c + 1] = p["itype"]
        vel_lut[c + 1] = p["vel"]
    flat = pid.ravel()
    rows, report = [], {}
    for role in BLOCKS:
        ids = np.array([c + 1 for c, p in enumerate(parts) if p["role"] == role],
                       dtype=np.int16)
        sel = np.flatnonzero(np.isin(flat, ids)) if ids.size else np.zeros(0, np.int64)
        if sel.size == 0:
            rows.append(np.zeros((0, 14)))
            report[role] = dict(n=0)
            continue
        p = flat[sel]
        rho = rho_lut[p]
        r = np.empty((sel.size, 14))
        r[:, 0:3] = grid.positions(sel).T
        r[:, 3] = 0.0                      # pressure: the deck starts at rest
        r[:, 4:7] = vel_lut[p]
        r[:, 7] = rho
        r[:, 8] = rho * grid.dx ** 3       # mass = rho * dx^3
        r[:, 9] = itype_lut[p]
        r[:, 10:13] = r[:, 0:3]            # reference position X0 = x
        r[:, 13] = rho
        rows.append(r)
        report[role] = dict(n=int(sel.size),
                            parts=[q["name"] for q in parts if q["role"] == role])
    return rows, report


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def write_text(path, rows, chunk=200000):
    n = sum(r.shape[0] for r in rows)
    fmt = "\t".join(["%.6g"] * 14) + "\r\n"   # ostream default precision is 6
    t0 = time.time()
    with open(path, "wb") as f:
        for r in rows:
            for s in range(0, r.shape[0], chunk):
                f.write("".join(fmt % tuple(v)
                                for v in r[s:s + chunk].tolist()).encode("ascii"))
    log("wrote %s (%s, %s rows, %.1f s)"
        % (path, human(os.path.getsize(path)), f"{n:,}", time.time() - t0))


def write_sphbin(path, rows, hsml, dt):
    n = sum(r.shape[0] for r in rows)
    ntotal = rows[0].shape[0]
    solnum = rows[1].shape[0]
    allrows = np.concatenate(rows, axis=0)
    mn = allrows[:, 0:3].min(axis=0)
    mx = allrows[:, 0:3].max(axis=0)
    g = (mn[0] - 0.05, mn[1] - 0.05, mn[2] - 0.05,
         mx[0] + 0.05, mx[1] + 0.05, mx[2] + 0.05)
    t0 = time.time()
    with open(path, "wb") as f:
        f.write(SPHBIN_HDR.pack(MAGIC, 1, 128, n, ntotal, solnum, 14, 0, 1,
                                hsml, dt, *g))
        f.write(np.ascontiguousarray(allrows, dtype="<f8").tobytes())
    log("wrote %s (%s, %s rows, %.1f s)"
        % (path, human(os.path.getsize(path)), f"{n:,}", time.time() - t0))


def load_deck(path):
    with open(path, "rb") as f:
        head = f.read(128)
        if head[:8] == MAGIC:
            h = SPHBIN_HDR.unpack(head)
            num, nf = h[3], h[6]
            if nf != 14:
                raise ValueError("%s: %d fields, expected 14" % (path, nf))
            body = np.frombuffer(f.read(), dtype="<f8")
            if body.size != num * 14:
                raise ValueError("%s: %d doubles, header says %d"
                                 % (path, body.size, num * 14))
            return body.reshape(num, 14)
        f.seek(0)
        a = np.loadtxt(f, dtype=np.float64, ndmin=2)
    if a.shape[1] != 14:
        raise ValueError("%s: %d columns, expected 14" % (path, a.shape[1]))
    return a


def lattice_keys(a, grid):
    idx = np.rint((a[:, 0:3] - grid.x0) / grid.dx).astype(np.int64)
    err = np.abs((a[:, 0:3] - grid.x0) - idx * grid.dx).max()
    if err > 1e-3 * grid.dx:
        raise ValueError("deck is not on the %.6g m lattice (off by %.3g m)"
                         % (grid.dx, err))
    return idx[:, 0] + grid.nx * (idx[:, 1] + grid.ny * idx[:, 2])


def check(ref_path, rows, grid):
    ref = load_deck(ref_path)
    got = np.concatenate(rows, axis=0)
    log("check against %s (%s rows)" % (ref_path, f"{ref.shape[0]:,}"))
    if ref.shape[0] != got.shape[0]:
        log("  total    FAIL  reference %s, generated %s (%+d)"
            % (f"{ref.shape[0]:,}", f"{got.shape[0]:,}",
               got.shape[0] - ref.shape[0]))
    st = 0
    for role, sel in (("fluid", lambda t: t > 0),
                      ("elastic", lambda t: t < -10),
                      ("rigid", lambda t: (t >= -10) & (t <= 0))):
        a = ref[sel(ref[:, 9])]
        b = got[sel(got[:, 9])]
        if a.shape[0] != b.shape[0]:
            log("  %-8s FAIL  reference %s, generated %s (%+d)"
                % (role, f"{a.shape[0]:,}", f"{b.shape[0]:,}",
                   b.shape[0] - a.shape[0]))
            st = 2
            continue
        if a.shape[0] == 0:
            log("  %-8s ok    empty in both")
            continue
        ka, kb = lattice_keys(a, grid), lattice_keys(b, grid)
        oa, ob = np.argsort(ka), np.argsort(kb)
        if np.array_equal(ka[oa], kb[ob]):
            dev = np.abs(a[oa, 0:3] - b[ob, 0:3]).max()
            it = np.abs(a[oa, 9] - b[ob, 9]).max()
            log("  %-8s ok    %s particles, same lattice cells, "
                "max |dx| = %.3g m, max |ditype| = %g"
                % (role, f"{a.shape[0]:,}", dev, it))
        else:
            miss = ka.size - np.isin(ka, kb).sum()
            extra = kb.size - np.isin(kb, ka).sum()
            log("  %-8s FAIL  %s in the reference only, %s here only"
                % (role, f"{miss:,}", f"{extra:,}"))
            st = 2
    if st == 0:
        log("check: identical particle sets")
    else:
        log("check: DIFFERENT")
    return st


# --------------------------------------------------------------------------
# box parts -> STL, so the same case can be run through the STL path
# --------------------------------------------------------------------------

def write_ascii_stl(path, tris, name="part"):
    with open(path, "w", encoding="ascii") as f:
        f.write("solid %s\n" % name)
        for A, B, C in tris:
            n = np.cross(B - A, C - A)
            ln = np.linalg.norm(n)
            n = n / ln if ln > 0 else n
            f.write("  facet normal %.9g %.9g %.9g\n    outer loop\n" % tuple(n))
            for v in (A, B, C):
                f.write("      vertex %.9g %.9g %.9g\n" % tuple(v))
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid %s\n" % name)


def dump_stl(case, outdir):
    os.makedirs(outdir, exist_ok=True)
    new = dict(spacing=case["spacing"], domain=case["domain"], parts=[])
    for p in case["parts"]:
        if part_boxes(p) is None:
            raise ValueError("--dump-stl only converts box/boxes parts (%r is %s)"
                             % (p.get("name"), part_geom(p)[0]))
        tris = part_triangles(p, None)
        fn = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(p.get("name", "part"))) + ".stl"
        write_ascii_stl(os.path.join(outdir, fn), tris, str(p.get("name", "part")))
        q = dict((k, v) for k, v in p.items() if k != "geom")
        q["geom"] = {"type": "stl", "file": fn}
        new["parts"].append(q)
        log("  %-16s -> %s (%s triangles)" % (p.get("name"), fn, f"{len(tris):,}"))
    cf = os.path.join(outdir, "case_stl.json")
    with open(cf, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=2)
    log("wrote %s -- the same case with every box part replaced by an STL" % cf)
    return cf


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build an SPH input deck from CAD geometry (STL) or boxes.")
    ap.add_argument("case", help="case file (JSON)")
    ap.add_argument("--out", help="write the 14-column text deck here")
    ap.add_argument("--sphbin", help="write the .sphbin container here")
    ap.add_argument("--check", help="compare against this deck (.sphbin or text)")
    ap.add_argument("--dump-stl", dest="dump_stl", metavar="DIR",
                    help="write every box part as an ASCII STL into DIR")
    ap.add_argument("--hsml", type=float, default=None,
                    help="hsml for the .sphbin header (default: the spacing)")
    ap.add_argument("--dt", type=float, default=1e-5, help="dt for the header")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    try:
        case = load_case(a.case)
        base = os.path.dirname(os.path.abspath(a.case))
        if a.dump_stl:
            dump_stl(case, a.dump_stl)
        grid, parts = build(case, base, verbose=not a.quiet)
        rows, report = assemble(grid, parts)
    except (ValueError, KeyError, OSError) as exc:
        if os.environ.get("GEOM2DECK_DEBUG"):
            raise
        print("error: %s" % exc, file=sys.stderr)
        return 1

    total = sum(r.shape[0] for r in rows)
    if total == 0:
        print("error: no particle ended up in the deck", file=sys.stderr)
        return 1
    hsml = a.hsml if a.hsml is not None else grid.dx
    if not a.quiet:
        for role in BLOCKS:
            r = rows[BLOCKS.index(role)]
            if r.shape[0] == 0:
                log("  %-8s %10s particles" % (role, "0"))
                continue
            log("  %-8s itype %-5g %12s particles  x[%.3f,%.3f] y[%.3f,%.3f] "
                "z[%.3f,%.3f]"
                % (role, float(r[0, 9]), f"{r.shape[0]:,}",
                   r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max(),
                   r[:, 2].min(), r[:, 2].max()))
        log("total %s particles, %s of water, %s of deck"
            % (f"{total:,}",
               f"{report['fluid']['n'] * grid.dx ** 3 * 1000:.1f} L",
               human(total * 14 * 8)))
        if abs(hsml - 0.01) > 1e-12:
            warn("hsml %.6g is not 0.01: rebuild the solver with SPH_HSML=%.6g "
                 "(cuda/src/sph.h) or it will apply the wrong kernel width"
                 % (hsml, hsml))

    if a.out:
        write_text(a.out, rows)
    if a.sphbin:
        write_sphbin(a.sphbin, rows, hsml, a.dt)
    if a.check:
        return check(a.check, rows, grid)
    if not (a.out or a.sphbin or a.dump_stl):
        log("nothing written: pass --out, --sphbin, --check or --dump-stl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
