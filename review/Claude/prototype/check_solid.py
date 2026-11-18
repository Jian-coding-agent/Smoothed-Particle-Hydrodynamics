"""Reference-configuration checks for the solid in input.dat.

1. Brute-force count of solid/solid pairs on the X0 lattice, split by
   n = round((Dr/0.01)^2): n<=8 strictly inside the support, n==9 knife edge.
2. GraCorrec determinant per solid particle, using the CPU's pair set
   (current positions at t=0, r^2 < (3*hsml)^2), to see whether any particle
   falls back to A = I (|Da| < 1e-6).
"""
import sys
import numpy as np
from scipy.spatial import cKDTree

path = sys.argv[1]
d = np.loadtxt(path)
NT, NS = 101250, 60320
sol = slice(NT, NT + NS)
x = d[sol, 0:3]
X0 = d[sol, 10:13]
m = d[sol, 8]
rho0 = d[sol, 13]

hsml = 0.01
h = 1.5 * hsml
coe = 42.0 / (256.0 * h * h * h * h * np.pi)
R2 = (3.0 * hsml) * (3.0 * hsml)

# ---- 1. reference-configuration pairs (directed) ----
tree0 = cKDTree(X0)
pairs0 = tree0.query_pairs(0.03 + 1e-9, output_type="ndarray")
g = X0[pairs0[:, 0]] - X0[pairs0[:, 1]]
n = np.rint((g * g).sum(1) / 1e-4).astype(int)
inside = int((n <= 8).sum()) * 2
edge = int((n == 9).sum()) * 2
print(f"[ref] directed solid pairs: n<=8 (inside) = {inside}, n==9 (knife edge) = {edge}")
cnt = np.bincount(np.concatenate([pairs0[n <= 8, 0], pairs0[n <= 8, 1]]), minlength=NS)
print(f"[ref] partners per solid (inside): min={cnt.min()} max={cnt.max()} mean={cnt.mean():.2f}")

# ---- 2. GraCorrec with the CPU pair set (current x at t=0) ----
tree = cKDTree(x)
cand = tree.query_pairs(0.03 + 1e-9, output_type="ndarray")
dd = x[cand[:, 0]] - x[cand[:, 1]]
r2 = dd[:, 0] * dd[:, 0] + dd[:, 1] * dd[:, 1] + dd[:, 2] * dd[:, 2]   # same order as 1.cpp
cand = cand[r2 < R2]
print(f"[cpu] solid/solid pairs in the t=0 list (undirected): {len(cand)}")

def dwdx_vec(gx, Dr):
    q = Dr / h
    w = np.where((q >= 0) & (q <= 2),
                 coe * ((2 - q) ** 4 - 2 * (1 + 2 * q) * (2 - q) ** 3), 0.0)
    return w[:, None] * (gx / Dr[:, None])

a = np.zeros((NS, 3, 3))
for (i, j) in ((0, 1), (1, 0)):
    p, q = cand[:, i], cand[:, j]
    gX = X0[p] - X0[q]
    Dr = np.sqrt((gX * gX).sum(1))
    gw = dwdx_vec(gX, Dr)                      # dwdx(X0_p - X0_q, Dr)
    w = (m[q] / rho0[q])[:, None, None]
    contrib = w * (-gX)[:, :, None] * gw[:, None, :]   # (X0_q - X0_p)_r * g_k
    np.add.at(a, p, contrib)
Da = np.linalg.det(a)
print(f"[gra] det(a): min={Da.min():.4g} max={Da.max():.4g}; |Da|<1e-6: {(np.abs(Da) < 1e-6).sum()}")
asym = np.abs(a - a.transpose(0, 2, 1)).max()
print(f"[gra] max |a - a^T| = {asym:.3g}  (a is symmetric up to rounding -> a*A^T = I)")
