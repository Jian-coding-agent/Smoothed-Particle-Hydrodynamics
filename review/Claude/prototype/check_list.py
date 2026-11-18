import sys, numpy as np
from scipy.spatial import cKDTree
inp, dump_ok, dump_shift, r_base, r_static, r_dyn = sys.argv[1:7]
NT, NS = 101250, 60320
X0 = np.loadtxt(inp, usecols=(10, 11, 12))[NT:NT + NS]
pairs = cKDTree(X0).query_pairs(0.03 + 1e-9, output_type="ndarray")
g = X0[pairs[:, 0]] - X0[pairs[:, 1]]
n = np.rint((g * g).sum(1) / 1e-4).astype(int)
def keyset(p):  # directed pairs as int64 keys (p, q) in solid-local indices
    return np.concatenate([p[:, 0] * NS + p[:, 1], p[:, 1] * NS + p[:, 0]])
inside = keyset(pairs[n <= 8]); edge = keyset(pairs[n == 9])
def load(d):
    st = np.fromfile(d + "/solidlist_start.bin", dtype=np.int32)
    q = np.fromfile(d + "/solidlist_q.bin", dtype=np.int32) - NT
    p = np.repeat(np.arange(NS), np.diff(st))
    return p.astype(np.int64) * NS + q, np.diff(st)
for name, d in (("unshifted", dump_ok), ("shifted +0.025 y", dump_shift)):
    keys, cnt = load(d)
    miss = np.setdiff1d(inside, keys); extra = np.setdiff1d(keys, inside)
    bad = np.unique(miss // NS)
    print(f"{name:18s}: list={len(keys)} inside-pairs missing={len(miss)} "
          f"(solids affected={len(bad)}), extra={len(extra)} of which knife-edge={np.isin(extra, edge).sum()}")
cols = (3,)
pb = np.loadtxt(r_base, usecols=cols)[NT:NT + NS]
ps = np.loadtxt(r_static, usecols=cols)[NT:NT + NS]
pd = np.loadtxt(r_dyn, usecols=cols)[NT:NT + NS]
print(f"solid trace(Pia) after 1 step: |unshifted| median={np.median(abs(pb)):.4g} max={abs(pb).max():.4g} Pa")
print(f"  shifted, SOLDDYN=1 (cell walk) vs unshifted: max|diff|={abs(pd-pb).max():.4g} Pa")
print(f"  shifted, default static list  vs unshifted: max|diff|={abs(ps-pb).max():.4g} Pa, "
      f"median={np.median(abs(ps-pb)):.4g}, solids off by >1 kPa: {(abs(ps-pb)>1e3).sum()}")
