#!/usr/bin/env python3
"""Compare result<N>.sphbin files of two run directories.

usage: cmp.py <dirA> <dirB> [N ...]
Prints, per file: max|dx|,|dy|,|dz| over all particles, max|dp| over fluid,
max|d trace(Pia)| over solids, and whether the files are byte-identical.
"""
import struct, sys, os, glob, hashlib
import numpy as np


def read(path):
    with open(path, "rb") as f:
        h = f.read(128)
        assert h[:8] == b"SPHBIN\0\0", path
        ver, hs, num, ntotal, solnum, nfields, step, flags = struct.unpack("<8I", h[8:40])
        dt = "<f4" if (flags & 4) else "<f8"
        a = np.frombuffer(f.read(), dtype=dt).reshape(num, nfields)
    return dict(num=num, ntotal=ntotal, solnum=solnum, step=step, a=a)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    A, B = sys.argv[1], sys.argv[2]
    idx = [int(v) for v in sys.argv[3:]]
    if not idx:
        idx = sorted(int(os.path.basename(p)[6:-7]) for p in glob.glob(os.path.join(A, "result*.sphbin")))
    for n in idx:
        pa, pb = os.path.join(A, f"result{n}.sphbin"), os.path.join(B, f"result{n}.sphbin")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"result{n}: missing"); continue
        a, b = read(pa), read(pb)
        nt, ns = a["ntotal"], a["solnum"]
        d = np.abs(a["a"] - b["a"])
        dx = d[:, :3].max()
        dpf = d[:nt, 3].max() if nt else 0.0
        dps = d[nt:nt + ns, 3].max() if ns else 0.0
        ident = sha(pa) == sha(pb)
        print(f"result{n} (step {a['step']}): max|dx|={dx:.3e} m  fluid max|dp|={dpf:.3e} Pa  "
              f"solid max|dtr|={dps:.3e} Pa  {'BYTE-IDENTICAL' if ident else ''}")


if __name__ == "__main__":
    main()
