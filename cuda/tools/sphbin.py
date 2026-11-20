#!/usr/bin/env python3
"""Convert a 14-column text deck to the .sphbin container, and inspect it.

Format (little endian, 128-byte header, then num*nfields float64, row-major):

    off  size  field
      0     8  magic  "SPHBIN\\0\\0"
      8     4  version              = 1
     12     4  header_size          = 128
     16     4  num
     20     4  ntotal   (itype > 0)
     24     4  solnum   (itype < -10)
     28     4  nfields  (14 = input deck, 4 = result: x,y,z,p)
     32     4  step     (0 for input files)
     36     4  flags    (bit1: itype implied by the block split;
                          bit2: payload is float32 instead of float64)
     40     8  hsml     (the solver refuses a deck written for another h)
     48     8  dt
     56    48  gx0,gy0,gz0,gx1,gy1,gz1  (informational; the solver recomputes)
    104    24  reserved (zero)

Input rows: x,y,z,p,vx,vy,vz,rho,mass,itype,X0,X0y,X0z,rho0 (the 14 text columns).
Result rows: x,y,z,p -- itype is not stored; the block split in the header is
what says which rows are fluid, solid and wall.

The deck must be ordered fluid|solid|wall.  How many distinct itype values a
block uses does not matter (a fluid block may carry several), only that the
class never goes backwards.

Usage:
    python sphbin.py make  <deck.txt> <deck.sphbin> [--hsml 0.01] [--dt 1e-5]
    python sphbin.py info  <deck.sphbin>
"""
import sys, struct, os
import numpy as np

MAGIC = b"SPHBIN\0\0"
HDR = struct.Struct("<8sIIIIIIII dd 6d 24x")   # 128 bytes
assert HDR.size == 128

def deck_class(itype):
    """0 = fluid, 1 = solid, 2 = wall (the block order the solver requires)."""
    return np.where(itype > 0, 0, np.where(itype < -10, 1, 2))

def write_header(f, num, ntotal, solnum, nfields, step, flags, hsml, dt, g):
    f.write(HDR.pack(MAGIC, 1, 128, num, ntotal, solnum, nfields, step, flags,
                     hsml, dt, *g))

def read_header(f):
    b = f.read(128)
    (magic, version, hsize, num, ntotal, solnum, nfields, step, flags,
     hsml, dt, *rest) = HDR.unpack(b)
    if magic != MAGIC:
        raise ValueError("not an .sphbin file")
    g = rest[:6]
    return dict(num=num, ntotal=ntotal, solnum=solnum, nfields=nfields,
                step=step, flags=flags, hsml=hsml, dt=dt, grid=g, version=version)

def cmd_make(src, dst, hsml, dt):
    a = np.loadtxt(src, dtype=np.float64, delimiter="\t", ndmin=2)
    n, nf = a.shape
    if nf != 14:
        a = np.loadtxt(src, dtype=np.float64, ndmin=2)
        n, nf = a.shape
    assert nf == 14, f"expected 14 columns, got {nf}"
    it = a[:, 9]
    cls = deck_class(it)
    ntotal = int((cls == 0).sum())
    solnum = int((cls == 1).sum())
    nwall  = int((cls == 2).sum())
    # the kernels require the block layout fluid | solid | wall.  A block may use
    # several itype values (a fluid block can mix 1 and 2), so the test is on the
    # *class*, not on the raw type: the class sequence must never decrease.
    bad = np.flatnonzero(np.diff(cls) < 0)
    if bad.size:
        i = int(bad[0])
        raise SystemExit(f"deck is not ordered as fluid|solid|wall: record {i} has "
                         f"itype {it[i]:g} (class {cls[i]}) after record {i+1} "
                         f"with itype {it[i+1]:g} (class {cls[i+1]})")
    x = a[:, 0]; y = a[:, 1]; z = a[:, 2]
    gx0, gx1 = float(x.min()) - 0.05, float(x.max()) + 0.05
    gy0, gy1 = float(y.min()) - 0.05, float(y.max()) + 0.05
    gz0, gz1 = float(z.min()) - 0.05, float(z.max()) + 0.05
    with open(dst, "wb") as f:
        write_header(f, n, ntotal, solnum, 14, 0, 1, hsml, dt,
                     (gx0, gy0, gz0, gx1, gy1, gz1))
        f.write(np.ascontiguousarray(a, dtype="<f8").tobytes())
    print(f"{src} -> {dst}: num={n} fluid={ntotal} solid={solnum} wall={nwall} "
          f"({os.path.getsize(dst)/1048576:.1f} MB)")

def cmd_info(path):
    with open(path, "rb") as f:
        h = read_header(f)
        dt = "<f4" if (h["flags"] & 4) else "<f8"
        data = np.frombuffer(f.read(), dtype=dt)
    print(f"{path}: version={h['version']} num={h['num']} ntotal={h['ntotal']} "
          f"solnum={h['solnum']} nfields={h['nfields']} step={h['step']} "
          f"hsml={h['hsml']} dt={h['dt']}")
    print("  grid:", [round(v, 4) for v in h["grid"]])
    expect = h["num"] * h["nfields"]
    print(f"  records={data.size} (expect {expect}) {'OK' if data.size == expect else 'MISMATCH'}")
    if h["nfields"] == 14:
        it = data.reshape(-1, 14)[:, 9]
        print("  itype counts:", int((it > 0).sum()), int((it < -10).sum()),
              int(((it >= -10) & (it <= 0)).sum()))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "make":
        hsml = float(sys.argv[sys.argv.index("--hsml") + 1]) if "--hsml" in sys.argv else 0.01
        dt = float(sys.argv[sys.argv.index("--dt") + 1]) if "--dt" in sys.argv else 1e-5
        cmd_make(sys.argv[2], sys.argv[3], hsml, dt)
    elif sys.argv[1] == "info":
        cmd_info(sys.argv[2])
    else:
        print(__doc__); sys.exit(1)
