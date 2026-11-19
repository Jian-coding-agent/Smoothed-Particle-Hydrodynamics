import struct, sys
import numpy as np

def read_sphbin(path):
    with open(path, "rb") as f:
        h = f.read(128)
        magic = h[:8]
        assert magic == b"SPHBIN\0\0", magic
        ver, hs, num, ntotal, solnum, nfields, step, flags = struct.unpack("<8I", h[8:40])
        dt = "<f4" if (flags & 4) else "<f8"
        data = np.frombuffer(f.read(), dtype=dt)
        assert data.size == num * nfields, (data.size, num, nfields)
        return dict(num=num, ntotal=ntotal, solnum=solnum, nfields=nfields,
                    step=step, flags=flags, a=data.reshape(num, nfields))

def read_text(path):
    return np.loadtxt(path, dtype=np.float64, delimiter="\t", ndmin=2)

def main(binpath, textpath, label):
    b = read_sphbin(binpath)
    t = read_text(textpath)
    n = min(len(t), b["num"])
    print(f"[{label}] bin: num={b['num']} nfields={b['nfields']} step={b['step']}  "
          f"text rows={len(t)}  compared={n}")
    dx = np.abs(b["a"][:n, 0] - t[:n, 0]).max()
    dy = np.abs(b["a"][:n, 1] - t[:n, 1]).max()
    dz = np.abs(b["a"][:n, 2] - t[:n, 2]).max()
    dp = np.abs(b["a"][:n, 3] - t[:n, 3]).max()
    print(f"  max|dx|={dx:.3e} max|dy|={dy:.3e} max|dz|={dz:.3e} max|dp|={dp:.3e}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cmp")
