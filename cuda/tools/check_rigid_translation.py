#!/usr/bin/env python3
"""Regression test for the A3 static solid pair list (review finding C1).

The list is built once, from the *reference* configuration, and must therefore
be a function of X0 alone.  A rigid translation of the solid -- which changes
the current positions but no reference distance -- must not change it, and must
not change any stress by more than rounding noise.

Before the fix, the list was searched in a cell list binned from the current
positions: a +25 mm rigid offset silently dropped 235 296 in-support pairs and
made trace(Pia) wrong by up to 5.3 MPa on 19 760 solids, while the deck is a
restart that can carry such an offset.

Usage:
    python check_rigid_translation.py [--exe PATH] [--input input.dat]
                                      [--offset 0.025] [--fp32]

Run it from the repository root (the deck is 21 MB and is copied twice, into a
scratch directory next to the current directory).

Exit code 0 = pass, 1 = regression, 2 = usage/IO error.
"""
import argparse
import array
import os
import shutil
import subprocess
import sys
import tempfile

SPH_NUM = 237298
SPH_NTOTAL = 101250
SPH_SOLNUM = 60320
# 1-based lines 101251..161570 are the solid particles; column 2 (0-based 1) is y
SOLID_FIRST_ROW = SPH_NTOTAL          # 0-based row index of the first solid
SOLID_LAST_ROW = SPH_NTOTAL + SPH_SOLNUM - 1


def read_deck(path):
    with open(path) as f:
        return f.read().splitlines()


def write_shifted(src_lines, dst_path, dy):
    """X0 unchanged, current y (column 1) shifted by dy for the solid rows."""
    out = []
    for i, line in enumerate(src_lines):
        if SOLID_FIRST_ROW <= i <= SOLID_LAST_ROW:
            tok = line.split()
            tok[1] = repr(float(tok[1]) + dy)
            out.append(" ".join(tok))
        else:
            out.append(line)
    with open(dst_path, "w") as f:
        f.write("\n".join(out) + "\n")


def run(exe, workdir, fp32):
    env = dict(os.environ)
    env["SPH_MAXSTEP"] = "1"
    env["SPH_DUMPSTEP"] = "1"           # writes gpu_Pia*.bin after the first step
    env.pop("SPH_PROFILE", None)
    env.pop("SPH_BENCH", None)
    env.pop("SPH_SOLDDYN", None)
    env.pop("SPH_LOADA", None)
    cmd = [exe] + (["--fp32"] if fp32 else ["--fp64"])
    p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit(f"{exe} failed with {p.returncode}")
    npair = None
    for line in p.stdout.splitlines():
        if line.startswith("[A3] solid pair list:"):
            npair = int(line.split(":")[1].split()[0])
    if npair is None:
        raise SystemExit("could not parse the [A3] line; was the list enabled?")
    return npair, p.stdout


def load_trace(workdir):
    comps = []
    for name in ("Pia11", "Pia22", "Pia33"):
        a = array.array("d")
        with open(os.path.join(workdir, "gpu_%s.bin" % name), "rb") as f:
            a.frombytes(f.read())
        if len(a) != SPH_NUM:
            raise SystemExit("gpu_%s.bin has %d entries, expected %d" % (name, len(a), SPH_NUM))
        comps.append(a)
    return [comps[0][i] + comps[1][i] + comps[2][i]
            for i in range(SPH_NTOTAL, SPH_NTOTAL + SPH_SOLNUM)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join("..", "build", "sph_cuda.exe"))
    ap.add_argument("--input", default="input.dat")
    ap.add_argument("--offset", type=float, default=0.025, help="rigid y offset in metres")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-3, help="max |delta trace(Pia)| in Pa")
    ap.add_argument("--keep", action="store_true", help="keep the scratch directories")
    args = ap.parse_args()

    exe = os.path.abspath(args.exe)
    deck = os.path.abspath(args.input)
    if not os.path.isfile(exe):
        raise SystemExit("executable not found: %s" % exe)
    if not os.path.isfile(deck):
        raise SystemExit("input deck not found: %s" % deck)

    # The scratch tree is created next to the current directory rather than in
    # %TEMP%: the solver writes its dump files relative to its working
    # directory, and on locked-down Windows setups a child process may not be
    # allowed to write into the system temp directory (it fails silently).
    scratch = tempfile.mkdtemp(prefix="sph_rigid_", dir=os.getcwd())
    a_dir = os.path.join(scratch, "base")
    b_dir = os.path.join(scratch, "shifted")
    os.makedirs(a_dir)
    os.makedirs(b_dir)
    try:
        shutil.copyfile(deck, os.path.join(a_dir, "input.dat"))
        write_shifted(read_deck(deck), os.path.join(b_dir, "input.dat"), args.offset)

        n0, out0 = run(exe, a_dir, args.fp32)
        n1, out1 = run(exe, b_dir, args.fp32)
        print("[rigid] unshifted list: %d pairs" % n0)
        print("[rigid] +%.3f m     list: %d pairs" % (args.offset, n1))

        t0 = load_trace(a_dir)
        t1 = load_trace(b_dir)
        err = sorted(abs(b - a) for a, b in zip(t0, t1))
        mx = err[-1]
        p99 = err[int(0.99 * (len(err) - 1))]
        # magnitude of the reference itself, for scale
        ref = sorted(abs(v) for v in t0)
        print("[rigid] |trace(Pia)| reference: median %.3g, p99 %.3g Pa"
              % (ref[len(ref) // 2], ref[int(0.99 * (len(ref) - 1))]))
        print("[rigid] delta trace(Pia): max %.3g, p99 %.3g Pa" % (mx, p99))

        ok = True
        if n0 != n1:
            print("[rigid] FAIL: the static pair list depends on the current positions "
                  "(%d vs %d pairs)" % (n0, n1))
            ok = False
        if mx > args.tol:
            print("[rigid] FAIL: stress changed by %.3g Pa (tolerance %.3g)"
                  % (mx, args.tol))
            ok = False
        print("[rigid] %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        if args.keep:
            print("[rigid] scratch: %s" % scratch)
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
