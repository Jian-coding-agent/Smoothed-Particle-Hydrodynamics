# Binary decks, runtime sizing, binary results

This revision makes one `sph_cuda.exe` serve **any** deck:

* the particle counts, the fluid/solid/wall split and the neighbour grid are read
  from the deck at start-up (no `SPH_NUM` / `SPH_NTOTAL` / `SPH_SOLNUM` macros,
  no fixed grid box);
* decks are stored in a self-describing binary container; the legacy 14-column
  text deck is still accepted as a fallback;
* results are written as binary `result<N>.sphbin`.

Only the physical scheme constants (`hsml`, `dt`, `c0`, `lamd`, `muyou`, `MaxV`)
stay compile-time.

## Container format (`.sphbin`)

Little endian, 128-byte header, then `num * nfields` values, row-major.

| off | size | field |
|---:|---:|---|
| 0 | 8 | magic `SPHBIN\0\0` |
| 8 | 4 | version = 1 |
| 12 | 4 | header size = 128 |
| 16 | 4 | num (total particles) |
| 20 | 4 | ntotal (fluid, itype > 0) |
| 24 | 4 | solnum (solid, itype < -10) |
| 28 | 4 | nfields |
| 32 | 4 | step (0 for decks) |
| 36 | 4 | flags: bit0 = itype present, bit1 = result file, bit2 = float32 data |
| 40 | 8 | hsml |
| 48 | 8 | dt |
| 56 | 48 | grid x0,y0,z0,x1,y1,z1 (informational; the solver recomputes it) |
| 104 | 24 | reserved |

* **deck**: `nfields = 14`, float64: `x y z p vx vy vz rho mass itype X0x X0y X0z rho0`
  (the same order as the text columns).  The converter counts `itype` and checks
  that the blocks are ordered fluid | solid | wall.
* **result**: `nfields = 4`: `x y z ppp`, where `ppp` is the fluid pressure for
  `i < ntotal`, `Pia11+Pia22+Pia33` for the solid block and 0 for rigid
  particles.  `itype` is implied by the header counts.  float64 by default;
  `SPH_OUT_F32=1` writes float32 and sets flags bit 2.

A 1,203,508-particle result is 38.5 MB in float64 and 19.3 MB in float32; the
old 4-column text file over the 908,308-row subset was 25.6 MB.

## Convert a text deck

```bat
python tools\sphbin.py make input_1203k.dat input_1203k.sphbin --hsml 0.01 --dt 1e-5
python tools\sphbin.py info input_1203k.sphbin
```

## Run

The file name is arbitrary; it is the first argument:

```bat
sph_cuda.exe cases\dam_1203k.sphbin --fp64
sph_cuda.exe cases\dam_1203k.sphbin --fp32
set SPH_MAXSTEP=501
set SPH_OUT_F32=1          :: float32 results (half the size)
```

Text decks keep working (`sph_cuda.exe input.dat`); the loader sniffs the magic.

## What changed in the code

* `SphPtrsT` carries a `SphDims` (counts, grid origin/dims).  The host fills it
  from the deck once; every kernel reads it.  Allocation, launch geometry and
  every kernel loop are sized from it.
* The neighbour grid is derived from the data
  (`min - (3h + 0.02)`, `max + (3h + 0.02)`), so a deck can never be binned into
  a stale box (that failure cost 5.4x on the CPU reference: 69% of the particles
  were clamped into boundary cells).
* `k_cell_scan` uses a chunked prefix sum and a 512-thread block, so it works
  for any grid size (the old one was a `static_assert` at 32,768 cells).
* The old test-only solid clamp is gone and the fluid reflects at `z > 1.0`,
  matching the new-case `1.cpp`.

## Measured on the new case

RTX 5080, 1,203,508 particles (552,500 fluid / 188,000 solid / 463,008 rigid),
53,613,008 neighbour pairs at t = 0, 501 steps:

| | fp64 | fp32 |
|---|---:|---:|
| ms/step | 21.15 | 4.27 |
| init (binary deck) | 0.175 s | 0.121 s |
| init (text deck) | 0.62 s | — |
| t = 0 pairs | 53,613,008 | 57,195,270 |
| step-500 pairs | 57,479,754 | 58,236,100 |
| static solid list | 14,858,832 pairs, ELL width 92 | same |

Dynamic sizing costs nothing measurable: the previous fixed-case build measured
21.11 ms/step (fp64) and 4.265 ms/step (fp32) on the same deck.

Results against the CPU reference after 500 steps: `x/y/z` identical at the
printed precision, `p` max 5e-3 Pa (fp64, 2e-6 relative) and 0.15 Pa (fp32).
The difference is summation order: the runtime grid (105x50x38, origin -0.08)
differs from the earlier hand-written box (104x49x37, origin -0.05), so the
neighbour enumeration order changes while the pair set is identical.

## Limitations / notes

* Result cadence is fixed at 500 steps and every particle is written (the old
  908,308-row bound and the `SurFD <= 2.5` filter were case-specific).
* The A3 solid list is still built once from `X0`.  On the old 237k deck its
  pair count varies by ~0.1% between runs because thousands of reference pairs
  sit exactly at `r = 3h` and the gradient-correction matrix is built with
  atomics; those pairs carry zero kernel weight.  The new deck's list is stable.
* The boundary conditions are those of the new case, so running the old 237k
  deck through this build does **not** reproduce the old validation numbers in
  `README.md` / `docs/DESIGN.md`; that table describes the pre-2026-11-19
  fixed-case build.
