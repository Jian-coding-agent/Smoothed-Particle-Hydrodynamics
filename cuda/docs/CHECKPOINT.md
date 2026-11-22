# Checkpoint / restart

A run can be stopped and continued from its last checkpoint, and the continuation
is **bit-identical** to a straight run of the same length — in both precisions.

> **Scope: the single-GPU path only.**  `SPH_CKPT` / `SPH_CKPT_IN` / `--resume`
> with `SPH_RANKS` set are refused at start-up: a decomposed run also has ghost
> copies and a halo that belongs to other ranks, so a restart would have to save
> the decomposition and every rank's state together.  Use the ordinary path
> (`SPH_RANKS` unset) for a checkpointed run.

```bat
:: one long run, checkpoint every 5000 steps and again at the end
set SPH_MAXSTEP=200000
set SPH_CKPT=dam.chk
set SPH_CKPT_EVERY=5000
sph_cuda.exe cases\dam_1203k.sphbin --fp32

:: continue it (same deck, same precision, same switches)
set SPH_MAXSTEP=100000
sph_cuda.exe cases\dam_1203k.sphbin --fp32 --resume=dam.chk
```

`SPH_MAXSTEP` counts the steps of **this** invocation, `--resume` supplies the
absolute step number, and result frames keep the names a straight run would have
given them (`result<N>.sphbin`, `step = N * SPH_OUTCADENCE`).

## A `result*.sphbin` is not a restart state

Binary output removes none of the reasons a result file cannot be resumed.  It
holds four values per particle (`x y z ppp`) out of the twenty the solver carries,
and the loader rejects it outright:

```
[input] result0.sphbin: unsupported container (version=1 header=128 nfields=4)
[ckpt]  result0.sphbin: not a checkpoint (bad magic or short header) -- a
        result*.sphbin is not a restart state
```

Even a 14-column dump would not be enough: three of the arrays a step reads are
not "current state" in the usual sense.

## What is actually carried across a step

| array | why it is state |
|---|---|
| `x`, `vx`, `rho` | the integration variables (`v += dt·a`, then `x += dt·v`; `rho += dt·drhodt`) |
| `SurF` | **one step of history**: `k_prepare` copies it into `SurFD`, and the next step's `pass1` selects the artificial viscosity `eta` from `SurFD[max(p,q)]` |
| `A` | gradient-correction matrix, built **once** by `GraCorrec`; its neighbour search runs in the live positions while its weights use `X0`, so recomputing it after the fluid has moved gives a different matrix |
| `vecn` | surface normals, built **once** by `NorVector` from the *initial* `rho`; recomputing them from a deformed state changes the fluid–solid coupling |
| `u`, `xc`, `rhoc` | **fp32 only**: the displacement form `x = X0 + u` for solid particles and the Kahan compensation of the `x`/`rho` sums. Neither can be reconstructed from `x` (the compensation is sub-ulp by construction) |

Everything else is written before it is read inside one step and is therefore
*not* stored: `p`, `irho` (from `rho`, in `k_prepare`), `drhodt`, `dxvdt`,
`viscos`, `SurF`→`SurFD`, `SForce`, `PIJW`, `F`, `DFdt`, `Pia`, and the whole
cell list.  `X0`, `rho0`, `mass` and `itype` belong to the deck, not to the run:
they are not stored, and a fingerprint of them is (see below).

This list is the reason the feature exists in the solver rather than in a script:
`SurF` and the two init-only arrays are exactly what a naive "dump x, vx, rho"
restart gets wrong, and it gets them wrong quietly.

## Container (`SPHCHK`)

Little endian, 128-byte header, then the arrays back to back.  All values are in
the precision of the run (`float32` or `float64`).

| off | size | field |
|---:|---:|---|
| 0 | 8 | magic `SPHCHK\0\0` |
| 8 | 4 | version = 2 (version 1 is the pre-merge single-GPU build; see below) |
| 12 | 4 | header size = 128 |
| 16 | 4 | num |
| 20 | 4 | ntotal (fluid) |
| 24 | 4 | solnum (solid) |
| 28 | 4 | narray (6 fp64, 9 fp32) |
| 32 | 4 | flags: bit0 fp32 data, bit1 compensated arrays, bit2 solid clamp, bit3 static solid list, bit4 fluid list, bit5 factored `pass1`, bit6 guarded z-reflection |
| 36 | 4 | `GraCorrec` fallbacks in the writing run (C8) |
| 40 | 8 | step (completed steps) |
| 48 | 8 | hsml |
| 56 | 8 | dt |
| 64 | 8 | solid clamp `z` |
| 72 | 8 | solid clamp `y0` |
| 80 | 8 | solid clamp `y1` |
| 88 | 8 | deck fingerprint (FNV-1a over `mass`, `itype`, `X0`, `rho0`) |
| 96 | 8 | payload bytes |
| 104 | 8 | c0 |
| 112 | 8 | lamd |
| 120 | 8 | muyou |

Array order and length (per particle): `x` (3), `vx` (3), `rho` (1), `SurF` (1),
`A` (9), `vecn` (3), then fp32 only: `u` (3), `xc` (3), `rhoc` (1).

| | fp64 | fp32 |
|---|---:|---:|
| bytes/particle | 160 | 108 |
| 1 203 508 particles | 183.6 MiB | 124.0 MiB |
| 237 298 particles (the text deck) | 36.2 MiB | 24.4 MiB |
| a `result*.sphbin` of the same run | 36.7 MiB | 18.4 MiB |

Writing 124 MiB takes about 0.1 s here (wall 1.51 s → 1.58 s over 300 steps),
i.e. the cost is the file, not the dump.  The file is written as `<path>.tmp`
and renamed over `<path>`, so an interrupted write leaves the previous
checkpoint intact.

Inspect one:

```bat
python tools\sphbin.py ckpt dam.chk
```

## What a resume checks, and refuses

A resume must be handed the same problem, so the loader compares the header with
the current run and stops with a specific message instead of continuing into
different physics:

* `num` / `ntotal` / `solnum` against the deck;
* precision — an fp64 checkpoint cannot resume an fp32 run (and vice versa);
* `hsml`, `dt` against this build;
* `c0`, `lamd`, `muyou` — a rebuild with different constitutive constants cannot
  continue an old checkpoint (nothing else in the header would notice);
* the solid clamp `SPH_SOLDBOX` (on/off and the three bounds);
* `SPH_SOLDDYN` / `SPH_FLUIDLIST` / `SPH_LITERALFLUID` — the optimised paths are
  bit-identical to their alternatives, but a run must not change them halfway;
* `SPH_REFLECT` — the guarded z-reflection and 1.cpp's unconditional flip differ
  for a particle already moving down at `z > 1`;
* the deck fingerprint — resuming with a different deck is refused (the message
  prints both hashes, so "wrong deck" and "different build" can be told apart);
* the payload size and the file size (a truncated checkpoint is refused).

`SPH_LOADA` is ignored on a resume (A comes from the checkpoint).

**Version 1 checkpoints** were written by the pre-merge single-GPU build.  That
build had no `SPH_REFLECT` switch and always used 1.cpp's unconditional flip, so
a v1 checkpoint is resumed with `SPH_REFLECT=flip`; the loader refuses it under
the new default with a message saying exactly that.  Version 2 stores the
reflection mode in flags bit 6.  Everything else in the v1 layout is unchanged,
so a v1 file is otherwise a valid restart state.

**Keep one build for a segmented run.**  The checkpoint belongs to the run that
wrote it, and the deck fingerprint is computed by the solver, so a rebuild can
change it: on 2026-11-20 a checkpoint written by the 07:13 build was refused by a
07:20 build of the same working tree, on an unchanged deck (the fingerprint went
from `0xa7d7b56745ef174f` to `0x79718dcfc3dca699`; the latter matches an
independent hash of the deck's mass/itype/X0/rho0 columns).  Nothing else in the
header noticed — which is why the guard exists, but it also means a rebuild
halfway through a long chained run can cost you the chain.

## Verified

RTX 5080, `dt = 1e-5`, 1 203 508-particle deck and the 237 298-particle text
deck.  Every "chained" run is compared **byte for byte** with the checkpoint of
the straight run:

| run | chained | straight | result |
|---|---|---|---|
| fp64, 1.2 M | 150 + resume 150 | 300 | SHA256 equal (183.6 MiB) |
| fp32, 1.2 M | 150 + resume 150 | 300 | SHA256 equal (124.0 MiB) |
| fp32, 1.2 M | 100 + 100 + 100 | 300 | SHA256 equal |
| fp64, 237 k | 20 + resume 20 | 40 | SHA256 equal |
| fp64, 237 k | 150 + resume 150 | 300 | SHA256 equal (merged single/multi build) |
| fp32, 237 k | 75 + resume 75 | 150 | SHA256 equal (merged single/multi build) |
| fp32, 1.2 M, cadence 100 | 150 + resume 150 | 300 | `result2.sphbin` equal |
| fp32, 1.2 M, 20 steps | with and without the new code | 20 | `result0/1.sphbin` identical |
| fp32, 1.2 M, S1 default | 500 + resume 501 | 1001 | `result1`/`result2` SHA256 equal |

The last row is the regression guard: adding the checkpoint code does not change
a single bit of a fresh run's output.  The resume costs one init (0.15–0.21 s,
including the static solid list) and then runs at the same ms/step.

## Notes

* The neighbour grid still comes from the **deck**, not from the checkpoint's
  positions.  A resumed run therefore bins into exactly the grid the original run
  used; recomputing it from the current positions would change the neighbour
  enumeration order and with it the last bits of every sum.
* Keep `SPH_OUTCADENCE` fixed across segments, otherwise the frame index
  (`step / cadence`) collides or skips.  `SPH_OUTCADENCE=0` disables result
  frames entirely (a checkpoint-only run).
* The two `step` fields mean different things, on purpose: a result's header
  says `N * cadence` but the frame holds the state **after** `N*cadence + 1`
  steps (the writer runs at the end of the iteration), while a checkpoint's
  `step` is the exact number of completed steps.  Do not compare them directly.
* The static solid pair list is rebuilt at every start (0.2 s at 1.2 M).  It is
  a function of `X0` and `A` only, so it cannot go stale; it is not stored.
* `SPH_STEP0=<n>` starts a fresh run at an absolute step number (it only affects
  the result file names and the header `step`; the resume path takes the number
  from the checkpoint).
* Not implemented: automatic restart after a crash (the periodic checkpoint is
  the mechanism — set `SPH_CKPT_EVERY`), rotation between two files, and
  compression.

## Latent bug found on the way

`build_cells_for()` histogrammed the positions of the struct it was given and
then scattered the **live** `g.x` into that histogram:

```cpp
launch_cell_hist(s, ...);          // s.x, i.e. X0 for the solid-list build
launch_cell_scatter(g, ...);       // g.x  <-- wrong configuration
```

The regression is recent and is *not* in the CPU reference (which rebuilds one
grid per step from the current `x` and has no second configuration) and not in
the A3/C1 commit either (there the call was `launch_cell_scatter(s, ...)`, which
is why `7eb5323` could report the 4 844 544-pair list and a rigid-translation
error of 5.8e-8 Pa).  It came in with the runtime-sizing revision `a106b7b`,
when `build_cells()` was split into `build_cells_for()` + a wrapper and this one
call kept `g`.

That is invisible while `x == X0` at start-up, which is the case for the 1.2 M
deck (`max |x - X0| = 0`).  It is not invisible for the text deck (`max |x - X0|`
= 24 mm): one particle in a different cell shifts every following cell's range,
the permutation `sorted` becomes garbage, and the static solid list silently
loses pairs — measured 4 363 227 pairs / ELL width 94 instead of the 4 844 544 /
width 92 that `README.md` documents.  A resume exposes it immediately, because
the restored `x` is never equal to `X0`.

Fixed by scattering `s`.  The 1.2 M case is bit-identical before and after the
fix (verified), and the text deck is back to its documented list.
