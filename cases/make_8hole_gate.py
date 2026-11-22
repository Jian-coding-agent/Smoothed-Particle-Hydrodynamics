#!/usr/bin/env python3
"""Write cases/8hole_gate.json -- the 8-hole gate case as a geometry spec.

This is the case ``dist/Input_3D/1.cpp`` hard-codes with 1300 lines of loop
nests.  The same information, as parameters:

    a 0.01 m lattice over a 3.63 x 3.37 x 1.03 m tank
    water in x[0.01,0.70] y[0.01,3.30] z[0.01,0.95]        (70 x 330 x 95)
    two 0.10 m thick gate slabs at x = 1.51 and 2.41, each
       9 piers 0.10 m wide, 0.40 m apart  (leaves 8 openings)
       8 upper beams over the openings, at z[0.61,0.70] and [0.91,1.00]
    one solid dam at x[3.51,3.60]
    four rigid walls: floor, upstream face, downstream face, left side,
       each 4 particle layers thick and lying outside the water column

Run
    python cases/make_8hole_gate.py            # writes cases/8hole_gate.json
"""
import json
import os

DX = 0.01
DOMAIN = {"min": [-0.03, -0.03, -0.03], "max": [3.60, 3.34, 1.00]}

WATER = [0.01, 0.01, 0.01, 0.70, 3.30, 0.95]

# gate: 9 piers, 8 bays.  pier n spans y = 0.01+0.40n .. 0.10+0.40n
PIERS = [(0.01 + 0.40 * n, 0.10 + 0.40 * n) for n in range(9)]
BAYS = [(0.11 + 0.40 * n, 0.40 + 0.40 * n) for n in range(8)]
BEAM_Z = [(0.61, 0.70), (0.91, 1.00)]

GATE_X = [(1.51, 1.60), (2.41, 2.50)]
DAM_X = (3.51, 3.60)
GATE_Z = (0.01, 1.00)


def gate_slab(x0, x1):
    """25 boxes: 9 piers full height + 8 bays x 2 beam levels."""
    boxes = [[x0, y0, GATE_Z[0], x1, y1, GATE_Z[1]] for y0, y1 in PIERS]
    for y0, y1 in BAYS:
        for z0, z1 in BEAM_Z:
            boxes.append([x0, y0, z0, x1, y1, z1])
    return boxes


def main():
    parts = [
        {"name": "water", "role": "fluid",
         "geom": {"type": "box", "min": WATER[0:3], "max": WATER[3:6]}},
        {"name": "gate_upstream", "role": "elastic",
         "geom": {"type": "boxes", "boxes": gate_slab(*GATE_X[0])}},
        {"name": "gate_downstream", "role": "elastic",
         "geom": {"type": "boxes", "boxes": gate_slab(*GATE_X[1])}},
        {"name": "dam", "role": "elastic",
         "geom": {"type": "box", "min": [DAM_X[0], 0.01, 0.01],
                  "max": [DAM_X[1], 3.30, 1.00]}},
        # the tank: four rigid slabs, four particle layers thick, outside the
        # water column (exactly the Monaghan layers 1.cpp writes by hand)
        {"name": "wall_floor", "role": "rigid",
         "geom": {"type": "box", "min": [-0.03, -0.03, -0.03],
                  "max": [3.60, 3.34, 0.00]}},
        {"name": "wall_upstream", "role": "rigid",
         "geom": {"type": "box", "min": [-0.03, -0.03, 0.01],
                  "max": [3.60, 0.00, 1.00]}},
        {"name": "wall_downstream", "role": "rigid",
         "geom": {"type": "box", "min": [-0.03, 3.31, 0.01],
                  "max": [3.60, 3.34, 1.00]}},
        {"name": "wall_left", "role": "rigid",
         "geom": {"type": "box", "min": [-0.03, 0.01, 0.01],
                  "max": [0.00, 3.30, 1.00]}},
    ]
    case = {"spacing": DX, "domain": DOMAIN, "parts": parts}

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "8hole_gate.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=1)
    nbox = sum(len(p["geom"].get("boxes", [p["geom"]])) for p in parts)
    print("wrote %s: %d parts, %d boxes" % (out, len(parts), nbox))
    print("expected particles: water %d, gate %d x2, dam %d, walls %d"
          % (70 * 330 * 95, 9 * 10 * 100 * 10 + 8 * 2 * 30 * 10 * 10,
             10 * 330 * 100, 4 * 364 * 338 + 2 * 4 * 364 * 100 + 4 * 330 * 100))


if __name__ == "__main__":
    main()
