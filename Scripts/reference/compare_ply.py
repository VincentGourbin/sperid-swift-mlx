#!/usr/bin/env python3
"""Compare two 3DGS .ply files property by property (same vertex order). Prints `PLY OK|FAILED …`.

Usage: .venv-ref/bin/python -I Scripts/reference/compare_ply.py a.ply b.ply [--tol 1e-5]
"""
import argparse
import sys

import numpy as np
from plyfile import PlyData


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--tol", type=float, default=1e-5)
    a = p.parse_args()
    va, vb = PlyData.read(a.a)["vertex"], PlyData.read(a.b)["vertex"]
    names_a = [q.name for q in va.properties]
    names_b = [q.name for q in vb.properties]
    if names_a != names_b or va.count != vb.count:
        print(f"PLY FAILED layout: {names_a} × {va.count} vs {names_b} × {vb.count}")
        sys.exit(1)
    worst = {n: float(np.max(np.abs(va[n].astype(np.float64) - vb[n].astype(np.float64)))) for n in names_a}
    bad = {n: d for n, d in worst.items() if d > a.tol}
    status = "FAILED" if bad else "OK"
    print(f"PLY {status} {va.count} vertices, max |Δ| {max(worst.values()):.3g} (tol {a.tol:g})"
          + (f", over tolerance: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
