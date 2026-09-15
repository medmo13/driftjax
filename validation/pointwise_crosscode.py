"""Pointwise cross-code error norms vs bundled dPV reference IVs (Obs 6).

Compares live DriftJax sweeps against examples/resources/deltapv_{ex1,ex2}_iv.npz
on a common bias grid: Linf, L2, relative L2 of J(V), plus Jsc/Voc/Pmax.
State fields (n/p/phi) are unavailable in the bundled references, so no
state-field norms are claimed. Writes JSON to docs/paper/records/.
"""
# ruff: noqa: E402 -- sys.path bootstrap precedes package imports

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import driftjax as dj

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESOURCES = Path(__file__).resolve().parent.parent / "examples" / "resources"


def matched_ex1(n):
    m = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=2e4
    )
    return dj.Device(
        layers=[(1e-4, m, 1e17), (1e-4, m, -1e17)],
        n_points=n,
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )


def matched_ex2(n):
    CdS = dj.material(
        Nc=2.2e18,
        Nv=1.8e19,
        Eg=2.4,
        eps=10,
        Et=0,
        mn=100,
        mp=25,
        tn=1e-8,
        tp=1e-13,
        Chi=4.0,
        A=1e4,
    )
    CdTe = dj.material(
        Nc=8e17,
        Nv=1.8e19,
        Eg=1.5,
        eps=9.4,
        Et=0,
        mn=320,
        mp=40,
        tn=5e-9,
        tp=5e-9,
        Chi=3.9,
        A=1e4,
    )
    return dj.Device(
        layers=[(2.5e-6, CdS, 1e17), (4e-4, CdTe, -1e15)],
        n_points=n,
        Snl=1.16e7,
        Snr=1.16e7,
        Spl=1.16e7,
        Spr=1e7,
    )


def main():
    out = {}
    for tag, builder in (("ex1", matched_ex1), ("ex2", matched_ex2)):
        ref = np.load(RESOURCES / f"deltapv_{tag}_iv.npz")
        rv, rj = np.asarray(ref["v"], dtype=float), np.asarray(ref["j"], dtype=float)
        t0 = time.perf_counter()
        sol = dj.simulate(builder(500), dj.Sweep(vmax=float(rv.max()), n_steps=25))
        wall = time.perf_counter() - t0
        v = np.asarray(sol.voltages, dtype=float)
        j = np.asarray(sol.current, dtype=float)
        lo, hi = max(rv.min(), v.min()), min(rv.max(), v.max())
        g = np.linspace(lo, hi, 200)
        ji = np.interp(g, v, j)
        ri = np.interp(g, rv, rj)
        d = np.abs(ji - ri)
        out[tag] = {
            "grid": [float(lo), float(hi), 200],
            "linf_dJ": float(d.max()),
            "l2_dJ": float(np.sqrt(np.trapezoid(d**2, g))),
            "rel_l2": float(
                np.sqrt(np.trapezoid(d**2, g) + 1e-60) / (np.sqrt(np.trapezoid(ri**2, g)) + 1e-60)
            ),
            "jsc_driftjax": float(abs(j[np.argmin(np.abs(v))])),
            "jsc_ref": float(abs(rj[np.argmin(np.abs(rv))])),
            "pce_pct_driftjax": float(sol.efficiency) * 100.0,
            "voc_driftjax": float(sol.voc),
            "wall_s": wall,
        }
        print(
            tag,
            {k: round(vv, 6) if isinstance(vv, float) else vv for k, vv in out[tag].items()},
            flush=True,
        )
    dest = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "paper"
        / "records"
        / "pointwise_crosscode.json"
    )
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)
    main()
