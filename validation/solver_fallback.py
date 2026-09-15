"""Solver-fallback provenance record (R2/R3).

Runs the canonical singular 3-layer perovskite (N=15, 5 biases) and the
well-conditioned homojunction control through simulate(), and records the
per-bias truncated-SVD fallback flags (Solution.fallback_used) alongside
convergence diagnostics. Regenerates docs/paper/records/solver_fallback.json.

Expected outcome: 3-layer reports fallback_used=[True]*5 (dgbsv raises
singular-matrix on every bias; lstsq steps; sweep converges to ~1e-8);
homojunction reports [False]*5 (clean pivoted-banded solves).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import driftjax as dj

RECORD = (
    Path(__file__).resolve().parent.parent / "docs" / "paper" / "records" / "solver_fallback.json"
)


def _three_layer(n_points=15):
    mt = dj.material(Eg=1.6, Chi=3.9, eps=20.0, Nc=1e18, Nv=1e18, mn=100.0, mp=100.0, A=2e4)
    mat = dj.material(Eg=1.4, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18, mn=130.0, mp=160.0, A=2e4)
    return dj.Device(
        n_points=n_points,
        layers=[(2e-5, mt, 1e18), (6e-5, mat, -1e18), (2e-5, mat, 1e18)],
        Snl=1e7,
        Snr=0.0,
        Spl=0.0,
        Spr=1e7,
    )


def _homojunction(n_points=15):
    mat = dj.material(Eg=1.4, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18, mn=130.0, mp=160.0, A=2e4)
    return dj.Device(
        n_points=n_points,
        layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
        Snl=1e7,
        Snr=0.0,
        Spl=0.0,
        Spr=1e7,
    )


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    t0 = time.time()
    out = {"environment": {"dtype": "float64", "x64": True}, "cases": {}}
    for name, dev in (
        ("three_layer_perovskite_N15", _three_layer()),
        ("homojunction_N15", _homojunction()),
    ):
        sol = dj.simulate(dev, dj.Sweep(n_steps=5, vmax=1.0))
        out["cases"][name] = {
            "converged": bool(sol.converged),
            "efficiency": float(sol.efficiency),
            "max_residual": float(sol.max_residual),
            "fallback_used": [bool(x) if x is not None else None for x in list(sol.fallback_used)],
            "voltages_V": [float(v) for v in sol.voltages],
        }
    out["wall_s"] = round(time.time() - t0, 1)
    RECORD.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
