"""Extended DGSM: standardized coordinates, N=256, bootstrap rank CIs (Obs 10).

Same 4-parameter homojunction problem as examples/research/14_dgsm_global.py
but (a) standardized dimensionless coordinates z in [-1,1]^4 (bounds-mapped,
so linear-mobility and log-lifetime coordinates are comparable), (b) 256
Sobol samples, (c) 200 bootstrap resamples for rank confidence, (d) Spearman
rank correlation vs the published 32-sample ordering.

Writes JSON to docs/paper/records/dgsm_extended.json.
"""
# ruff: noqa: E402 -- sys.path bootstrap precedes package imports

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import driftjax as dj
from driftjax import Newton, Sweep, simulate

NAMES = ["mu_n", "mu_p", "log10_tau_n", "log10_tau_p"]
LO = np.array([50.0, 50.0, -9.0, -9.0])
HI = np.array([200.0, 200.0, -7.0, -7.0])
BASE = dict(
    Chi=3.9,
    Eg=1.5,
    eps=9.4,
    Nc=8e17,
    Nv=1.8e19,
    Et=0,
    A=2e4,
    mn=100.0,
    mp=100.0,
    tn=1e-8,
    tp=1e-8,
)
NPOINTS, NSTEPS, N, NBOOT = 500, 15, 256, 200

mat0 = dj.material(**BASE)


def eff_of_z(z):
    p = jnp.asarray(LO) + (jnp.asarray(z) + 1.0) / 2.0 * jnp.asarray(HI - LO)
    mv = eqx.tree_at(
        lambda mm: (mm.mn, mm.mp, mm.tn, mm.tp), mat0, (p[0], p[1], 10 ** p[2], 10 ** p[3])
    )
    dev = dj.Device(
        n_points=NPOINTS,
        layers=[(1e-4, mv, 1e17), (1e-4, mv, -1e17)],
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )
    return simulate(dev, Sweep(vmax=1.1, n_steps=NSTEPS), solver=Newton()).efficiency


def main():
    jax.config.update("jax_enable_x64", True)
    from scipy.stats.qmc import Sobol

    grad_z = jax.jacobian(eff_of_z)
    sob = Sobol(d=4, seed=12345)
    Z = 2.0 * sob.random_base2(m=int(np.log2(N))) - 1.0
    t0 = time.perf_counter()
    G = np.asarray([grad_z(jnp.asarray(z)) for z in Z])
    grad_wall = time.perf_counter() - t0
    dgsm = np.mean(G**2, axis=0)
    order = list(np.argsort(-dgsm))
    # bootstrap rank CIs
    rng = np.random.default_rng(7)
    top1 = np.zeros(4, dtype=int)
    for _ in range(NBOOT):
        idx = rng.integers(0, N, N)
        db = np.mean(G[idx] ** 2, axis=0)
        top1[int(np.argmax(db))] += 1
    # Spearman vs published 32-sample ordering (hole lifetime dominant).
    # Published order (ex14): tau_p first; recompute rank correlation on names.
    ref_order = ["log10_tau_p", "log10_tau_n", "mu_n", "mu_p"]
    new_order = [NAMES[i] for i in order]
    from scipy.stats import spearmanr

    rho = float(
        spearmanr(
            [ref_order.index(n) for n in NAMES], [new_order.index(n) for n in NAMES]
        ).statistic
    )
    out = {
        "N": N,
        "nboot": NBOOT,
        "coords": "standardized z in [-1,1]^4",
        "dgsm": dgsm.tolist(),
        "order": new_order,
        "top1_bootstrap_fraction": (top1 / NBOOT).tolist(),
        "spearman_vs_published32": rho,
        "grad_wall_s": grad_wall,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "dgsm"}, indent=1), flush=True)
    dest = (
        Path(__file__).resolve().parent.parent / "docs" / "paper" / "records" / "dgsm_extended.json"
    )
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
