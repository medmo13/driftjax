"""Scaling + adjoint-vs-FD break-even measurements (referee Obs 2, 11).

Part A — forward scaling: warm single-bias Newton solve at
N = 50, 100, 200, 400, 800, 1600 (tol 1e-10, <=30 steps, mean of 5;
N=1600 mean of 3), plus warm sweep/gradient at selected N.

Part B — break-even: on a homojunction device (N=100, 10 biases),
wall time of the adjoint gradient vs central-FD gradients at
P = 2, 4, 8 parameters (2P sweeps each). Reports the measured
cost ratio and the break-even dimension.

Writes JSON to docs/paper/records/scaling_breakeven.json.
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

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import driftjax as dj
from driftjax.science.contacts import boundary_bias
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq, solve_newton


def make_device(n, mat=None):
    if mat is None:
        mat = dj.material(
            Chi=3.9,
            Eg=1.5,
            eps=9.4,
            Nc=8e17,
            Nv=1.8e19,
            mn=100,
            mp=100,
            tn=1e-8,
            tp=1e-8,
            A=1e4,
        )
    return dj.Device(
        n_points=n,
        layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )


def timed_newton(n, n_measure=5):
    dev = make_device(n)
    ls = spectrum(normalize=False)
    cell = dj.simulator.init_cell(dev.design(), ls, alpha_mode="beer-lambert")
    from driftjax.science.contacts import boundary_eq

    pot0 = solve_eq(cell, boundary_eq(cell), equilibrium_guess(cell).phi)
    bound = boundary_bias(cell, 0.5 / dj.units.energy)
    solve_newton(cell, bound, pot0, tol=1e-10, max_steps=30)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        solve_newton(cell, bound, pot0, tol=1e-10, max_steps=30)
    return (time.perf_counter() - t0) / n_measure * 1000.0


def timed_sweep(n, n_steps=10):
    dev = make_device(n)
    dj.simulate(dev, dj.Sweep(vmax=1.0, n_steps=n_steps))
    t0 = time.perf_counter()
    dj.simulate(dev, dj.Sweep(vmax=1.0, n_steps=n_steps))
    return (time.perf_counter() - t0) * 1000.0


def timed_grad(n, n_steps=10):
    dev = make_device(n)

    def eff(d):
        return dj.simulate(d, dj.Sweep(vmax=1.0, n_steps=n_steps)).efficiency

    eqx.filter_grad(eff)(dev)
    t0 = time.perf_counter()
    eqx.filter_grad(eff)(dev)
    return (time.perf_counter() - t0) * 1000.0


def breakeven(n=100, n_steps=10, plist=(2, 4, 8)):
    """Adjoint wall time vs central-FD wall time at P params.

    Parameters: first P entries of [logMobility_n, logMobility_p,
    logLifetime_n, logLifetime_p, logNd, logNa, Eg, Chi] style builder
    vector on a fixed homojunction template.
    """
    base_mat = dict(eps=9.4, Nc=8e17, Nv=1.8e19, A=1e4)

    def device_from_x(x):
        # x = [log10(mn), log10(mp), log10(tn), log10(tp),
        #      log10(Nd), log10(Na), Eg, Chi]
        m = dj.material(
            mn=10.0 ** x[0],
            mp=10.0 ** x[1],
            tn=10.0 ** x[2],
            tp=10.0 ** x[3],
            Eg=x[6],
            Chi=x[7],
            **base_mat,
        )
        return dj.Device(
            n_points=n,
            layers=[(1e-4, m, 10.0 ** x[4]), (1e-4, m, -(10.0 ** x[5]))],
            Snl=1e7,
            Snr=0,
            Spl=0,
            Spr=1e7,
        )

    x0_full = [2.0, 2.0, -8.0, -8.0, 17.0, 17.0, 1.5, 3.9]

    def eff(x, p):
        full = jnp.asarray(x0_full).at[:p].set(jnp.asarray(x))
        return dj.simulate(
            device_from_x(full),
            dj.Sweep(vmax=1.0, n_steps=n_steps),
        ).efficiency

    rows = []
    for p in plist:
        xp = np.asarray(x0_full)[:p]
        # Per-coordinate steps: relative 1e-4 (absolute h fails on O(1)
        # params like Eg/Chi while suiting log-params).
        hvec = 1e-4 * np.maximum(1.0, np.abs(xp))
        # adjoint: warmup call, then timed call
        jax.grad(lambda xx, pp=p: eff(xx, pp))(jnp.asarray(xp))
        t0 = time.perf_counter()
        g_adj = np.asarray(jax.grad(lambda xx, pp=p: eff(xx, pp))(jnp.asarray(xp)))  # noqa: B023
        t_adj = (time.perf_counter() - t0) * 1000.0
        # central FD: 2P sweeps (warm already via adjoint compilation)
        t0 = time.perf_counter()
        g_fd = np.zeros(p)
        for a in range(p):
            e = np.zeros(p)
            e[a] = 1.0
            ha = hvec[a]
            g_fd[a] = (float(eff(xp + ha * e, p)) - float(eff(xp - ha * e, p))) / (2 * ha)
        t_fd = (time.perf_counter() - t0) * 1000.0
        # Mixed metric: relative where |g| resolves, absolute floor where
        # the sensitivity is ~0 (e.g. Chi here: adj 2e-12, FD exactly 0;
        # pure relative error reports a spurious 1.0).
        scale = max(float(np.max(np.abs(g_adj))), 1e-12)
        rel = float(np.max(np.abs(g_fd - g_adj) / np.maximum(np.abs(g_adj), 1e-3 * scale)))
        rows.append(
            {
                "P": p,
                "t_adj_ms": t_adj,
                "t_fd_ms": t_fd,
                "ratio_fd_over_adj": t_fd / max(t_adj, 1e-9),
                "max_rel_fd_err": rel,
            }
        )
    return rows


def main():
    import os

    out = {"scaling_newton_ms": {}, "sweep_ms": {}, "grad_ms": {}}
    ns = (50, 100, 200, 400, 800, 1600)
    if os.environ.get("BREAKEVEN_ONLY"):
        with open(
            Path(__file__).resolve().parent.parent
            / "docs"
            / "paper"
            / "records"
            / "scaling_breakeven.json"
        ) as fh:
            prev = json.load(fh)
        out.update({k: prev[k] for k in ("scaling_newton_ms", "sweep_ms", "grad_ms")})
        ns = ()
    for n in ns:
        out["scaling_newton_ms"][str(n)] = timed_newton(n, n_measure=3 if n >= 800 else 5)
        print(f"newton N={n}: {out['scaling_newton_ms'][str(n)]:.1f} ms", flush=True)
    for n in (100, 200, 400):
        out["sweep_ms"][str(n)] = timed_sweep(n)
        out["grad_ms"][str(n)] = timed_grad(n)
        print(
            f"N={n} sweep={out['sweep_ms'][str(n)]:.1f} grad={out['grad_ms'][str(n)]:.1f} ms",
            flush=True,
        )
    out["breakeven"] = breakeven()
    for r in out["breakeven"]:
        print(r, flush=True)
    dest = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "paper"
        / "records"
        / "scaling_breakeven.json"
    )
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
