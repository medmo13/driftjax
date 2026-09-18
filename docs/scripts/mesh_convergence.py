#!/usr/bin/env python
"""Mesh convergence study for DriftJax native GE backend."""
import json, os, numpy as np, jax.numpy as jnp
import driftjax as dj

def make_homojunction(n_points):
    m = dj.material(Eg=1.55, Chi=3.9, eps=12.0, Nc=2.2e19, Nv=2.2e19, mn=1000.0, mp=300.0, tn=1e-6, tp=1e-6, A=3e4)
    return dj.Device(n_points=n_points, layers=[(5e-6, m, 1e18), (4e-4, m, 1e14), (5e-6, m, 1e18)], Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7)

def run():
    protocol = dj.Sweep(vmax=1.1, n_steps=10)
    n_ref = 500
    cell_ref = make_homojunction(n_ref)
    sol_ref = dj.simulate(cell_ref, protocol, solver=dj.Newton(backend="native_ge_eq"))
    jsc_ref = float(sol_ref.jsc)
    print(f"Reference (N={n_ref}): Jsc={jsc_ref:.6f} PCE={float(sol_ref.efficiency):.4%}")
    results = []
    for n in [50, 100, 150, 200, 250, 300, 400]:
        cell = make_homojunction(n)
        sol = dj.simulate(cell, protocol, solver=dj.Newton(backend="native_ge_eq"))
        err = abs(float(sol.jsc) - jsc_ref) / abs(jsc_ref)
        results.append({"n_points": n, "jsc": float(sol.jsc), "pce": float(sol.efficiency), "err_jsc": err, "steps": int(sol.newton_iters)})
        print(f"N={n}: Jsc={float(sol.jsc):.6f} err={err:.2e} steps={sol.newton_iters}")
    with open("docs/paper/records/mesh_convergence.json", "w") as f:
        json.dump({"n_ref": n_ref, "jsc_ref": jsc_ref, "results": results}, f, indent=2)

if __name__ == "__main__":
    run()