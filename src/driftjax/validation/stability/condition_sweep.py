"""Tier V — Adjoint error versus Jacobian conditioning (signature experiment).

Tests that the dense adjoint solve is accurate across conditioning regimes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _build_dense_jacobian(A, B, C, n):
    """Build dense Jacobian from block-tridiagonal (A, B, C)."""
    J = jnp.zeros((3 * n, 3 * n))
    for i in range(n):
        J = J.at[3 * i : 3 * i + 3, 3 * i : 3 * i + 3].set(A[i])
    for i in range(n - 1):
        J = J.at[3 * i : 3 * i + 3, 3 * (i + 1) : 3 * (i + 1) + 3].set(B[i])
        J = J.at[3 * (i + 1) : 3 * (i + 1) + 3, 3 * i : 3 * i + 3].set(C[i + 1])
    return J


def _adjoint_error(J_dense, n):
    """Compute dense adjoint residual."""
    g = jax.random.normal(jax.random.key(42), (3 * n,))
    g = g / (jnp.linalg.norm(g) + 1e-30)

    t0 = time.perf_counter()
    lam_dense = jnp.linalg.solve(J_dense.T, g)
    t_dense = time.perf_counter() - t0

    r_dense = float(jnp.linalg.norm(J_dense.T @ lam_dense - g) / (jnp.linalg.norm(g) + 1e-30))

    return r_dense, t_dense


def run_condition_sweep(devices=None) -> dict:
    """Run the condition-number sweep experiment."""
    import driftjax as dj
    from driftjax.numerics.analytic_jacobian import banded_jacobian
    from driftjax.science.contacts import boundary_bias
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq, solve_newton
    from driftjax.units import energy

    if devices is None:
        devices = _default_device_spectrum()

    ls = spectrum()
    cases = []

    for dev_spec in devices:
        name = dev_spec["name"]
        try:
            dev = dj.Device(
                n_points=dev_spec["n_points"],
                layers=dev_spec["layers"],
                Snl=dev_spec.get("Snl", 1e7),
                Snr=dev_spec.get("Snr", 0),
                Spl=dev_spec.get("Spl", 0),
                Spr=dev_spec.get("Spr", 1e7),
            )
            cell = init_cell(dev.design(), ls)
            bound = boundary_bias(cell, 0.4 / energy)
            pot_eq = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)
            pot, info = solve_newton(cell, bound, pot_eq, tol=1e-10)

            if not info.get("converged", True):
                cases.append({"name": name, "status": "not_converged", "kappa": None})
                continue

            n = cell.x.shape[0]
            A, B, C = banded_jacobian(cell, bound, pot)
            J_dense = _build_dense_jacobian(A, B, C, n)
            kappa = float(np.linalg.cond(np.array(J_dense)))

            r_dense, t_dense = _adjoint_error(J_dense, n)

            if r_dense < 1e-6:
                status = "dense_ok"
            else:
                status = "dense_failed"

            cases.append(
                {
                    "name": name,
                    "status": status,
                    "n_points": n,
                    "kappa": kappa,
                    "r_dense": r_dense,
                    "t_dense_s": t_dense,
                }
            )

        except Exception as e:
            cases.append({"name": name, "status": f"error: {e}", "kappa": None})

    valid = [c for c in cases if c.get("kappa") is not None]
    n_ok = sum(1 for c in valid if c["status"] == "dense_ok")

    return {
        "cases": cases,
        "summary": {"total": len(valid), "dense_ok": n_ok, "dense_failed": len(valid) - n_ok},
    }


def _default_device_spectrum():
    """Use material objects (not dict layers)."""
    import driftjax as dj

    mat_si = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4
    )
    mat_psc = dj.material(
        Chi=3.9, Eg=1.59, eps=9.4, Nc=2.2e18, Nv=1e19, mn=100, mp=100, tn=1e-7, tp=1e-7, A=1e4
    )
    return [
        {"name": "Si_N50", "n_points": 50, "layers": [(4e-5, mat_si, 1e17), (1e-4, mat_si, -1e15)]},
        {
            "name": "PSC_N50",
            "n_points": 50,
            "layers": [
                (3e-5, mat_psc, 1e16),
                (4e-5, mat_psc, 0),
                (5e-5, mat_psc, -1e16),
            ],
        },
    ]


def print_summary(result):
    s = result["summary"]
    print(f"\n{'=' * 70}")
    print("ADJOINT ERROR vs JACOBIAN CONDITIONING")
    print(f"{'=' * 70}")
    print(f"  Cases: {s['total']}  Dense OK: {s['dense_ok']}  Dense failed: {s['dense_failed']}")
    print(f"\n{'Case':<25s} {'κ(J)':>10s} {'r_dense':>10s} {'Status':<15s}")
    print("-" * 65)
    for c in result["cases"]:
        if c.get("kappa") is None:
            print(f"  {c['name']:<23s} {'N/A':>10s} {'N/A':>10s} {c['status']:<15s}")
        else:
            print(
                f"  {c['name']:<23s} {c['kappa']:>10.2e} {c['r_dense']:>10.2e} {c['status']:<15s}"
            )
    print()


def save_record(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    result = run_condition_sweep()
    print_summary(result)
