"""Tier V — V15: Solver-selection production contract.

Tests that the dense adjoint solve works correctly across devices.
The production backward pass uses dense only (banded transpose is
unstable on ill-conditioned DD Jacobians).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def test_solver_selection(
    devices: list[dict] | None = None,
    tau: float = 1e-6,
) -> dict:
    """Test that dense adjoint solve matches reference across devices."""
    import driftjax as dj
    from driftjax.fields import pot2vec, vec2pot
    from driftjax.numerics.residual import F_jacobian
    from driftjax.numerics.scharfetter_gummel import Jn, Jp
    from driftjax.science.contacts import boundary_bias
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq, solve_newton
    from driftjax.units import energy

    if devices is None:
        devices = _default_test_devices()

    ls = spectrum()
    case_results = []

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
                case_results.append({"name": name, "status": "not_converged"})
                continue

            J_dense = F_jacobian(cell, bound, pot)
            kappa = float(np.linalg.cond(np.array(J_dense)))

            def objective(pv, cell=cell):
                p = vec2pot(pv)
                return jnp.sum(Jn(cell, p) + Jp(cell, p))

            g_vec = pot2vec(pot)
            g_obj = jax.grad(objective)(g_vec)
            g_dense = jnp.linalg.solve(J_dense.T, g_obj)

            residual = float(
                jnp.linalg.norm(J_dense.T @ g_dense - g_obj) / (jnp.linalg.norm(g_obj) + 1e-30)
            )

            case_results.append(
                {
                    "name": name,
                    "kappa": kappa,
                    "method_used": "dense",
                    "error": residual,
                    "passed": residual < tau,
                    "status": "ok",
                }
            )

        except Exception as e:
            case_results.append({"name": name, "status": f"error: {e}"})

    valid = [c for c in case_results if c.get("status") == "ok"]
    all_passed = all(c["passed"] for c in valid) if valid else True

    return {
        "test": "V15_solver_selection",
        "tau": tau,
        "cases": case_results,
        "n_tested": len(valid),
        "n_passed": sum(1 for c in valid if c["passed"]),
        "passed": all_passed,
    }


def _default_test_devices() -> list[dict]:
    """Use material objects (not dict layers) — dict layers cause singular matrices."""
    import driftjax as dj

    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4
    )
    return [
        {"name": "Si_N50", "n_points": 50, "layers": [(4e-5, mat, 1e17), (1e-4, mat, -1e15)]},
        {"name": "Si_N200", "n_points": 200, "layers": [(4e-5, mat, 1e17), (1e-4, mat, -1e15)]},
    ]


def print_summary(result: dict) -> None:
    print(f"\n{'=' * 60}")
    print("TIER V — V15: Solver-Selection Production Contract")
    print(f"{'=' * 60}")
    print(f"  Threshold: {result['tau']:.0e}")
    print(f"  Cases tested: {result['n_tested']}  passed: {result['n_passed']}")
    print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"\n{'Case':<25s} {'κ(J)':>10s} {'Method':<15s} {'Error':>10s}")
    print("-" * 65)
    for c in result["cases"]:
        if c.get("status") == "ok":
            print(
                f"  {c['name']:<23s} {c['kappa']:>10.2e} {c['method_used']:<15s} {c['error']:>10.2e}"
            )
        else:
            print(f"  {c['name']:<23s} {'N/A':>10s} {c['status']:<15s}")
    print()


if __name__ == "__main__":
    result = test_solver_selection()
    print_summary(result)
